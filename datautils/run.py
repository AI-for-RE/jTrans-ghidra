#!/usr/bin/env python3
"""Run feature extraction over a directory of binaries, then build the pair data.

Port of the IDA Pro driver.  Where the original spawned ``idat64 -S process.py`` per
binary, each worker here runs ``process.py`` in its own Python process -- one Ghidra
JVM per binary, so a failed or runaway analysis cannot take down the whole run.

Each binary is stripped into ``dataset_strip/`` first: Ghidra must not see the symbol
table, or call targets would render as ``call strcpy`` instead of ``call sub_401000``
and leak the very labels the model is meant to learn.  The ground truth still comes
from the unstripped copy, which ``process.py`` reads with pyelftools.
"""
import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Pool

from util.pairdata import pairdata

dataset_dir = './dataset/'
strip_path = "./dataset_strip/"
script_path = "./process.py"
SAVE_ROOT = "./extract"
PROJECT_ROOT = "./ghidra_proj"
LOG_ROOT = "./log"
STRIP = os.environ.get('STRIP', 'strip')  # GNU binutils strip, or e.g. llvm-strip


def getTarget(path, prefixfilter=None):
    target = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if prefixfilter is None:
                target.append(os.path.join(root, file))
            else:
                for prefix in prefixfilter:
                    if file.startswith(prefix):
                        target.append(os.path.join(root, file))
    return target


def strip_binary(target, output):
    """Strip ``target`` into ``output``; returns False if the strip tool refused."""
    cmd = [STRIP, '-s', target, '-o', output]
    print(' '.join(cmd))
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except OSError as err:
        print(f"[!] cannot run {STRIP}: {err}")
        return False
    if result.returncode != 0:
        print(f"[!] {STRIP} failed on {target}: {result.stdout.decode(errors='replace').strip()}")
        return False
    return True


def process_one(job):
    """Strip one binary and extract its features with process.py."""
    target, no_strip = job
    filename = os.path.basename(target)

    if no_strip:
        ghidra_input = target
    else:
        ghidra_input = os.path.join(strip_path, filename + '.strip')
        if not strip_binary(target, ghidra_input):
            return filename, -1

    cmd = [sys.executable, script_path, ghidra_input,
           '--unstrip-path', target,
           '--dataroot', dataset_dir,
           '--saveroot', SAVE_ROOT,
           '--project-location', PROJECT_ROOT,
           '--project-name', filename]
    print(' '.join(cmd))
    with open(os.path.join(LOG_ROOT, filename + '.log'), 'w') as log:
        returncode = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    return filename, returncode


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('-j', '--jobs', type=int, default=8,
                        help='number of binaries analyzed in parallel')
    parser.add_argument('--prefixfilter', nargs='*', default=None,
                        help='only process binaries whose name starts with one of these')
    parser.add_argument('--no-strip', action='store_true',
                        help='analyze the binaries as-is (symbol names will leak into the asm)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    start = time.time()

    for directory in (strip_path, SAVE_ROOT, PROJECT_ROOT, LOG_ROOT):
        os.makedirs(directory, exist_ok=True)

    target_list = getTarget(dataset_dir, args.prefixfilter)
    print(f'[*] {len(target_list)} binaries to process')

    failed = []
    with Pool(processes=args.jobs) as pool:
        jobs = [(target, args.no_strip) for target in target_list]
        for filename, returncode in pool.imap_unordered(process_one, jobs):
            if returncode != 0:
                failed.append(filename)
                print(f'[!] {filename} failed (exit {returncode}), see {LOG_ROOT}/{filename}.log')

    print('[*] Features Extracting Done')
    if failed:
        print(f'[!] {len(failed)} of {len(target_list)} binaries failed: {", ".join(failed)}')
    pairdata(SAVE_ROOT)
    end = time.time()
    print(f"[*] Time Cost: {end - start} seconds")
