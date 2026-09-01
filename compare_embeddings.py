"""Compare two jTrans embedding pkls function-by-function.

This is the check that decides whether eval_save_batched.py may replace eval_save.py.
Batching changes matmul reduction order, and --padding longest changes the tensor
shapes fed to the model; both are *argued* to be harmless, and this measures whether
they actually are.

Entries are matched on (proj, funcname, opt_key), not on list position, so the two
pkls need not have been produced from the same dataset ordering or the same subset --
a --limit run compares fine against a full one.

Usage:
    python compare_embeddings.py reference.pkl candidate.pkl

What to look for:
    max abs diff around 1e-6 and min cosine ~1.0 means the two are the same
    computation. Cosines below ~0.9999 on more than a stray function mean the change
    is not equivalence-preserving and the pkl should not be used.
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np
import torch


def load_keyed(path):
    """Load a pkl into {(proj, funcname, opt_key): 1-D float array}."""
    with open(path, "rb") as f:
        ebds = pickle.load(f)

    out = {}
    skipped = 0
    duplicates = 0
    for entry in ebds:
        proj = entry.get("proj")
        funcname = entry.get("funcname")
        for opt_key, value in entry.items():
            if opt_key in ("proj", "funcname"):
                continue
            if isinstance(value, torch.Tensor):
                vec = value.detach().cpu().numpy().reshape(-1)
            elif isinstance(value, np.ndarray):
                vec = value.reshape(-1)
            else:
                # Still an integer index into datas -- this variant was never embedded.
                skipped += 1
                continue
            key = (proj, funcname, opt_key)
            # A collision would silently drop one of the two vectors and make the
            # comparison compare the wrong pair of functions, so say so rather than
            # letting the dict quietly absorb it.
            if key in out:
                duplicates += 1
            out[key] = vec
    if duplicates:
        print(f"WARNING: {path} has {duplicates} duplicate (proj, funcname, opt_key) "
              f"keys -- comparison results for those functions are meaningless.")
    return out, skipped


def main():
    parser = argparse.ArgumentParser(description="Compare two jTrans embedding pkls")
    parser.add_argument("reference", help="Known-good pkl, e.g. from eval_save.py")
    parser.add_argument("candidate", help="Pkl to check, e.g. from eval_save_batched.py")
    parser.add_argument(
        "--cos_threshold", type=float, default=0.9999,
        help="Report any function whose cosine similarity falls below this.",
    )
    parser.add_argument(
        "--show", type=int, default=10,
        help="How many of the worst-matching functions to list.",
    )
    args = parser.parse_args()

    ref, ref_skipped = load_keyed(args.reference)
    cand, cand_skipped = load_keyed(args.candidate)
    print(f"reference: {len(ref)} embedded variants ({ref_skipped} unembedded)")
    print(f"candidate: {len(cand)} embedded variants ({cand_skipped} unembedded)")

    # Sort by the string form of each key component: proj/funcname come from the jTrans
    # dataset pickle, not a CSV, so their types are whatever was stored there -- a
    # numeric-looking funcname can arrive as int and make a plain sorted() raise
    # TypeError comparing int to str. Only affects output ordering, never the comparison.
    shared = sorted(set(ref) & set(cand), key=lambda k: tuple(str(p) for p in k))
    if not shared:
        raise SystemExit(
            "No (proj, funcname, opt_key) keys in common -- the two pkls do not "
            "describe the same functions, so there is nothing to compare."
        )
    print(f"comparing {len(shared)} variants present in both "
          f"({len(ref) - len(shared)} ref-only, {len(cand) - len(shared)} cand-only)")

    a = np.stack([ref[k] for k in shared]).astype(np.float64)
    b = np.stack([cand[k] for k in shared]).astype(np.float64)

    abs_diff = np.abs(a - b)
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))

    print()
    print(f"max abs diff   : {abs_diff.max():.3e}")
    print(f"mean abs diff  : {abs_diff.mean():.3e}")
    print(f"min cosine     : {cos.min():.8f}")
    print(f"mean cosine    : {cos.mean():.8f}")

    bad = np.where(cos < args.cos_threshold)[0]
    print(f"below {args.cos_threshold}: {len(bad)} / {len(shared)} "
          f"({100 * len(bad) / len(shared):.3f}%)")

    if len(bad):
        worst = bad[np.argsort(cos[bad])][:args.show]
        print(f"\nworst {len(worst)}:")
        for i in worst:
            proj, funcname, opt_key = shared[i]
            print(f"  cos={cos[i]:.8f}  maxdiff={abs_diff[i].max():.3e}  "
                  f"{proj}:{funcname}:{opt_key}")
        raise SystemExit(1)

    print("\nEquivalent within threshold.")


if __name__ == "__main__":
    main()
