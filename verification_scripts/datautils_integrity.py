"""Does ebds[i][opt_key] index the RIGHT entry of datas[i]?

The O0 failure signature (O0|O0 healthy, O0|anything-else at chance) is exactly what a
*consistent* mis-mapping of O0 variants produces. This checks the mapping directly,
upstream of the model: regenerate each variant's token string straight from the raw
per-binary _extract.pkl, then ask whether datas[i][ebds[i][o]] equals the string for
variant o -- or for some OTHER variant, or for nothing at all.
"""
import glob, os, pickle, random, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath('.'))
from data import load_paired_data, gen_funcstr

random.seed(0)

print("loading paired data (this is the slow part) ...", flush=True)
functions, ebds = load_paired_data(
    datapath='datautils/extract/out', filt=None, alldata=True,
    convert_jump=True, add_ebd=True, min_variants=1)
print(f"functions={len(functions)} ebds={len(ebds)}", flush=True)

# map comp_string -> raw extract pkl path.  filename is
# {lib}-{version}-{compiler}-{compver}-{arch}-{opt}-{soname}_extract.pkl
# and comp_string is {version}-{compiler}-{compver}-{arch}-{opt}
paths = glob.glob('datautils/extract/out/*/*/*_extract.pkl')
by_comp = {}
for p in paths:
    b = os.path.basename(p)[:-len('_extract.pkl')]
    lib, _, rest = b.partition('-')
    comp = rest.rpartition('-')[0]          # drop the trailing soname
    by_comp[comp] = p
print(f"raw extract pkls indexed: {len(by_comp)}", flush=True)

_cache = {}
def raw(comp):
    if comp not in _cache:
        p = by_comp.get(comp)
        _cache[comp] = pickle.load(open(p, 'rb')) if p else {}
    return _cache[comp]

verdict = Counter()
by_opt = defaultdict(Counter)
examples = []

idxs = list(range(len(ebds)))
random.shuffle(idxs)
CHECKED = 0
for i in idxs:
    if CHECKED >= 400:
        break
    e = ebds[i]
    fname = e.get('funcname')
    opt_keys = [k for k in e if k not in ('proj', 'funcname')]
    if len(opt_keys) < 2:
        continue

    # regenerate the true token string for every variant of this function
    truth = {}
    for o in opt_keys:
        d = raw(o)
        entry = d.get(fname)
        if entry is None:
            continue
        f = (entry[0], entry[1], entry[2], entry[3], entry[4])
        try:
            truth[o] = gen_funcstr(f, True)
        except Exception:
            pass
    if len(truth) < 2:
        continue
    CHECKED += 1

    for o in opt_keys:
        if o not in truth:
            continue
        optlvl = o.rsplit('-', 1)[-1]
        got = functions[i][e[o]]
        if got == truth[o]:
            verdict['CORRECT'] += 1
            by_opt[optlvl]['CORRECT'] += 1
        else:
            others = [oo for oo, s in truth.items() if s == got]
            if others:
                verdict['WRONG_VARIANT'] += 1
                by_opt[optlvl]['WRONG_VARIANT'] += 1
                if len(examples) < 6:
                    examples.append((fname, o, others))
            else:
                verdict['NOT_ANY_VARIANT'] += 1
                by_opt[optlvl]['NOT_ANY_VARIANT'] += 1
                if len(examples) < 6:
                    examples.append((fname, o, ['<no variant of this function>']))

print()
print(f"functions checked: {CHECKED}")
print("overall:", dict(verdict))
print()
print(f"{'opt':4} {'CORRECT':>9} {'WRONG_VARIANT':>15} {'NOT_ANY':>9}")
for optlvl in ['O0', 'O1', 'O2', 'O3', 'Os']:
    c = by_opt.get(optlvl)
    if not c:
        continue
    tot = sum(c.values())
    print(f"{optlvl:4} {c['CORRECT']:9} {c['WRONG_VARIANT']:15} {c['NOT_ANY_VARIANT']:9}"
          f"   ({100*c['CORRECT']/tot:.1f}% correct of {tot})")
if examples:
    print("\nmismatch examples (funcname, requested variant, what it actually matched):")
    for ex in examples:
        print("  ", ex)
