"""Is the IDA-vs-Ghidra rendering divergence O0-SPECIFIC, or uniform across opt levels?

If uniform, it cannot explain a collapse that only affects O0-cross pairs.

REQUIRES A SEPARATE CHECKOUT OF THE ORIGINAL jTrans REPOSITORY.

IDA_DIR below points into Models/jTrans, which is the unmodified upstream
https://github.com/vul337/jTrans -- NOT this Ghidra port. It supplies the IDA-derived
token streams that this script compares against the Ghidra-derived ones, so without that
checkout there is nothing to compare and the script cannot run. Clone it alongside this
repository at Models/jTrans and run its own extraction to populate datautils/extract.
The checkout used when this was written was commit 1d40515.
"""
import pickle, sys, os, difflib
from collections import Counter
sys.path.insert(0, '/home/users/u7003724/AI-For-RE/binsim_analyzer/Models/jTrans-ghidra')
os.chdir('/home/users/u7003724/AI-For-RE/binsim_analyzer/Models/jTrans-ghidra')
from data import gen_funcstr

IDA_DIR = '/home/users/u7003724/AI-For-RE/binsim_analyzer/Models/jTrans/datautils/extract'
GH_DIR  = '/home/users/u7003724/AI-For-RE/binsim_analyzer/Models/jTrans-ghidra/datautils/extract/out/libpng/v1.6.57'

def toks(e):
    return gen_funcstr((e[0], e[1], e[2], e[3], e[4]), True)

print(f"{'opt':4} {'shared':>7} {'identical':>10} {'different':>10} {'median sim':>11}  top token substitutions (IDA -> Ghidra)")
for opt in ['O0','O1','O3']:
    BIN = f'libpng-v1.6.57-gcc-12-x86-{opt}-libpng16.so'
    ida = pickle.load(open(f'{IDA_DIR}/{BIN}_extract.pkl','rb'))
    gh  = pickle.load(open(f'{GH_DIR}/{BIN}_extract.pkl','rb'))
    shared = sorted(set(ida) & set(gh))
    ident=0; diff=0; sims=[]; subs=Counter(); tokdiff=0; toktot=0
    for n in shared:
        try: a,b = toks(ida[n]), toks(gh[n])
        except Exception: continue
        ta,tb = a.split(' '), b.split(' ')
        toktot += max(len(ta),len(tb))
        if a==b: ident+=1; continue
        diff+=1
        sm=difflib.SequenceMatcher(None,ta,tb,autojunk=False)
        sims.append(sm.ratio())
        for tag,i1,i2,j1,j2 in sm.get_opcodes():
            if tag=='replace':
                tokdiff += max(i2-i1, j2-j1)
                for x,y in zip(ta[i1:i2], tb[j1:j2]):
                    subs[f'{x} -> {y}'] += 1
            elif tag in ('delete','insert'):
                tokdiff += (i2-i1)+(j2-j1)
    sims.sort()
    med = sims[len(sims)//2] if sims else 1.0
    top = '; '.join(f'{k} ({v})' for k,v in subs.most_common(3))
    print(f"{opt:4} {len(shared):7} {ident:10} {diff:10} {med:11.3f}  {top}")
    print(f"{'':4} {'':7} token-level divergence: {100*tokdiff/max(toktot,1):.2f}% of tokens")
