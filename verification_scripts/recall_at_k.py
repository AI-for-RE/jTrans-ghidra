"""Is jTrans's O0 failure pure chance, or weak-but-real signal?

Chance means the true match's rank is UNIFORM over the candidate pool: recall@k = k/N
for every k, and median rank = N/2. Weak signal means the rank distribution is shifted
toward the front even when it rarely reaches rank 1.

Uses the full ranking, not just k<=10, so the question is answered rather than inferred.
"""
import pickle, sys, os
import numpy as np, torch
sys.path.insert(0,'/home/users/u7003724/AI-For-RE/binsim_analyzer/Models/jTrans-ghidra')
os.chdir('/home/users/u7003724/AI-For-RE/binsim_analyzer/Models/jTrans-ghidra')
from data import gen_funcstr
from transformers import BertTokenizer, BertModel

class BinBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config); self.config=config
        self.embeddings.position_embeddings=self.embeddings.word_embeddings

GH='/home/users/u7003724/AI-For-RE/binsim_analyzer/Models/jTrans-ghidra/datautils/extract/out/libpng/v1.6.57'
def load(opt):
    return pickle.load(open(f'{GH}/libpng-v1.6.57-gcc-12-x86-{opt}-libpng16.so_extract.pkl','rb'))

dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=BinBertModel.from_pretrained('models/jTrans-finetune').eval().to(dev)
tok=BertTokenizer.from_pretrained('./jtrans_tokenizer/')
def embed(ss):
    out=[]
    with torch.no_grad():
        for i in range(0,len(ss),64):
            r=tok(ss[i:i+64],add_special_tokens=True,max_length=512,padding='max_length',
                  truncation=True,return_tensors='pt')
            out.append(model(input_ids=r['input_ids'].to(dev),
                             attention_mask=r['attention_mask'].to(dev)).pooler_output.cpu().numpy())
    return np.vstack(out)

d={o:load(o) for o in ['O0','O1','O3']}
names=sorted(set(d['O0'])&set(d['O1'])&set(d['O3']),key=str)
E={}
for o in ['O0','O1','O3']:
    E[o]=embed([gen_funcstr((d[o][n][0],d[o][n][1],d[o][n][2],d[o][n][3],d[o][n][4]),True) for n in names])
    E[o]/=np.linalg.norm(E[o],axis=1,keepdims=True)
N=len(names)
print(f"functions = {N}   (each query ranks the true match against {N-1} distractors)\n")

KS=[1,5,10,25,50,100,200]
print(f"{'pair':10} " + " ".join(f"r@{k:<5}" for k in KS) + "  median rank   mean pct-rank")
print(f"{'chance':10} " + " ".join(f"{k/N:<7.3f}" for k in KS) + f"  {N/2:>10.0f}   {0.5:>12.3f}")
print("-"*94)
res={}
for a,b in [('O0','O3'),('O0','O1'),('O1','O3')]:
    S=E[a]@E[b].T
    # rank of the true match (1 = best) for each query
    order=np.argsort(-S,axis=1)
    ranks=np.array([np.where(order[i]==i)[0][0]+1 for i in range(N)])
    res[f'{a}|{b}']=ranks
    row=" ".join(f"{(ranks<=k).mean():<7.3f}" for k in KS)
    print(f"{a}|{b:8} {row}  {np.median(ranks):>10.0f}   {(ranks/N).mean():>12.3f}")

print()
print("Kolmogorov-Smirnov test of each rank distribution against UNIFORM (= pure chance):")
from scipy import stats
for k,r in res.items():
    ks=stats.kstest((r-0.5)/N,'uniform')
    verdict = "indistinguishable from chance" if ks.pvalue>0.01 else "significantly better than chance"
    print(f"  {k:9}  D={ks.statistic:.4f}  p={ks.pvalue:.2e}   -> {verdict}")
