"""THE decisive test: does IDA-derived input actually fix jTrans's O0 failure?

Embed the same functions from the same binary twice -- once from IDA-derived token
streams, once from Ghidra-derived -- and measure cross-optimisation retrieval directly.
If IDA recovers O0 and Ghidra does not, the frontend is the cause. If neither does, the
frontend is exonerated and the failure is in the model/checkpoint.

REQUIRES A SEPARATE CHECKOUT OF THE ORIGINAL jTrans REPOSITORY.

IDA_DIR below points into Models/jTrans, which is the unmodified upstream
https://github.com/vul337/jTrans -- NOT this Ghidra port. It supplies the IDA-derived
token streams that this script compares against the Ghidra-derived ones, so without that
checkout there is nothing to compare and the script cannot run. Clone it alongside this
repository at Models/jTrans and run its own extraction to populate datautils/extract.
The checkout used when this was written was commit 1d40515.
"""
import pickle, sys, os
import numpy as np, torch
sys.path.insert(0, '/home/users/u7003724/AI-For-RE/binsim_analyzer/Models/jTrans-ghidra')
os.chdir('/home/users/u7003724/AI-For-RE/binsim_analyzer/Models/jTrans-ghidra')
from data import gen_funcstr
from transformers import BertTokenizer, BertModel

class BinBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings.position_embeddings = self.embeddings.word_embeddings

IDA_DIR='/home/users/u7003724/AI-For-RE/binsim_analyzer/Models/jTrans/datautils/extract'
GH_DIR ='/home/users/u7003724/AI-For-RE/binsim_analyzer/Models/jTrans-ghidra/datautils/extract/out/libpng/v1.6.57'
OPTS=['O0','O1','O3']
def load(src,opt):
    B=f'libpng-v1.6.57-gcc-12-x86-{opt}-libpng16.so'
    p=f'{IDA_DIR}/{B}_extract.pkl' if src=='ida' else f'{GH_DIR}/{B}_extract.pkl'
    return pickle.load(open(p,'rb'))

dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=BinBertModel.from_pretrained('models/jTrans-finetune').eval().to(dev)
tok=BertTokenizer.from_pretrained('./jtrans_tokenizer/')

def embed(strs):
    out=[]
    with torch.no_grad():
        for i in range(0,len(strs),64):
            r=tok(strs[i:i+64],add_special_tokens=True,max_length=512,
                  padding='max_length',truncation=True,return_tensors='pt')
            o=model(input_ids=r['input_ids'].to(dev),attention_mask=r['attention_mask'].to(dev))
            out.append(o.pooler_output.cpu().numpy())
    return np.vstack(out)

def report(src):
    d={o:load(src,o) for o in OPTS}
    names=sorted(set(d['O0'])&set(d['O1'])&set(d['O3']), key=str)
    T={}
    for o in OPTS:
        T[o]=embed([gen_funcstr((d[o][n][0],d[o][n][1],d[o][n][2],d[o][n][3],d[o][n][4]),True) for n in names])
        T[o]/=np.linalg.norm(T[o],axis=1,keepdims=True)
    print(f"\n=== {src.upper()}-derived   ({len(names)} functions, pool = {len(names)-1} distractors) ===")
    for a,b in [('O0','O3'),('O0','O1'),('O1','O3')]:
        S=T[a]@T[b].T
        same=np.diag(S)
        off=S[~np.eye(len(names),dtype=bool)]
        sep=(same.mean()-off.mean())/off.std()
        r1=(S.argmax(axis=1)==np.arange(len(names))).mean()
        print(f"  {a} vs {b}:  recall@1={r1:.3f}   same={same.mean():.4f} diff={off.mean():.4f}  separation={sep:+.2f} sd")

report('ida')
report('ghidra')
