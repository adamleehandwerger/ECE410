"""
gamma_sweep.py — RBF gamma sweep around 0.25
=============================================
Fixed: [128,64,64] features / [95,95,95,120,95] = 500 SVs
"""
import os, sys, warnings
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

NUM_CLASSES = 5
CLASS_NAMES = ["Normal","PVC","AFib","VT","SVT"]
FRAC_BITS   = 10; SCALE = 1 << FRAC_BITS
NORMAL_RR   = 308
FEAT_SINGLE = 128; HALF_SINGLE = 64
FEAT_10BEAT = 64;  HALF_10BEAT = 32
FEAT_100RR  = 64
SV_ALLOC    = [95, 95, 95, 120, 95]
_BEAT_SYMS  = set("NLReEjJAaSVF/fQ")
PHYSIONET_CACHE = os.path.expanduser("~/.physionet_cache")
BMAP = {"N":0,"L":0,"R":0,"e":0,"j":0,"V":1,"E":1,"F":3,
        "A":4,"a":4,"J":4,"S":4,"/":4,"f":4,"Q":4}
SOURCES = [
    (['100','101','102','103','104','105','106','107','108','109',
      '111','112','113','114','115','116','117','118','119','121',
      '122','123','124','200','201','202','203','205','207','208',
      '209','210','212','213','214','215','217','219','220','221',
      '222','223','228','230','231','232','233','234'], 'mitdb'),
    ([f'e{i:04d}' for i in range(1,79)], 'svdb'),
    ([f'I{i:02d}' for i in range(1,76)], 'incartdb'),
]

def rdrecord_cached(rec, pn_dir):
    import wfdb, wfdb.io.download as wdl
    local = os.path.join(PHYSIONET_CACHE, pn_dir)
    if not os.path.exists(os.path.join(local, rec+".hea")):
        os.makedirs(local, exist_ok=True)
        wdl.dl_files(pn_dir, local, [rec+".hea"], keep_subdirs=False)
        hdr = wfdb.rdheader(os.path.join(local, rec))
        dats = list({s+".dat" for s in (hdr.seg_name if hasattr(hdr,'seg_name') and hdr.seg_name else [rec])})
        wdl.dl_files(pn_dir, local, dats+[rec+".atr"], keep_subdirs=False)
    return wfdb.rdrecord(os.path.join(local,rec)), wfdb.rdann(os.path.join(local,rec),'atr')

def load_raw(max_per_class=300):
    raw = {i:[] for i in range(NUM_CLASSES)}
    for rec_list, pn_dir in SOURCES:
        if all(len(v)>=max_per_class for v in raw.values()): break
        for rec in rec_list:
            if all(len(v)>=max_per_class for v in raw.values()): break
            try: r,ann = rdrecord_cached(rec, pn_dir)
            except: continue
            sig = r.p_signal[:,0].astype(np.float32)
            bs,bsy=[],[]
            for s,sym in zip(ann.sample,ann.symbol):
                if sym in _BEAT_SYMS: bs.append(s); bsy.append(sym)
            if len(bs)<3: continue
            all_s=np.array(bs,dtype=np.int32)
            afib=[]
            if hasattr(ann,'aux_note'):
                in_a=False; a0=None
                for s,sym,aux in zip(ann.sample,ann.symbol,ann.aux_note):
                    if sym=='+' and aux:
                        if '(AFIB' in aux: in_a=True; a0=s
                        elif in_a and '(' in aux: afib.append((a0,s)); in_a=False
                if in_a and a0: afib.append((a0,len(sig)))
            for bi,(si,sym) in enumerate(zip(all_s.tolist(),bsy)):
                in_a2=any(a<=si<=b for a,b in afib)
                if in_a2 and sym in ('N','L','R','e','j','V','A','a','J','S'): cls=2
                elif sym in BMAP: cls=BMAP[sym]
                else: continue
                if len(raw[cls])>=max_per_class: continue
                raw[cls].append((sig,all_s,bi))
    return raw

def build_X(raw):
    X,y=[],[]
    for cls in range(NUM_CLASSES):
        for (sig,all_s,bi) in raw[cls]:
            n=len(sig); s=int(all_s[bi])
            if s<HALF_SINGLE or s+HALF_SINGLE>n: continue
            sg1=sig[s-HALF_SINGLE:s+HALF_SINGLE].copy()
            pk1=np.max(np.abs(sg1))
            if pk1<1e-6: continue
            sg1=(sg1/pk1).astype(np.float32)
            i0=max(0,bi-5); i1=min(len(all_s),bi+5)
            sg2s=[]
            for bj in range(i0,i1):
                bs2=int(all_s[bj])
                if bs2>=HALF_10BEAT and bs2+HALF_10BEAT<=n:
                    sg=sig[bs2-HALF_10BEAT:bs2+HALF_10BEAT].astype(np.float32)
                    pk2=np.max(np.abs(sg))
                    if pk2>1e-6: sg2s.append(sg/pk2)
            if not sg2s: continue
            sg2=np.mean(sg2s,axis=0).astype(np.float32)
            j0=max(0,bi-100)
            rr=np.diff(all_s[j0:bi+1]).astype(np.float32)
            if len(rr)<2: continue
            sg3=np.interp(np.linspace(0,1,FEAT_100RR),np.linspace(0,1,len(rr)),
                          np.clip(rr/NORMAL_RR,0,2)).astype(np.float32)
            X.append(np.concatenate([sg1,sg2,sg3])); y.append(cls)
    return np.array(X,np.float32), np.array(y,np.int32)

def q16(X): return np.clip(np.round(X*SCALE),-(1<<15),(1<<15)-1).astype(np.int16)
def qs16(v): return max(-32768,min(32767,int(round(v*SCALE))))

def eval_gamma(X_tr, y_tr, X_te, y_te, gamma):
    X_te_qi = q16(X_te).astype(np.int32)
    sv_qi, sv_al_q, sv_bs_q = [], [], []
    for c in range(NUM_CLASSES):
        y_bin = np.where(y_tr==c,1,-1)
        clf   = SVC(kernel="rbf", gamma=gamma, C=1.0, random_state=42)
        clf.fit(X_tr, y_bin)
        als=clf.dual_coef_[0]; svs=clf.support_vectors_
        budget=SV_ALLOC[c]
        idx=(np.argsort(-np.abs(als))[:budget] if len(als)>budget else np.arange(len(als)))
        sv_qi.append(q16(svs[idx]).astype(np.int32))
        sv_al_q.append([qs16(a) for a in als[idx]])
        sv_bs_q.append(qs16(float(clf.intercept_[0])))

    scores=np.zeros((len(X_te_qi),NUM_CLASSES),np.float64)
    for c in range(NUM_CLASSES):
        for sv_row,aq in zip(sv_qi[c],sv_al_q[c]):
            diff=X_te_qi-sv_row
            dist_q=(diff.astype(np.int64)**2>>FRAC_BITS).sum(axis=1)
            k=np.exp(-gamma*dist_q/SCALE)
            scores[:,c]+=(aq/SCALE)*k
        scores[:,c]+=sv_bs_q[c]/SCALE

    preds=np.argmax(scores,axis=1)
    acc=float(np.mean(preds==y_te))
    cm=confusion_matrix(y_te,preds,labels=list(range(NUM_CLASSES)))
    vt_r  = cm[3,3]/cm[3].sum() if cm[3].sum()>0 else 0
    svt_r = cm[4,4]/cm[4].sum() if cm[4].sum()>0 else 0
    vt_err= [(CLASS_NAMES[j],cm[3,j]) for j in range(NUM_CLASSES) if j!=3 and cm[3,j]>0]
    return acc, vt_r, svt_r, cm, vt_err

print("Loading data...")
raw = load_raw(max_per_class=300)
X, y = build_X(raw)
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
print(f"Train {len(X_tr)}  Test {len(X_te)}\n")

GAMMAS = [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.50]

print(f"{'gamma':>7}  {'Q6.10':>7}  {'VT_rec':>8}  {'SVT_rec':>9}  VT errors")
print("-"*60)
best = (0, None, None)
for g in GAMMAS:
    acc, vt_r, svt_r, cm, vt_err = eval_gamma(X_tr,y_tr,X_te,y_te,g)
    err_str = "  ".join(f"→{n}×{v}" for n,v in vt_err) if vt_err else "0 errors ◄"
    flag = " ◄" if (not vt_err) else (" ▲" if vt_r > 0.917 else "")
    print(f"{g:>7.2f}  {acc:>7.4f}  {vt_r:>7.1%}  {svt_r:>8.1%}   {err_str}{flag}")
    if acc > best[0]: best = (acc, g, cm)

print(f"\nBest Q6.10: {best[0]:.4f} at gamma={best[1]}")
if best[2] is not None:
    print(f"\nFull confusion matrix at gamma={best[1]}:")
    hdr = f"  {'':10}"+"".join(f"  {n:>7}" for n in CLASS_NAMES)
    print(hdr); print("  "+"-"*(len(hdr)-2))
    for i,row in enumerate(best[2]):
        cells="".join(f"  {v:6d}{'*' if i!=j and v>0 else ' '}" for j,v in enumerate(row))
        rec=row[i]/row.sum() if row.sum()>0 else 0
        print(f"  {CLASS_NAMES[i]:10}{cells}   recall={rec:.1%}")
