"""
H521b — Eps sweep: where does the accuracy-robustness tradeoff kick in?
Standalone version of h521 Experiment 3.
4-class dataset, corners of square, gap=1.5, std=0.45
"""
import os, sys, numpy as np, torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE = "/run/media/mohit/NewVolume1/adversarial-robustness-toolbox/dissertation_extension"
OUT_PNG = os.path.join(BASE, "results/fashion_mnist/h521b_eps_sweep.png")
OUT_TXT = os.path.join(BASE, "results/fashion_mnist/h521b_eps_sweep_output.txt")
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

class Tee:
    def __init__(self, path):
        self._f = open(path, "w", buffering=1); self._s = sys.stdout
    def write(self, m): self._s.write(m); self._f.write(m); self._f.flush()
    def flush(self): self._s.flush(); self._f.flush()
    def __getattr__(self, n): return getattr(self._s, n)
sys.stdout = Tee(OUT_TXT)

torch.manual_seed(42)
criterion = nn.CrossEntropyLoss()

N_CLASSES = 4
CLASS_CENTERS_4 = [(-0.75,-0.75),(+0.75,-0.75),(-0.75,+0.75),(+0.75,+0.75)]
STD_4 = 0.45

def make_dataset_4class(n_per_class, seed=42):
    rng = np.random.default_rng(seed)
    Xs, Ys = [], []
    for ci,(cx,cy) in enumerate(CLASS_CENTERS_4):
        pts = rng.normal([cx,cy], STD_4, (n_per_class,2)).astype(np.float32)
        Xs.append(pts); Ys.extend([ci]*n_per_class)
    X = np.vstack(Xs); Y = np.array(Ys, dtype=np.int64)
    idx = rng.permutation(len(X)); return X[idx], Y[idx]

class MLP4(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,N_CLASSES))
    def forward(self,x): return self.net(x)

def train_std(Xtr,Ytr,epochs=200):
    m=MLP4(); opt=optim.Adam(m.parameters(),lr=1e-3)
    Xt,Yt=torch.from_numpy(Xtr),torch.from_numpy(Ytr)
    for _ in range(epochs):
        m.train(); opt.zero_grad(); criterion(m(Xt),Yt).backward(); opt.step()
    m.eval(); return m

def train_at(Xtr,Ytr,eps,epochs=200,steps=7):
    alpha=eps/4; m=MLP4(); opt=optim.Adam(m.parameters(),lr=1e-3)
    Xt,Yt=torch.from_numpy(Xtr),torch.from_numpy(Ytr)
    for _ in range(epochs):
        m.train()
        Xadv=Xt.clone()+torch.zeros_like(Xt).uniform_(-eps,eps)
        for _ in range(steps):
            Xadv.requires_grad_(True); criterion(m(Xadv),Yt).backward()
            with torch.no_grad():
                Xadv=Xadv.detach()+alpha*Xadv.grad.sign()
                Xadv=torch.max(torch.min(Xadv,Xt+eps),Xt-eps)
        opt.zero_grad(); criterion(m(Xadv.detach()),Yt).backward(); opt.step()
    m.eval(); return m

def fgsm_asr(m,Xt,Yt,eps):
    with torch.no_grad(): correct=(m(Xt).argmax(1)==Yt)
    if correct.sum()==0: return float("nan")
    Xc=Xt[correct].clone().requires_grad_(True); Yc=Yt[correct]
    criterion(m(Xc),Yc).backward()
    with torch.no_grad():
        Xadv=Xc.detach()+eps*Xc.grad.sign()
        return (m(Xadv).argmax(1)!=Yc).float().mean().item()

def eval_model(m,Xt,Yt,eps):
    with torch.no_grad(): acc=(m(Xt).argmax(1)==Yt).float().mean().item()
    return acc, fgsm_asr(m,Xt,Yt,eps)

EPS_SWEEP=[0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.75,1.00,1.25,1.50]
N_PC=300
Xtr4,Ytr4=make_dataset_4class(N_PC,seed=42)
Xte4,Yte4=make_dataset_4class(500,seed=999)
Xte4_t,Yte4_t=torch.from_numpy(Xte4),torch.from_numpy(Yte4)

print(f"4-class dataset: corners ±0.75, std={STD_4}, N={N_PC}/class")
print(f"Nearest-neighbour gap ≈ {1.5 - 2*STD_4:.2f}")
print()

std4=train_std(Xtr4,Ytr4)

print(f"{'eps':>6}  {'STD_acc':>8}  {'STD_ASR':>8}  {'AT_acc':>8}  {'AT_ASR':>8}  {'acc_drop':>9}")
print("-"*60)

eps_results=[]
for eps in EPS_SWEEP:
    at4=train_at(Xtr4,Ytr4,eps)
    std_acc,std_asr=eval_model(std4,Xte4_t,Yte4_t,eps)
    at_acc, at_asr =eval_model(at4, Xte4_t,Yte4_t,eps)
    drop=std_acc-at_acc
    eps_results.append(dict(eps=eps,std_acc=std_acc,std_asr=std_asr,at_acc=at_acc,at_asr=at_asr,drop=drop))
    print(f"{eps:>6.2f}  {std_acc:>8.3f}  {std_asr:>8.3f}  {at_acc:>8.3f}  {at_asr:>8.3f}  {drop:>+9.3f}")

# Figure
CMAP4=["royalblue","tomato","forestgreen","darkorange"]
eps_arr=np.array([r["eps"] for r in eps_results])
std_acc_arr=np.array([r["std_acc"] for r in eps_results])
at_acc_arr=np.array([r["at_acc"] for r in eps_results])
std_asr_arr=np.array([r["std_asr"] for r in eps_results])
at_asr_arr=np.array([r["at_asr"] for r in eps_results])
drop_arr=np.array([r["drop"] for r in eps_results])

EPS_BOUNDARY=[0.10,0.40,0.75,1.25]
fig,axes=plt.subplots(2,4,figsize=(22,10))
fig.suptitle(f"H521b: Eps Sweep — Accuracy-Robustness Tradeoff\n4-class corners ±0.75, std={STD_4}, gap≈{1.5-2*STD_4:.2f}",fontsize=13,fontweight="bold")

ax=axes[0,0]
ax.plot(eps_arr,std_acc_arr,"o-",color="tomato",lw=2,ms=7,label="Standard")
ax.plot(eps_arr,at_acc_arr,"s--",color="royalblue",lw=2,ms=7,label="AT (trained at ε)")
ax.axvline(STD_4,color="gray",ls=":",lw=1,label=f"std={STD_4}")
ax.set_xlabel("ε",fontsize=10); ax.set_ylabel("Clean accuracy",fontsize=10)
ax.set_title("Clean accuracy vs ε",fontsize=9); ax.legend(fontsize=8); ax.grid(True,alpha=0.3); ax.set_ylim(0,1.05)

ax=axes[0,1]
ax.plot(eps_arr,std_asr_arr,"o-",color="tomato",lw=2,ms=7,label="STD FGSM ASR")
ax.plot(eps_arr,at_asr_arr,"s--",color="royalblue",lw=2,ms=7,label="AT FGSM ASR")
ax.set_xlabel("ε",fontsize=10); ax.set_ylabel("FGSM ASR",fontsize=10)
ax.set_title("Attack success rate vs ε",fontsize=9); ax.legend(fontsize=8); ax.grid(True,alpha=0.3); ax.set_ylim(0,1.05)

ax=axes[0,2]
colors=["#d62728" if d>0.02 else "#2ca02c" for d in drop_arr]
bars=ax.bar(eps_arr,drop_arr,color=colors,edgecolor="k",lw=0.8,alpha=0.85,width=0.07)
for e,d in zip(eps_arr,drop_arr):
    ax.text(e,d+0.005 if d>=0 else d-0.025,f"{d:+.2f}",ha="center",fontsize=7)
ax.axhline(0,color="k",lw=1); ax.axhline(0.02,color="gray",ls="--",lw=1,label="2% threshold")
ax.set_xlabel("ε",fontsize=10); ax.set_ylabel("Accuracy drop",fontsize=10)
ax.set_title("Cost of AT (STD acc − AT acc)\nred=tradeoff, green=free",fontsize=9)
ax.legend(fontsize=8); ax.grid(True,axis="y",alpha=0.3)

ax=axes[0,3]
sc=ax.scatter(at_asr_arr,at_acc_arr,c=eps_arr,cmap="plasma",s=80,zorder=5,edgecolors="k",lw=0.5)
ax.plot(at_asr_arr,at_acc_arr,"--",color="royalblue",lw=1,alpha=0.5)
plt.colorbar(sc,ax=ax,label="ε")
for e,fa,ac in zip(eps_arr,at_asr_arr,at_acc_arr):
    ax.annotate(f"ε={e:.2f}",(fa,ac),fontsize=6,xytext=(3,3),textcoords="offset points")
ax.set_xlabel("FGSM ASR",fontsize=10); ax.set_ylabel("Clean accuracy",fontsize=10)
ax.set_title("Accuracy-Robustness Pareto frontier",fontsize=9); ax.grid(True,alpha=0.3)

xx4,yy4=np.meshgrid(np.linspace(-3,3,200),np.linspace(-3,3,200))
grid4=torch.tensor(np.c_[xx4.ravel(),yy4.ravel()].astype(np.float32))

for col,eps_b in enumerate(EPS_BOUNDARY):
    at_b=train_at(Xtr4,Ytr4,eps=eps_b)
    with torch.no_grad(): preds_g=at_b(grid4).argmax(1).numpy().reshape(xx4.shape)
    at_acc_b,at_asr_b=eval_model(at_b,Xte4_t,Yte4_t,eps_b)
    ax=axes[1,col]
    ax.contourf(xx4,yy4,preds_g,levels=[-0.5,0.5,1.5,2.5,3.5],colors=CMAP4,alpha=0.30)
    ax.contour(xx4,yy4,preds_g.astype(float),levels=[0.5,1.5,2.5],colors="k",linewidths=1.2)
    for ci,(cx,cy) in enumerate(CLASS_CENTERS_4):
        mask=Ytr4==ci
        ax.scatter(Xtr4[mask,0],Xtr4[mask,1],c=CMAP4[ci],s=12,alpha=0.5,linewidths=0)
    ax.set_xlim(-3,3); ax.set_ylim(-3,3)
    ax.set_title(f"AT boundary  ε={eps_b:.2f}\nacc={at_acc_b:.3f}  FGSM_ASR={at_asr_b:.3f}",fontsize=9)
    ax.set_xlabel("x₁",fontsize=9); ax.set_ylabel("x₂",fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PNG,dpi=150,bbox_inches="tight")
print(f"\nFigure → {OUT_PNG}")
sys.stdout.flush()
