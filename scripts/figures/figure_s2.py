from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
mpl.rcParams.update({
    "font.family":"sans-serif", "font.sans-serif":["Arial","Helvetica","DejaVu Sans","sans-serif"],
    "font.size":8, "axes.labelsize":8, "axes.titlesize":9, "xtick.labelsize":8, "ytick.labelsize":8,
    "legend.fontsize":8, "svg.fonttype":"none", "svg.hashsalt":"phase7b1-fixed",
    "pdf.fonttype":42, "axes.spines.right":False, "axes.spines.top":False,
    "axes.linewidth":0.8, "legend.frameon":False, "savefig.facecolor":"white"
})
BLUE="#4477AA"; ORANGE="#EE7733"; CYAN="#66CCEE"; GREY="#777777"; LIGHT="#D9E4EC"; BLACK="#222222"
def sha256(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def save_all(fig, outbase):
    outbase=Path(outbase); outbase.parent.mkdir(parents=True, exist_ok=True)
    meta={"Creator":"Phase 7B1 Python/matplotlib", "CreationDate":None, "ModDate":None}
    fig.savefig(outbase.with_suffix(".svg"), bbox_inches="tight", metadata={"Date":"2026-09-05", "Creator":"Phase 7B1 Python/matplotlib"})
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight", metadata=meta)
    fig.savefig(outbase.with_suffix(".png"), dpi=300, bbox_inches="tight", metadata={"Software":"Phase 7B1 Python/matplotlib"})
    fig.savefig(outbase.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    files={p.suffix.lstrip("."): {"sha256":sha256(p), "bytes":p.stat().st_size} for p in [outbase.with_suffix(x) for x in [".svg",".pdf",".png",".tiff"]]}
    im=Image.open(outbase.with_suffix(".png")); files["png"]["pixels"]=list(im.size)
    (outbase.parent/"output_sha256.json").write_text(json.dumps(files,indent=2,sort_keys=True)+"\n",encoding="utf-8")

ROOT=Path(__file__).resolve().parents[2]
df=pd.read_csv(ROOT/"aggregate_source_data"/"figure_s2_source_data.csv")
seed=df[df.record_level=="fold_seed"].copy(); ens=df[df.record_level=="fold_ensemble"].copy()
metrics=[("auc","AUC"),("accuracy","Accuracy"),("balanced_accuracy","Balanced accuracy"),("brier","Brier score")]
fig,axs=plt.subplots(2,2,figsize=(183/25.4,120/25.4),sharex=True,constrained_layout=True)
for j,(col,label) in enumerate(metrics):
    ax=axs.flat[j]
    for k,s in enumerate([42,43,44]):
        q=seed[seed.seed.astype(str)==str(s)].sort_values("fold"); ax.scatter(q.fold.astype(int)+(k-1)*.12,q[col],s=22,color=[BLUE,CYAN,ORANGE][k],marker=["o","s","^"][k],label=f"Seed {s}",alpha=.9,zorder=3)
    q=ens.sort_values("fold"); ax.plot(q.fold.astype(int),q[col],color=BLACK,lw=1,marker="D",ms=3.5,label="3-seed mean",zorder=4)
    ax.set_ylabel(label); ax.set_xticks(range(10)); ax.grid(axis="y",color="#DDDDDD",lw=.5); ax.set_axisbelow(True)
    ax.text(-.12,1.02,chr(97+j),weight="bold",fontsize=9,transform=ax.transAxes)
    if j>=2: ax.set_xlabel("Outer fold")
axs.flat[0].legend(ncol=2,loc="best")
fig.suptitle("Descriptive fold and seed variability",fontsize=10,weight="bold")
save_all(fig,ROOT/"generated_figures"/"figure_s2"/"figure_s2_fold_seed_predictive_variability")
