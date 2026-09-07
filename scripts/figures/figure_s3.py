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
df=pd.read_csv(ROOT/"aggregate_source_data"/"figure_s3_source_data.csv")
fig=plt.figure(figsize=(183/25.4,112/25.4),constrained_layout=True); gs=fig.add_gridspec(1,2,width_ratios=[1.1,1])
ax=fig.add_subplot(gs[0,0]); bins=np.linspace(0,1,21)
for dx,color in [("Control",BLUE),("ASD",ORANGE)]:
    q=df[df.diagnosis==dx]; ax.hist(q.raw_probability,bins=bins,histtype="step",lw=1.5,color=color,label=f"Raw: {dx}")
    ax.hist(q.secondary_platt_probability,bins=bins,histtype="stepfilled",alpha=.14,color=color,label=f"Secondary Platt: {dx}")
ax.set_xlabel("Predicted ASD probability"); ax.set_ylabel("Participants per bin"); ax.legend(fontsize=8); ax.text(-.12,1.02,"a",weight="bold",fontsize=9,transform=ax.transAxes); ax.set_title("Probability distributions",loc="left",weight="bold")
ax2=fig.add_subplot(gs[0,1]); edges=np.linspace(0,1,11)
for col,color,mark,lab in [("raw_probability",BLUE,"o","Raw"),("secondary_platt_probability",ORANGE,"s","Secondary Platt")]:
    ids=np.digitize(df[col],edges[1:-1],right=True); xs=[]; ys=[]; ns=[]
    for i in range(10):
        q=df[ids==i]
        if len(q): xs.append(q[col].mean()); ys.append(q.label.mean()); ns.append(len(q))
    ax2.plot(xs,ys,marker=mark,color=color,lw=1.3,ms=4,label=lab)
ax2.plot([0,1],[0,1],ls="--",color=GREY,lw=.8,label="Perfect calibration")
ax2.set_xlim(0,1); ax2.set_ylim(0,1); ax2.set_aspect("equal",adjustable="box"); ax2.set_xlabel("Mean predicted probability"); ax2.set_ylabel("Observed ASD fraction"); ax2.legend(loc="upper left"); ax2.grid(color="#E5E5E5",lw=.5); ax2.text(-.15,1.02,"b",weight="bold",fontsize=9,transform=ax2.transAxes); ax2.set_title("10-bin reliability",loc="left",weight="bold")
fig.suptitle("Raw and secondary Platt-calibrated probabilities",fontsize=10,weight="bold")
save_all(fig,ROOT/"generated_figures"/"figure_s3"/"figure_s3_probability_calibration")
