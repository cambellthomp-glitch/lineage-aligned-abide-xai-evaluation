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
df=pd.read_csv(ROOT/"aggregate_source_data"/"figure_s1_source_data.csv")
flow=df[df.record_type=="flow"].copy(); site=df[df.record_type=="site_dx"].copy().sort_values("site")
fig=plt.figure(figsize=(183/25.4,118/25.4), constrained_layout=True)
gs=fig.add_gridspec(1,2,width_ratios=[0.85,1.6])
ax=fig.add_subplot(gs[0,0]); ax.axis("off")
steps=[("Phenotype snapshot",1112),("Named records",1035),("Local CC200 matches",880),("Canonical cohort",879)]
ys=np.linspace(.88,.22,len(steps))
for i,((lab,n),y) in enumerate(zip(steps,ys)):
    ax.add_patch(mpl.patches.FancyBboxPatch((.08,y-.075),.84,.15,boxstyle="round,pad=0.015",facecolor=LIGHT,edgecolor=BLUE,lw=1))
    ax.text(.5,y+.018,lab,ha="center",va="center",weight="bold",fontsize=8)
    ax.text(.5,y-.03,f"n = {n:,}",ha="center",va="center",fontsize=9,color=BLUE,weight="bold")
    if i<len(steps)-1: ax.annotate("",(.5,ys[i+1]+.085),(.5,y-.085),arrowprops=dict(arrowstyle="-|>",color=GREY,lw=.9))
ax.text(.02,.96,"a",weight="bold",fontsize=9,transform=ax.transAxes)
ax.text(.5,.005,"77 no_filename; 155 named records lacked local derivatives;\n1 structurally unreadable file. These are not imaging-QC failures.",ha="center",va="bottom",fontsize=8,color=BLACK)
ax2=fig.add_subplot(gs[0,1]); y=np.arange(len(site));
ax2.barh(y,site.Control,color=BLUE,label="Control",edgecolor="white",lw=.4)
ax2.barh(y,site.ASD,left=site.Control,color=ORANGE,label="ASD",edgecolor="white",lw=.4)
ax2.set_yticks(y,site.site); ax2.invert_yaxis(); ax2.set_xlabel("Participants"); ax2.set_ylabel("Acquisition site")
ax2.grid(axis="x",color="#DDDDDD",lw=.5,zorder=0); ax2.set_axisbelow(True); ax2.legend(loc="lower right")
ax2.text(-.11,1.02,"b",weight="bold",fontsize=9,transform=ax2.transAxes)
ax2.set_title("Verified site × diagnosis composition",loc="left",weight="bold")
save_all(fig,ROOT/"generated_figures"/"figure_s1"/"figure_s1_participant_flow_site_composition")
