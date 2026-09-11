#!/usr/bin/env python3
"""Observer-2 paired effect bootstraps from packaged per-image numeric metrics."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np
SEEDS=(41,42,43); HIGH={"AUC","sensitivity","F1","IoU","length_weighted_coverage","long_branch_coverage"}; LOW={"thin_FN","far_FP","missed"}

def main():
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    rows=list(csv.DictReader(a.input.open(encoding="utf-8")));rows=[r for r in rows if r["observer"]=="OBSERVER_2"]
    cases=sorted({r["image_id"] for r in rows}); output=[]
    if len(rows)!=120 or len(cases)!=20: raise RuntimeError("Observer-2 input must contain 2 architectures x 3 seeds x 20 images")
    case_rng=np.random.default_rng(20260818); two_rng=np.random.default_rng(20265840)
    for metric in ["AUC","sensitivity","F1","IoU","thin_FN","far_FP","missed","length_weighted_coverage","long_branch_coverage"]:
        if metric not in HIGH|LOW: continue
        mat=np.empty((3,20))
        for si,s in enumerate(SEEDS):
            for ci,c in enumerate(cases):
                v={r["architecture"]:float(r[metric]) for r in rows if int(r["seed"])==s and r["image_id"]==c}
                mat[si,ci]=v["DPD-2"]-v["Standard U-Net"] if metric in HIGH else v["Standard U-Net"]-v["DPD-2"]
        effect=mat.mean(0);idx=case_rng.integers(0,20,size=(10000,20));cb=effect[idx].mean(1)
        si=two_rng.integers(0,3,size=(10000,3));ci=two_rng.integers(0,20,size=(10000,20));tw=mat[si[:,:,None],ci[:,None,:]].mean((1,2))
        status=lambda x:"CI includes zero" if x[0]<=0<=x[1] else "CI excludes zero"
        cci=np.quantile(cb,[.025,.975]);tci=np.quantile(tw,[.025,.975])
        output.append({"observer":"OBSERVER_2","metric":metric,"case_bootstrap_mean":effect.mean(),"case_ci_2_5":cci[0],"case_ci_97_5":cci[1],"case_ci_zero_status":status(cci),"two_way_mean":mat.mean(),"two_way_ci_2_5":tci[0],"two_way_ci_97_5":tci[1],"two_way_ci_zero_status":status(tci),"n_images":20,"n_seeds":3})
    with a.output.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=output[0]);w.writeheader();w.writerows(output)

if __name__=="__main__":main()
