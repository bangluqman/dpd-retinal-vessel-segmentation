#!/usr/bin/env python3
"""Reproduce seed-wise, case-only, and two-way uncertainty analyses."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np

SEEDS=(41,42,43); MODELS=("unet_standard","unet_dpd_v3")
METRICS=(("auc","higher"),("sensitivity","higher"),("f1","higher"),("iou","higher"),("thin_fn_pixels","lower"),("far_fp_pixels","lower"),("missed_segments","lower"),("length_weighted_coverage","higher"),("long_branch_coverage","higher"))
COUNTS={"thin_fn_pixels","far_fp_pixels","missed_segments"}; STARE_UNDEFINED={"im0005","im0240","im0324"}

def read(path): return list(csv.DictReader(path.open(encoding="utf-8")))
def write(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def matrix(dataset,metric,direction,rows):
    if dataset=="CHASE_DB1": units=sorted({r["subject"] for r in rows})
    else: units=sorted({r["case"] for r in rows})
    if dataset=="STARE" and metric=="long_branch_coverage": units=[u for u in units if u not in STARE_UNDEFINED]
    out=np.empty((3,len(units)))
    for si,seed in enumerate(SEEDS):
        for ui,unit in enumerate(units):
            vals={}
            for model in MODELS:
                selected=[r for r in rows if r["model"]==model and int(r["seed"])==seed and r["subject" if dataset=="CHASE_DB1" else "case"]==unit]
                numbers=[float(r[metric]) for r in selected]
                if not numbers: raise RuntimeError(f"Missing {dataset}/{metric}/{seed}/{unit}/{model}")
                vals[model]=float(np.sum(numbers) if dataset=="CHASE_DB1" and metric in COUNTS else np.mean(numbers))
            out[si,ui]=vals["unet_dpd_v3"]-vals["unet_standard"] if direction=="higher" else vals["unet_standard"]-vals["unet_dpd_v3"]
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--drive",type=Path,required=True);p.add_argument("--stare",type=Path,required=True);p.add_argument("--chase",type=Path,required=True);p.add_argument("--output_dir",type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    datasets={"DRIVE":read(a.drive),"STARE":read(a.stare),"CHASE_DB1":read(a.chase)}; matrices={}
    for d,rows in datasets.items():
        expected=168 if d=="CHASE_DB1" else 120
        if len(rows)!=expected or {int(r["seed"]) for r in rows}!=set(SEEDS) or {r["model"] for r in rows}!=set(MODELS): raise RuntimeError(f"Frozen input audit failed for {d}")
        for m,direction in METRICS: matrices[d,m]=matrix(d,m,direction,rows)
    seed_rows=[];summary=[];case_rows=[];two_rows=[]
    for d in datasets:
        case_rng=np.random.default_rng(20260818)
        for m,_ in METRICS:
            mat=matrices[d,m]; effects=mat.mean(axis=0); se=mat.mean(axis=1)
            for seed,value in zip(SEEDS,se): seed_rows.append({"dataset":d,"metric":m,"seed":seed,"oriented_effect":value,"n_independent_units":mat.shape[1]})
            summary.append({"dataset":d,"metric":m,"seed41_effect":se[0],"seed42_effect":se[1],"seed43_effect":se[2],"mean_effect":se.mean(),"sample_sd":se.std(ddof=1),"positive_seed_count":int((se>0).sum()),"n_independent_units":mat.shape[1]})
            idx=case_rng.integers(0,mat.shape[1],size=(10000,mat.shape[1])); boot=effects[idx].mean(axis=1)
            case_rows.append({"dataset":d,"metric":m,"mean_oriented_effect":effects.mean(),"ci_2_5":np.quantile(boot,.025),"ci_97_5":np.quantile(boot,.975),"n_independent_units":mat.shape[1],"n_bootstrap":10000,"bootstrap_seed":20260818})
    two_rng=np.random.default_rng(20265840)
    for d in datasets:
        for m,_ in METRICS:
            mat=matrices[d,m]; si=two_rng.integers(0,3,size=(10000,3));ui=two_rng.integers(0,mat.shape[1],size=(10000,mat.shape[1]));boot=mat[si[:,:,None],ui[:,None,:]].mean((1,2))
            two_rows.append({"dataset":d,"metric":m,"mean_oriented_effect":mat.mean(),"ci_2_5":np.quantile(boot,.025),"ci_97_5":np.quantile(boot,.975),"n_seeds":3,"n_units":mat.shape[1],"n_bootstrap":10000,"bootstrap_seed":20265840})
    write(a.output_dir/"seedwise_effects.csv",seed_rows);write(a.output_dir/"seedwise_effect_summary.csv",summary);write(a.output_dir/"bootstrap_case.csv",case_rows);write(a.output_dir/"bootstrap_seed_case.csv",two_rows)

if __name__=="__main__":main()

