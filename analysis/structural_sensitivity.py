#!/usr/bin/env python3
"""Run the 11 prespecified structural-sensitivity settings from a frozen-mask manifest."""
from __future__ import annotations
import argparse,csv
from collections import defaultdict
from statistics import mean, stdev
from pathlib import Path
import sys
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.structural_metrics import CONFIGS,evaluate_structural

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    inputs=list(csv.DictReader(a.manifest.open(encoding="utf-8"))); rows=[]; internal=[]
    for item in inputs:
        truth=cv2.imread(item["ground_truth_path"],cv2.IMREAD_GRAYSCALE)>127; fov=cv2.imread(item["fov_path"],cv2.IMREAD_GRAYSCALE)>127
        pred=cv2.imread(item["prediction_path"],cv2.IMREAD_GRAYSCALE)>127
        for cfg in CONFIGS:
            values=evaluate_structural(truth,pred,fov,cfg)
            base={"config_id":cfg.config_id,"rho":cfg.rho,"branch_tolerance":cfg.tolerance,"min_branch_length":cfg.min_branch,
                  "long_branch_threshold":cfg.long_branch,"far_fp_distance":cfg.far_fp,"architecture":item["architecture"],
                  "seed":int(item["seed"]),"image_id":item["image_id"]}
            rows.append({**base,**{k:values[k] for k in ("thin_FN","far_FP","recovered","partial","missed","length_weighted_coverage","long_branch_coverage")}})
            internal.append({**base,**values})
    if len(inputs)==120 and len(rows)!=1320: raise RuntimeError("Structural-sensitivity row count must be 1320")
    with a.output.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

    # Dataset/seed aggregation uses pooled covered/total branch length, never a
    # mean of per-image coverage percentages.
    groups=defaultdict(list)
    for row in internal: groups[row["config_id"],row["architecture"],row["seed"]].append(row)
    per_seed=[]
    for (_cfg,_arch,_seed),group in groups.items():
        first=group[0]; total=sum(x["evaluated_branch_length"] for x in group); covered=sum(x["covered_branch_length"] for x in group)
        long_total=sum(x["evaluated_long_branch_length"] for x in group);long_covered=sum(x["covered_long_branch_length"] for x in group)
        per_seed.append({**{k:first[k] for k in ("config_id","rho","branch_tolerance","min_branch_length","long_branch_threshold","far_fp_distance","architecture","seed")},
            "images":len(group),"thin_FN":sum(x["thin_FN"] for x in group),"far_FP":sum(x["far_FP"] for x in group),
            "recovered":sum(x["recovered"] for x in group),"partial":sum(x["partial"] for x in group),"missed":sum(x["missed"] for x in group),
            "evaluated_branches":sum(x["evaluated_branches"] for x in group),"evaluated_centerline_pixels":total,
            "length_weighted_coverage":covered/total if total else 0.0,"long_branches":sum(x["long_branches"] for x in group),
            "long_centerline_pixels":long_total,"long_branch_coverage":long_covered/long_total if long_total else None})
    per_seed.sort(key=lambda x:([c.config_id for c in CONFIGS].index(x["config_id"]),x["architecture"],x["seed"]))
    def write_sibling(suffix,data):
        path=a.output.with_name(a.output.stem+suffix+".csv")
        with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
    write_sibling("_per_seed",per_seed)

    lookup={(x["config_id"],x["architecture"],x["seed"]):x for x in per_seed};effects=[]
    metrics=(("thin_FN","Standard - DPD"),("far_FP","Standard - DPD"),("recovered","DPD - Standard"),("missed","Standard - DPD"),("length_weighted_coverage","DPD - Standard"),("long_branch_coverage","DPD - Standard"))
    for cfg in CONFIGS:
        for metric,orientation in metrics:
            values=[]
            for seed in (41,42,43):
                s=lookup[cfg.config_id,"Standard U-Net",seed][metric];d=lookup[cfg.config_id,"DPD-2",seed][metric]
                values.append((s-d) if orientation=="Standard - DPD" else (d-s))
            effects.append({"config_id":cfg.config_id,"metric":metric,"orientation_positive_favors_DPD":orientation,
                            "seed41_effect":values[0],"seed42_effect":values[1],"seed43_effect":values[2],"mean_effect":mean(values),"sample_sd":stdev(values)})
    write_sibling("_effects",effects)
    effect_lookup={(x["config_id"],x["metric"]):x for x in effects};summaries=[]
    def direction(n): return "positive in all 3 seeds" if n==3 else f"positive in {n}/3 seeds"
    for cfg in CONFIGS:
        out={"config_id":cfg.config_id}
        for metric,label in (("thin_FN","thin_FN"),("missed","missed"),("length_weighted_coverage","LW_coverage"),("long_branch_coverage","long_coverage"),("far_FP","far_FP")):
            e=effect_lookup[cfg.config_id,metric];n=sum(float(e[f"seed{s}_effect"])>0 for s in (41,42,43))
            out[f"{label}_mean_effect"]=e["mean_effect"];out[f"{label}_seed_positive_count"]=n;out[f"{label}_direction"]=direction(n)
        summaries.append(out)
    write_sibling("_summary",summaries)

    raw_lookup={(x["config_id"],x["architecture"],x["seed"],x["image_id"]):x for x in internal};case_rows=[]
    for cfg in CONFIGS:
        for metric,direction_kind in (("thin_FN","lower"),("missed","lower"),("length_weighted_coverage","higher"),("long_branch_coverage","higher")):
            counts={"DPD":0,"Standard":0,"tie":0,"undefined":0}
            for image in sorted({x["image_id"] for x in internal}):
                svals=[raw_lookup[cfg.config_id,"Standard U-Net",seed,image][metric] for seed in (41,42,43)]
                dvals=[raw_lookup[cfg.config_id,"DPD-2",seed,image][metric] for seed in (41,42,43)]
                if any(v is None for v in svals+dvals): counts["undefined"]+=1;continue
                sv,dv=mean(svals),mean(dvals)
                effect=(sv-dv) if direction_kind=="lower" else (dv-sv)
                counts["DPD" if effect>0 else "Standard" if effect<0 else "tie"]+=1
            case_rows.append({"config_id":cfg.config_id,"metric":metric,"DPD_favored_images":counts["DPD"],"Standard_favored_images":counts["Standard"],"tied_images":counts["tie"],"jointly_undefined_images":counts["undefined"]})
    write_sibling("_case_direction",case_rows)

if __name__=="__main__":main()
