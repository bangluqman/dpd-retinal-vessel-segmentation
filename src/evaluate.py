#!/usr/bin/env python3
"""Full-resolution DRIVE evaluation; no tiling, TTA, or postprocessing."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import sys
import cv2, numpy as np, torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.drive_data import load_split, normalize
from src.models import build_model
from src.structural_metrics import CONFIGS, evaluate_structural

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data_root",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--model",required=True); p.add_argument("--output_dir",type=Path,required=True); p.add_argument("--observer",type=int,choices=[1,2],default=1)
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False); model=build_model(a.model); model.load_state_dict(ck["model_state_dict"],strict=True); model.to(device).eval()
    stats=ck["normalization"]; rows=[]
    for case in load_split(a.data_root,"test",a.observer):
        x=torch.from_numpy(np.ascontiguousarray(normalize(case.processed,stats)[None,None])).float().to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda",enabled=device.type=="cuda"): prob=torch.sigmoid(model(x))[0,0].float().cpu().numpy()
        pred=prob >= .5; valid=case.fov>0; truth=case.target>0
        tp=int((pred&truth&valid).sum()); tn=int((~pred&~truth&valid).sum()); fp=int((pred&~truth&valid).sum()); fn=int((~pred&truth&valid).sum())
        row={"image_id":case.key,"AUC":roc_auc_score(truth[valid],prob[valid]),"accuracy":(tp+tn)/(tp+tn+fp+fn),
             "sensitivity":tp/(tp+fn),"specificity":tn/(tn+fp),"precision":tp/(tp+fp),"F1":2*tp/(2*tp+fp+fn),"IoU":tp/(tp+fp+fn)}
        row.update({k:v for k,v in evaluate_structural(truth,pred,valid,CONFIGS[0]).items() if not k.startswith(("evaluated_","covered_"))})
        rows.append(row)
        cv2.imwrite(str(a.output_dir/f"{case.key}_probability_u16.png"),np.floor(np.clip(prob,0,1)*65535).astype(np.uint16))
        cv2.imwrite(str(a.output_dir/f"{case.key}_prediction_threshold_0p50.png"),pred.astype(np.uint8)*255)
    with (a.output_dir/"per_image_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    (a.output_dir/"evaluation.json").write_text(json.dumps({"threshold":.5,"observer":a.observer,"images":len(rows),"D0":CONFIGS[0].__dict__},indent=2)+"\n")

if __name__=="__main__": main()

