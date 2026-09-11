#!/usr/bin/env python3
"""Dataset-free release checks."""
from __future__ import annotations
import csv,json
from pathlib import Path
import sys
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models import build_model,trainable_parameters
from src.structural_metrics import CONFIGS,evaluate_structural

EXPECTED={"unet_standard":17261825,"unet_pmr2":17426433,"unet_dpd_v3":17426433}
for name,count in EXPECTED.items():
    model=build_model(name).eval()
    assert trainable_parameters(model)==count,(name,trainable_parameters(model))
    with torch.no_grad(): out=model(torch.zeros(1,1,65,68))
    assert tuple(out.shape)==(1,1,65,68)

for path in sorted((ROOT/"configs").glob("*.yml")):
    assert yaml.safe_load(path.read_text(encoding="utf-8")) is not None
assert len(CONFIGS)==11 and CONFIGS[0].config_id=="D0_PRIMARY"
split=json.loads((ROOT/"splits/drive.json").read_text());assert len(split["optimization"])==16 and len(split["development"])==4 and len(split["official_test"])==20

truth=np.zeros((32,32),np.uint8);truth[5:28,16]=1;pred=truth.copy();fov=np.ones_like(truth)
toy=evaluate_structural(truth,pred,fov,CONFIGS[0]);assert toy["thin_FN"]==0 and toy["far_FP"]==0

csv_count=0
for path in sorted((ROOT/"results").rglob("*.csv")):
    with path.open(newline="",encoding="utf-8") as f:
        reader=csv.reader(f); header=next(reader); assert header and len(set(header))==len(header),path
        for row in reader:
            assert len(row)==len(header),(path,len(row),len(header));csv_count+=1
print(f"PASS models={len(EXPECTED)} configs={len(list((ROOT/'configs').glob('*.yml')))} csv_data_rows={csv_count} toy_structural=PASS")

