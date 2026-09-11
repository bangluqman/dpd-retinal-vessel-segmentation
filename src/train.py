#!/usr/bin/env python3
"""DRIVE training procedure used in the study. Requires a separately obtained DRIVE dataset."""

from __future__ import annotations
import argparse, json, random
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.drive_data import load_split, training_statistics, extract_patches, build_loader, normalize
from src.models import build_model, trainable_parameters

OPT_IDS = {"21","22","23","24","27","28","29","30","31","32","33","34","36","37","38","39"}
DEV_IDS = {"25","26","35","40"}


def loss_fn(logits, target, fov):
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * fov).sum() / fov.sum().clamp_min(1)
    p, t = torch.sigmoid(logits) * fov, target * fov
    inter = (p * t).sum((1,2,3)); den = p.sum((1,2,3)) + t.sum((1,2,3))
    return bce + 1 - ((2 * inter + 1e-6) / (den + 1e-6)).mean()


@torch.no_grad()
def validate(model, cases, stats, device, amp):
    model.eval(); values = []
    for case in cases:
        arrays = (normalize(case.processed, stats), case.target, case.fov)
        x, y, f = [torch.from_numpy(np.ascontiguousarray(a[None,None])).float().to(device) for a in arrays]
        with torch.autocast(device_type="cuda", enabled=amp): values.append(float(loss_fn(model(x), y, f)))
    return float(np.mean(values))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, required=True); p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--model", choices=["unet_standard","unet_s2d_replace_early_ablation","unet_pmr2","unet_dpd_v3","unet_dpd_all4_ablation"], required=True)
    p.add_argument("--seed", type=int, choices=[41,42,43], required=True); p.add_argument("--workers", type=int, default=0)
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    all_training = load_split(args.data_root, "training")
    opt = [c for c in all_training if c.key in OPT_IDS]; dev = [c for c in all_training if c.key in DEV_IDS]
    if {c.key for c in opt} != OPT_IDS or {c.key for c in dev} != DEV_IDS: raise RuntimeError("Prespecified split unavailable")
    stats = training_statistics(opt)
    expected = {"mean":0.450456161371489,"std":0.12463032643796439,"z_min":-3.5828728675842285,"z_max":4.409391403198242}
    if any(abs(stats[k] - v) > 1e-7 for k,v in expected.items()): raise RuntimeError(f"Normalization audit failed: {stats}")
    loader = build_loader(extract_patches(opt, stats), 4, args.workers, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = device.type == "cuda"
    model = build_model(args.model).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=amp); best = float("inf"); history = []
    for epoch in range(1,151):
        model.train(); total = 0.0
        for x,y,f in loader:
            x,y,f = x.to(device),y.to(device),f.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp): loss = loss_fn(model(x),y,f)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer); scaler.update(); total += float(loss)
        scheduler.step(); validation = validate(model,dev,stats,device,amp) if epoch == 1 or epoch % 5 == 0 or epoch == 150 else None
        history.append({"epoch":epoch,"train_loss":total/len(loader),"validation_loss":validation,"lr":scheduler.get_last_lr()[0]})
        if validation is not None and validation < best:
            best = validation
            torch.save({"model_state_dict":model.state_dict(),"model":args.model,"seed":args.seed,"split_seed":42,
                        "base_channels":64,"threshold":.5,"parameters":trainable_parameters(model),"epoch":epoch,
                        "validation_loss":validation,"normalization":stats}, args.output_dir / "checkpoint_best.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history,indent=2)+"\n")


if __name__ == "__main__": main()

