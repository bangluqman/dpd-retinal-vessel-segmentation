"""DRIVE loading, study preprocessing, patches, sampling, and augmentation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


@dataclass
class DriveCase:
    key: str
    processed: np.ndarray
    target: np.ndarray
    fov: np.ndarray


def _key(path: Path) -> str:
    match = re.match(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"No DRIVE identifier in {path.name}")
    return match.group(1)


def _files(directory: Path) -> dict[str, Path]:
    return {_key(p): p for p in sorted(directory.glob("*")) if p.is_file()}


def _binary(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127).astype(np.uint8)


def preprocess(rgb: np.ndarray) -> np.ndarray:
    green = rgb[:, :, 1]
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(green).astype(np.float32) / 255.0


def load_split(data_root: Path, split: str, observer: int = 1) -> list[DriveCase]:
    root = data_root / split
    images = _files(root / "images")
    manuals = _files(root / ("1st_manual" if observer == 1 else "2nd_manual"))
    masks = _files(root / "mask")
    if set(images) != set(manuals) or not set(images).issubset(masks):
        raise RuntimeError(f"Incomplete/unmatched {split} image, manual, or FOV files")
    cases = []
    for key in sorted(images, key=int):
        rgb = np.asarray(Image.open(images[key]).convert("RGB"), dtype=np.uint8)
        target, fov = _binary(manuals[key]), _binary(masks[key])
        if rgb.shape[:2] != target.shape or target.shape != fov.shape:
            raise RuntimeError(f"Shape mismatch for case {key}")
        cases.append(DriveCase(key, preprocess(rgb), target, fov))
    return cases


def training_statistics(cases: list[DriveCase]) -> dict[str, float]:
    values = np.concatenate([c.processed[c.fov > 0] for c in cases])
    mean, std = float(values.mean(dtype=np.float64)), float(values.std(dtype=np.float64))
    z = (values - mean) / max(std, 1e-8)
    return {"mean": mean, "std": std, "z_min": float(z.min()), "z_max": float(z.max())}


def normalize(image: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    z = (image - stats["mean"]) / stats["std"]
    return np.clip((z - stats["z_min"]) / (stats["z_max"] - stats["z_min"] + 1e-8), 0, 1).astype(np.float32)


def _covered(length: int, patch: int, stride: int) -> int:
    return patch if length <= patch else patch + math.ceil((length - patch) / stride) * stride


def _pad(a: np.ndarray, patch: int, stride: int, reflect: bool):
    ph, pw = _covered(a.shape[0], patch, stride) - a.shape[0], _covered(a.shape[1], patch, stride) - a.shape[1]
    return np.pad(a, ((0, ph), (0, pw)), mode="reflect" if reflect else "constant")


@dataclass
class Patch:
    image: np.ndarray
    target: np.ndarray
    fov: np.ndarray


def extract_patches(cases, stats, patch_size=224, stride=112, min_fov=0.05):
    output = []
    for case in cases:
        image = _pad(normalize(case.processed, stats), patch_size, stride, True)
        target = _pad(case.target, patch_size, stride, False)
        fov = _pad(case.fov, patch_size, stride, False)
        for top in range(0, image.shape[0] - patch_size + 1, stride):
            for left in range(0, image.shape[1] - patch_size + 1, stride):
                region = np.s_[top:top + patch_size, left:left + patch_size]
                if fov[region].mean() >= min_fov:
                    output.append(Patch(image[region], target[region], fov[region]))
    if not output:
        raise RuntimeError("No patches extracted")
    return output


def sampling_weights(records):
    values = []
    for r in records:
        valid = r.fov > 0
        fraction = float(r.target[valid].mean()) if valid.any() else 0.0
        values.append(1.0 + min(fraction / 0.08, 0.75))
    return torch.DoubleTensor(values)


def augment(image, target, fov):
    if random.random() < .5:
        image, target, fov = [torch.flip(x, (-1,)) for x in (image, target, fov)]
    if random.random() < .5:
        image, target, fov = [torch.flip(x, (-2,)) for x in (image, target, fov)]
    if random.random() < .75:
        k = random.randint(0, 3)
        image, target, fov = [torch.rot90(x, k, (-2, -1)) for x in (image, target, fov)]
    image = image * random.uniform(.90, 1.10) + random.uniform(-.04, .04)
    if random.random() < .30:
        image = image.clamp(0, 1).pow(random.uniform(.85, 1.20))
    if random.random() < .20:
        image = image + torch.randn_like(image) * random.uniform(.005, .015)
    return image.clamp(0, 1), target, fov


class PatchDataset(Dataset):
    def __init__(self, records, do_augment=True): self.records, self.do_augment = list(records), do_augment
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        r = self.records[index]
        items = [torch.from_numpy(np.ascontiguousarray(a[None])).float() for a in (r.image, r.target, r.fov)]
        return augment(*items) if self.do_augment else tuple(items)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)


def build_loader(records, batch_size, workers, seed):
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(sampling_weights(records), len(records), replacement=True, generator=generator)
    return DataLoader(PatchDataset(records), batch_size=batch_size, sampler=sampler, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      worker_init_fn=seed_worker, drop_last=False)

