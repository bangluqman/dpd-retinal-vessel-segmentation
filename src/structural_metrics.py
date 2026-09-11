"""Pixel-domain structural diagnostics (primary D0 and prespecified sensitivity settings)."""

from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class StructuralConfig:
    config_id: str; rho: float; tolerance: int; min_branch: int; long_branch: int; far_fp: float


CONFIGS = (
    StructuralConfig("D0_PRIMARY",2,1,5,20,5), StructuralConfig("RHO_1",1,1,5,20,5),
    StructuralConfig("RHO_3",3,1,5,20,5), StructuralConfig("TOL_0",2,0,5,20,5),
    StructuralConfig("TOL_2",2,2,5,20,5), StructuralConfig("MINBR_3",2,1,3,20,5),
    StructuralConfig("MINBR_7",2,1,7,20,5), StructuralConfig("LONGBR_15",2,1,5,15,5),
    StructuralConfig("LONGBR_25",2,1,5,25,5), StructuralConfig("FARFP_3",2,1,5,20,3),
    StructuralConfig("FARFP_7",2,1,5,20,7),
)


def skeletonize(binary):
    work = (binary > 0).astype(np.uint8) * 255; skeleton = np.zeros_like(work)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    for _ in range(max(work.shape)):
        eroded = cv2.erode(work, element); opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened)); work = eroded
        if cv2.countNonZero(work) == 0: break
    return skeleton > 0


def thin_region(truth, rho):
    truth = truth.astype(np.uint8)
    skeleton = skeletonize(truth)
    radius = cv2.distanceTransform(truth, cv2.DIST_L2, 5)
    center = skeleton & (radius <= rho + 1e-6)
    k = 2 * int(np.ceil(rho)) + 1
    expanded = cv2.dilate(center.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))) > 0
    return expanded & (truth > 0), center


def branch_records(thin_centerline, prediction, fov, tolerance, min_length):
    skeleton = (thin_centerline > 0) & (fov > 0)
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, np.ones((3,3),np.uint8), borderType=cv2.BORDER_CONSTANT) - skeleton.astype(np.uint8)
    branch_pixels = skeleton & (neighbors < 3)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(branch_pixels.astype(np.uint8), connectivity=8)
    if tolerance > 0:
        k = 2 * tolerance + 1
        tolerant = cv2.dilate(((prediction > 0) & (fov > 0)).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))) > 0
    else: tolerant = (prediction > 0) & (fov > 0)
    output = []
    for label in range(1,count):
        pixels = labels == label; length = int(stats[label, cv2.CC_STAT_AREA])
        if length >= min_length: output.append((length, float(tolerant[pixels].mean())))
    return output


def evaluate_structural(truth, prediction, fov, config: StructuralConfig):
    truth, prediction, fov = truth > 0, prediction > 0, fov > 0
    thin, thin_centerline = thin_region(truth & fov, config.rho)
    branches = branch_records(thin_centerline, prediction, fov, config.tolerance, config.min_branch)
    recovered = sum(c >= .8 for _,c in branches); partial = sum(.2 <= c < .8 for _,c in branches); missed = sum(c < .2 for _,c in branches)
    total_length = sum(n for n,_ in branches); covered = sum(n*c for n,c in branches)
    long = [(n,c) for n,c in branches if n >= config.long_branch]; long_total = sum(n for n,_ in long)
    distance = cv2.distanceTransform((~truth).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return {
        "thin_FN": int((thin & ~prediction & fov).sum()),
        "far_FP": int((prediction & ~truth & fov & (distance > config.far_fp)).sum()),
        "recovered": recovered, "partial": partial, "missed": missed,
        "length_weighted_coverage": covered / total_length if total_length else 0.0,
        "long_branch_coverage": sum(n*c for n,c in long) / long_total if long_total else None,
        "evaluated_branches": len(branches), "long_branches": len(long),
        "evaluated_branch_length": total_length, "covered_branch_length": covered,
        "evaluated_long_branch_length": long_total, "covered_long_branch_length": sum(n*c for n,c in long),
    }
