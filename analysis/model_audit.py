#!/usr/bin/env python3
"""Parameter and convolutional-MAC audit using the study counting convention."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import MODEL_REGISTRY, build_model, trainable_parameters  # noqa: E402

INPUT_SHAPE = (1, 1, 565, 584)


def conv_macs(model: nn.Module, shape=INPUT_SHAPE):
    details = []
    hooks = []
    names = {module: name for name, module in model.named_modules()}

    def hook(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            elements = output.numel()
            per_element = (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
            operation = "Conv2d"
        elif isinstance(module, nn.ConvTranspose2d):
            elements = inputs[0].numel()
            per_element = (module.out_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
            operation = "ConvTranspose2d"
        else:
            return
        details.append((names[module], operation, tuple(inputs[0].shape), tuple(output.shape), elements * per_element))

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(module.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(shape))
    for handle in hooks:
        handle.remove()
    if tuple(output.shape) != shape:
        raise RuntimeError(f"Output shape mismatch: {tuple(output.shape)} != {shape}")
    return sum(row[-1] for row in details), details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--layers", type=Path)
    args = parser.parse_args()
    summary = []
    layer_rows = []
    for name in MODEL_REGISTRY:
        model = build_model(name)
        macs, details = conv_macs(model)
        summary.append({"architecture": name, "trainable_parameters": trainable_parameters(model), "GMAC": macs / 1e9})
        layer_rows.extend({"architecture": name, "layer": d[0], "operation": d[1],
                           "input_shape": "x".join(map(str, d[2])), "output_shape": "x".join(map(str, d[3])),
                           "MAC": d[4]} for d in details)
    if args.output:
        with args.output.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=summary[0]); w.writeheader(); w.writerows(summary)
    if args.layers:
        with args.layers.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=layer_rows[0]); w.writeheader(); w.writerows(layer_rows)
    for row in summary:
        print(f"{row['architecture']}: {row['trainable_parameters']} parameters; {row['GMAC']:.12f} GMAC")


if __name__ == "__main__":
    main()

