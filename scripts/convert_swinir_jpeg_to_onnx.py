#!/usr/bin/env python3
"""
Convert the official SwinIR color JPEG artifact reduction weights to ONNX.

This script keeps SwinIR JPEG support local because no public browser-ready
SwinIR JPEG ONNX artifact was found. It downloads the official model code and
the DeepInv mirror of the official weights, then writes an ONNX model into
public/models/ for the browser app to serve.

Usage:
    python scripts/convert_swinir_jpeg_to_onnx.py --quality 40

Requirements:
    pip install torch onnx timm
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
from pathlib import Path
import ssl
import sys
import urllib.request

import torch
import torch.onnx


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "public" / "models"
BUILD_DIR = REPO_ROOT / "models" / "swinir"
NETWORK_URL = "https://api.github.com/repos/JingyunLiang/SwinIR/contents/models/network_swinir.py"
WEIGHT_BASE_URL = "https://huggingface.co/deepinv/swinir/resolve/main"

try:
    import certifi

    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    SSL_CONTEXT = ssl.create_default_context()


def download(url: str, path: Path) -> None:
    if path.exists():
        print(f"Using existing {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, context=SSL_CONTEXT) as response:
        path.write_bytes(response.read())


def download_network(path: Path) -> None:
    if path.exists():
        print(f"Using existing {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official SwinIR model code")
    with urllib.request.urlopen(NETWORK_URL, context=SSL_CONTEXT) as response:
        payload = json.load(response)

    path.write_bytes(base64.b64decode(payload["content"]))


def load_swinir_class(network_path: Path):
    spec = importlib.util.spec_from_file_location("network_swinir", network_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {network_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["network_swinir"] = module
    spec.loader.exec_module(module)
    return module.SwinIR


def create_model(SwinIR):
    return SwinIR(
        upscale=1,
        in_chans=3,
        img_size=126,
        window_size=7,
        img_range=255.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="",
        resi_connection="1conv",
    )


def load_weights(model: torch.nn.Module, weight_path: Path) -> None:
    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = checkpoint.get("params", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()


def convert(quality: int, opset: int) -> Path:
    if quality not in {10, 20, 30, 40}:
        raise ValueError("--quality must be one of 10, 20, 30, or 40")

    network_path = BUILD_DIR / "network_swinir.py"
    weight_name = f"006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg{quality}.pth"
    weight_path = BUILD_DIR / weight_name
    output_path = MODEL_DIR / weight_name.replace(".pth", ".onnx")

    download_network(network_path)
    download(f"{WEIGHT_BASE_URL}/{weight_name}", weight_path)

    SwinIR = load_swinir_class(network_path)
    model = create_model(SwinIR)
    load_weights(model, weight_path)

    dummy = torch.randn(1, 3, 126, 126)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {output_path}")
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )

    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    size_mib = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_path} ({size_mib:.1f} MiB)")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", type=int, default=40)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    convert(args.quality, args.opset)


if __name__ == "__main__":
    main()
