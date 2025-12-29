#!/usr/bin/env python3
"""
Convert Real-ESRGAN PyTorch model to ONNX format for browser inference.

This script converts the realesr-animevideov3 model to ONNX format
which can be used with ONNX Runtime Web in the browser.

Usage:
    python convert_model.py

Requirements:
    pip install torch onnx basicsr realesrgan

The script will:
1. Download the realesr-animevideov3.pth model if not present
2. Convert it to ONNX format with dynamic input sizes
3. Optimize the model for web inference
4. Output: models/realesr-animevideov3.onnx
"""

import os
import sys
import urllib.request
import torch
import torch.onnx

# Add Real-ESRGAN to path if installed
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
except ImportError:
    print("Please install required packages:")
    print("  pip install torch onnx basicsr realesrgan")
    sys.exit(1)

# Model URLs and configurations
MODELS = {
    'realesr-animevideov3': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth',
        'scale': 4,
        'arch': 'SRVGGNetCompact',
        'num_feat': 64,
        'num_conv': 16,
    },
    'realesr-animevideov3-x2': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth',
        'scale': 2,
        'arch': 'SRVGGNetCompact',
        'num_feat': 64,
        'num_conv': 16,
    },
}

def download_model(model_name: str, output_dir: str = 'models') -> str:
    """Download the model if not present."""
    os.makedirs(output_dir, exist_ok=True)

    model_info = MODELS[model_name]
    pth_path = os.path.join(output_dir, f'{model_name}.pth')

    if not os.path.exists(pth_path):
        print(f"Downloading {model_name}...")
        urllib.request.urlretrieve(model_info['url'], pth_path)
        print(f"Downloaded to {pth_path}")
    else:
        print(f"Model already exists at {pth_path}")

    return pth_path

def create_model(model_name: str) -> torch.nn.Module:
    """Create the model architecture."""
    model_info = MODELS[model_name]

    if model_info['arch'] == 'SRVGGNetCompact':
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=model_info['num_feat'],
            num_conv=model_info['num_conv'],
            upscale=model_info['scale'],
            act_type='prelu'
        )
    else:
        raise ValueError(f"Unknown architecture: {model_info['arch']}")

    return model

def convert_to_onnx(
    model_name: str,
    output_dir: str = 'models',
    input_height: int = 480,
    input_width: int = 640,
    opset_version: int = 17
) -> str:
    """Convert PyTorch model to ONNX format."""

    # Download and load model
    pth_path = download_model(model_name, output_dir)
    model = create_model(model_name)

    # Load weights
    state_dict = torch.load(pth_path, map_location='cpu')
    if 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
    elif 'params' in state_dict:
        state_dict = state_dict['params']

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_height, input_width)

    # Output path
    onnx_path = os.path.join(output_dir, f'{model_name}.onnx')

    print(f"Converting to ONNX (opset {opset_version})...")

    # Export to ONNX with dynamic axes for flexible input sizes
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'}
        }
    )

    print(f"ONNX model saved to {onnx_path}")

    # Verify the model
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")

    # Print model info
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Model size: {file_size:.2f} MB")

    return onnx_path

def main():
    # Convert the main model
    model_name = 'realesr-animevideov3'
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'public', 'models')

    print(f"Converting {model_name} to ONNX...")
    print(f"Output directory: {output_dir}")

    onnx_path = convert_to_onnx(
        model_name=model_name,
        output_dir=output_dir,
        input_height=480,
        input_width=640,
        opset_version=17
    )

    print(f"\nDone! Model saved to: {onnx_path}")
    print("\nTo use this model in the browser:")
    print("1. Copy the .onnx file to your public/models directory")
    print("2. The app will load it automatically")

if __name__ == '__main__':
    main()
