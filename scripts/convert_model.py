#!/usr/bin/env python3
"""
Convert Real-ESRGAN and Real-CUGAN PyTorch models to ONNX format for browser inference.

This script converts models to ONNX format which can be used with ONNX Runtime Web.

Usage:
    python convert_model.py [model_name]
    python convert_model.py --all

Models:
    - realesrgan-anime-fast   (Real-ESRGAN animevideov3, 4x)
    - realesrgan-anime-plus   (Real-ESRGAN anime, 4x)
    - realesrgan-general-fast (Real-ESRGAN general, 4x)
    - realesrgan-general-plus (Real-ESRGAN plus, 4x)
    - realcugan-2x            (Real-CUGAN, 2x with denoising)
    - realcugan-4x            (Real-CUGAN, 4x with denoising)

Requirements:
    pip install torch onnx basicsr realesrgan

Output: public/models/<model_name>.onnx
"""

import os
import sys
import argparse
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

# Model configurations
MODELS = {
    'realesrgan-anime-fast': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth',
        'scale': 4,
        'arch': 'SRVGGNetCompact',
        'num_feat': 64,
        'num_conv': 16,
        'description': 'Real-ESRGAN Anime Fast (animevideov3) - Fast anime upscaling'
    },
    'realesrgan-anime-plus': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
        'scale': 4,
        'arch': 'RRDBNet',
        'num_feat': 64,
        'num_block': 6,
        'description': 'Real-ESRGAN Anime Plus - High quality anime upscaling'
    },
    'realesrgan-general-fast': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
        'scale': 4,
        'arch': 'SRVGGNetCompact',
        'num_feat': 64,
        'num_conv': 32,
        'description': 'Real-ESRGAN General Fast - Fast general content upscaling'
    },
    'realesrgan-general-plus': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        'scale': 4,
        'arch': 'RRDBNet',
        'num_feat': 64,
        'num_block': 23,
        'description': 'Real-ESRGAN General Plus - High quality general upscaling'
    },
    'realcugan-2x': {
        'url': 'https://github.com/bilibili/ailab/releases/download/Real-CUGAN/up2x-latest-denoise3x.pth',
        'scale': 2,
        'arch': 'CUGAN',
        'pro': False,
        'description': 'Real-CUGAN 2x - Conservative anime upscaling with denoising'
    },
    'realcugan-4x': {
        'url': 'https://github.com/bilibili/ailab/releases/download/Real-CUGAN/up4x-latest-denoise3x.pth',
        'scale': 4,
        'arch': 'CUGAN',
        'pro': False,
        'description': 'Real-CUGAN 4x - High quality anime upscaling with denoising'
    }
}


def download_model(model_name: str, output_dir: str = 'models') -> str:
    """Download the model if not present."""
    os.makedirs(output_dir, exist_ok=True)

    model_info = MODELS[model_name]
    pth_path = os.path.join(output_dir, f'{model_name}.pth')

    if not os.path.exists(pth_path):
        print(f"Downloading {model_name}...")
        print(f"  From: {model_info['url']}")
        urllib.request.urlretrieve(model_info['url'], pth_path)
        print(f"  Saved to: {pth_path}")
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
    elif model_info['arch'] == 'RRDBNet':
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=model_info['num_feat'],
            num_block=model_info['num_block'],
            num_grow_ch=32,
            scale=model_info['scale']
        )
    elif model_info['arch'] == 'CUGAN':
        # Real-CUGAN has a different architecture
        # For now, we'll use a simplified approach
        print(f"Note: Real-CUGAN models require the cugan package.")
        print("Install with: pip install realcugan-ncnn-py")
        print("Or use the web-realesrgan TensorFlow.js version")
        raise NotImplementedError("Real-CUGAN conversion requires additional setup")
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

    model_info = MODELS[model_name]
    print(f"\n{'='*60}")
    print(f"Converting: {model_name}")
    print(f"Description: {model_info['description']}")
    print(f"Scale: {model_info['scale']}x")
    print(f"{'='*60}\n")

    # Download and load model
    pth_path = download_model(model_name, output_dir)
    model = create_model(model_name)

    # Load weights
    print("Loading weights...")
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
    parser = argparse.ArgumentParser(
        description='Convert Real-ESRGAN/Real-CUGAN models to ONNX format'
    )
    parser.add_argument(
        'model',
        nargs='?',
        choices=list(MODELS.keys()),
        help='Model name to convert'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Convert all available models'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available models'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory (default: public/models)'
    )

    args = parser.parse_args()

    # Default output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'public', 'models'
        )

    # List models
    if args.list:
        print("\nAvailable models:\n")
        for name, info in MODELS.items():
            print(f"  {name}")
            print(f"    Scale: {info['scale']}x")
            print(f"    Arch: {info['arch']}")
            print(f"    Description: {info['description']}")
            print()
        return

    # Convert all models
    if args.all:
        print(f"Converting all models to: {output_dir}\n")
        for model_name in MODELS.keys():
            try:
                convert_to_onnx(model_name, output_dir)
            except NotImplementedError as e:
                print(f"Skipping {model_name}: {e}")
            except Exception as e:
                print(f"Error converting {model_name}: {e}")
        return

    # Convert single model
    if args.model:
        print(f"Output directory: {output_dir}\n")
        convert_to_onnx(args.model, output_dir)
        return

    # No arguments - show help
    parser.print_help()
    print("\nExamples:")
    print("  python convert_model.py realesrgan-anime-fast")
    print("  python convert_model.py --all")
    print("  python convert_model.py --list")


if __name__ == '__main__':
    main()
