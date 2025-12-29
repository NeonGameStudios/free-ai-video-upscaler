# AI Upscaling Models

This directory should contain the ONNX model files for video upscaling.

## Available Models

| Model ID | Name | Scale | Denoising | Description |
|----------|------|-------|-----------|-------------|
| `realesrgan-anime-fast` | Real-ESRGAN Anime Fast | 4x | No | Fast anime upscaling (animevideov3) |
| `realesrgan-anime-plus` | Real-ESRGAN Anime Plus | 4x | No | High quality anime upscaling |
| `realesrgan-general-fast` | Real-ESRGAN General Fast | 4x | No | Fast general content upscaling |
| `realesrgan-general-plus` | Real-ESRGAN General Plus | 4x | No | High quality general upscaling |
| `realcugan-2x` | Real-CUGAN 2x | 2x | Yes | Conservative anime with denoising |
| `realcugan-4x` | Real-CUGAN 4x | 4x | Yes | High quality anime with denoising |

## Getting the Models

### Option 1: Convert from PyTorch (Recommended for Real-ESRGAN)

Run the conversion script to download and convert official models:

```bash
cd scripts

# Install dependencies
pip install torch onnx basicsr realesrgan

# Convert a specific model
python convert_model.py realesrgan-anime-fast

# Or convert all Real-ESRGAN models
python convert_model.py --all

# List available models
python convert_model.py --list
```

### Option 2: Download Pre-converted Models

If you have access to pre-converted ONNX models, place them in this directory with the correct filenames:

- `realesrgan-anime-fast.onnx`
- `realesrgan-anime-plus.onnx`
- `realesrgan-general-fast.onnx`
- `realesrgan-general-plus.onnx`
- `realcugan-2x.onnx`
- `realcugan-4x.onnx`

## Model Information

### Real-ESRGAN Models
- **Source**: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- **License**: BSD-3-Clause
- **Architectures**: SRVGGNetCompact (fast), RRDBNet (plus)

### Real-CUGAN Models
- **Source**: [bilibili/ailab](https://github.com/bilibili/ailab/tree/main/Real-CUGAN)
- **License**: MIT
- **Features**: Built-in denoising (levels 0-3)

## File Structure

After adding models, the directory should look like:
```
public/
└── models/
    ├── README.md
    ├── realesrgan-anime-fast.onnx
    ├── realesrgan-anime-plus.onnx
    ├── realesrgan-general-fast.onnx
    ├── realesrgan-general-plus.onnx
    ├── realcugan-2x.onnx
    └── realcugan-4x.onnx
```

## Notes

- Real-CUGAN models require additional conversion steps (see web-realesrgan project)
- Models are ~2-17 MB each depending on architecture
- WebGPU acceleration provides best performance
- WASM fallback available for older browsers
