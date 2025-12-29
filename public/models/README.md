# Real-ESRGAN Model Setup

This directory should contain the ONNX model file for Real-ESRGAN.

## Getting the Model

### Option 1: Convert from PyTorch (Recommended)

Run the conversion script to download and convert the official model:

```bash
cd scripts
pip install torch onnx basicsr realesrgan
python convert_model.py
```

This will:
1. Download the official `realesr-animevideov3.pth` from the Real-ESRGAN repository
2. Convert it to ONNX format with dynamic input sizes
3. Save as `realesr-animevideov3.onnx` in this directory

### Option 2: Download Pre-converted Model

If you have access to a pre-converted ONNX model:
1. Download `realesr-animevideov3.onnx`
2. Place it in this directory (`public/models/`)

## Model Information

- **Model**: realesr-animevideov3
- **Architecture**: SRVGGNetCompact
- **Scale**: 4x upscaling
- **Size**: ~2.5 MB (ONNX format)
- **Optimized for**: Anime and animated content
- **License**: BSD-3-Clause (from xinntao/Real-ESRGAN)

## Verification

After placing the model, the file structure should be:
```
public/
└── models/
    └── realesr-animevideov3.onnx
```

## References

- [Real-ESRGAN GitHub](https://github.com/xinntao/Real-ESRGAN)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
