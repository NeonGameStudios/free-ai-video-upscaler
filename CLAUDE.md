# Claude Code Guidelines

## Git Configuration

This is a fork of sb2702/free-ai-video-upscaler.

- **origin** = NeonGameStudios/free-ai-video-upscaler (your fork) - push here
- **upstream** = sb2702/free-ai-video-upscaler (original) - fetch only, push disabled

Always push to `origin`, never to `upstream`. PRs should target `NeonGameStudios/free-ai-video-upscaler:main`.

## Build & Test

```bash
npm install    # Install dependencies
npm run dev    # Development server
npm run build  # Production build
```

## Project Structure

- `src/` - TypeScript source
- `public/models/` - ONNX model files (AnimeJaNai models included locally)
- `scripts/` - Model conversion utilities
