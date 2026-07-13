import { Upscaler } from '../src/upscaler';
import type { ModelType } from '../src/types/worker-messages';

const output = document.createElement('pre');
document.body.append(output);

function write(label: string, value: unknown): void {
  output.textContent += `${label} ${typeof value === 'string' ? value : JSON.stringify(value)}\n`;
}

async function run(): Promise<void> {
  write('webgpu', Boolean((navigator as any).gpu));
  const requestedModel = new URLSearchParams(location.search).get('model') as ModelType | null;
  const modelId: ModelType = requestedModel || 'realesr-animevideov3';
  const scale = modelId.startsWith('animejanai') ? 2 : 4;
  write('modelId', modelId);
  const input = new OffscreenCanvas(128, 128);
  const inputContext = input.getContext('2d', { alpha: false });
  if (!inputContext) throw new Error('Unable to create benchmark input canvas');

  const pixels = new ImageData(128, 128);
  for (let y = 0; y < 128; y++) {
    for (let x = 0; x < 128; x++) {
      const offset = (y * 128 + x) * 4;
      pixels.data[offset] = x * 2;
      pixels.data[offset + 1] = y * 2;
      pixels.data[offset + 2] = 128;
      pixels.data[offset + 3] = 255;
    }
  }
  inputContext.putImageData(pixels, 0, 0);

  const outputCanvas = new OffscreenCanvas(128 * scale, 128 * scale);
  const upscaler = new Upscaler({
    modelId,
    scale,
    tileSize: 128,
    tilePadding: 16,
  });

  const modelStarted = performance.now();
  await upscaler.init(outputCanvas, (progress, message) => {
    write('model', `${progress}% ${message}`);
  });
  write('modelMs', performance.now() - modelStarted);
  write('executionProvider', upscaler.getExecutionProvider());
  write('gpuRenderer', upscaler.isUsingGPUPath());

  const renderStarted = performance.now();
  await upscaler.render(input);
  write('renderMs', performance.now() - renderStarted);
  write('timing', upscaler.getLastTiming());

  const resultContext = outputCanvas.getContext('2d', { alpha: false });
  const sample = resultContext?.getImageData(0, 0, 1, 1).data;
  write('samplePixel', sample ? Array.from(sample) : null);
  await upscaler.dispose();
}

run().catch(error => write('error', error && error.stack ? error.stack : String(error)));
