import {
  BlobSource,
  Input,
  MP4,
  VideoSampleSink,
} from 'mediabunny';
import { Upscaler } from '../src/upscaler';

const output = document.createElement('pre');
document.body.append(output);

function write(label: string, value: unknown): void {
  output.textContent += `${label} ${typeof value === 'string' ? value : JSON.stringify(value)}\n`;
}

async function run(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const frameLimit = Math.max(1, Number(params.get('frames') || 6));
  const response = await fetch('/test-clips/BotsMaster-15sec.mp4');
  if (!response.ok) throw new Error(`Unable to fetch regression clip (${response.status})`);

  const input = new Input({
    formats: [MP4],
    source: new BlobSource(await response.blob()),
  });
  const track = await input.getPrimaryVideoTrack();
  if (!track) throw new Error('Regression clip has no video track');
  if (!(await track.canDecode())) throw new Error('Regression clip cannot be decoded');

  const outputCanvas = new OffscreenCanvas(track.displayWidth * 2, track.displayHeight * 2);
  const upscaler = new Upscaler({
    modelId: 'animejanai-v3-hd-superfast',
    scale: 2,
    tileSize: 512,
    tilePadding: 32,
  });
  await upscaler.init(outputCanvas, (progress, message) => write('model', `${progress}% ${message}`));

  const frameTimes: number[] = [];
  const sink = new VideoSampleSink(track);
  let provider = upscaler.getExecutionProvider();
  let gpuRenderer = upscaler.isUsingGPUPath();
  let lastTiming = upscaler.getLastTiming();
  let frames = 0;
  let checksum = 0;
  try {
    for await (const sample of sink.samples()) {
      const frame = sample.toVideoFrame();
      const started = performance.now();
      try {
        await upscaler.render(frame);
        frameTimes.push(performance.now() - started);

        const context = outputCanvas.getContext('2d', { alpha: false });
        if (!context) throw new Error('Regression output context unavailable');
        const probe = context.getImageData(
          0,
          0,
          Math.min(64, outputCanvas.width),
          Math.min(64, outputCanvas.height),
        ).data;
        for (let i = 0; i < probe.length; i += 4) {
          checksum = (checksum + probe[i] + probe[i + 1] * 3 + probe[i + 2] * 7) >>> 0;
        }
      } finally {
        frame.close();
        sample.close();
      }

      frames++;
      provider = upscaler.getExecutionProvider();
      gpuRenderer = upscaler.isUsingGPUPath();
      lastTiming = upscaler.getLastTiming();
      if (frames >= frameLimit) break;
    }
  } finally {
    await upscaler.dispose();
    input.dispose();
  }

  if (frames === 0 || checksum === 0) {
    throw new Error(`Regression produced no usable pixels (frames=${frames}, checksum=${checksum})`);
  }

  write('frames', frames);
  write('provider', provider);
  write('gpuRenderer', gpuRenderer);
  write('avgRenderMs', frameTimes.reduce((total, value) => total + value, 0) / frameTimes.length);
  write('lastTiming', lastTiming);
  write('pixelChecksum', checksum);
}

run().catch(error => write('error', error && error.stack ? error.stack : String(error)));
