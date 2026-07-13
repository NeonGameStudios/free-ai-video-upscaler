const fs = require('node:fs');
const path = require('node:path');
const typescript = require('typescript');

const sourcePath = path.resolve(__dirname, '../src/tiling.ts');
const source = fs.readFileSync(sourcePath, 'utf8');
const compiled = typescript.transpileModule(source, {
  compilerOptions: {
    target: typescript.ScriptTarget.ES2020,
    module: typescript.ModuleKind.CommonJS,
  },
}).outputText;
const tileModule = { exports: {} };
new Function('module', 'exports', compiled)(tileModule, tileModule.exports);
const { calculateTilePlan } = tileModule.exports;

function legacyInferredPixels(width, height, tileSize, tilePadding) {
  const overlap = Math.min(tilePadding * 2, tileSize - 1);
  const step = tileSize - overlap;
  const tilesX = width <= tileSize ? 1 : Math.ceil((width - overlap) / step);
  const tilesY = height <= tileSize ? 1 : Math.ceil((height - overlap) / step);
  let pixels = 0;

  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const x = width <= tileSize ? 0 : Math.min(tx * step, width - tileSize);
      const y = height <= tileSize ? 0 : Math.min(ty * step, height - tileSize);
      pixels += Math.min(tileSize, width - x) * Math.min(tileSize, height - y);
    }
  }
  return pixels;
}

const cases = [
  { width: 640, height: 360, tileSize: 512, padding: 32 },
  { width: 1280, height: 720, tileSize: 512, padding: 32 },
  { width: 1920, height: 1080, tileSize: 512, padding: 32 },
  { width: 1920, height: 1080, tileSize: 1024, padding: 32 },
];

for (const testCase of cases) {
  const plan = calculateTilePlan(
    testCase.width,
    testCase.height,
    testCase.tileSize,
    testCase.padding
  );
  const inferredPixels = plan.x.count * plan.y.count * plan.x.tileSize * plan.y.tileSize;
  const legacyPixels = legacyInferredPixels(
    testCase.width,
    testCase.height,
    testCase.tileSize,
    testCase.padding
  );
  const coveredWidth = plan.x.tileSize + (plan.x.count - 1) * plan.x.step;
  const coveredHeight = plan.y.tileSize + (plan.y.count - 1) * plan.y.step;
  const xStarts = Array.from({ length: plan.x.count }, (_, index) =>
    Math.min(index * plan.x.step, Math.max(0, testCase.width - plan.x.tileSize))
  );
  const yStarts = Array.from({ length: plan.y.count }, (_, index) =>
    Math.min(index * plan.y.step, Math.max(0, testCase.height - plan.y.tileSize))
  );

  if (coveredWidth < testCase.width || coveredHeight < testCase.height) {
    throw new Error(`Tile plan does not cover ${testCase.width}x${testCase.height}`);
  }
  if (xStarts.some((start, index) => index > 0 && start > xStarts[index - 1] + plan.x.tileSize)) {
    throw new Error(`Tile plan has a horizontal gap for ${testCase.width}x${testCase.height}`);
  }
  if (yStarts.some((start, index) => index > 0 && start > yStarts[index - 1] + plan.y.tileSize)) {
    throw new Error(`Tile plan has a vertical gap for ${testCase.width}x${testCase.height}`);
  }
  if (plan.x.tileSize > testCase.tileSize || plan.y.tileSize > testCase.tileSize) {
    throw new Error(`Tile plan exceeds maximum tile size for ${testCase.width}x${testCase.height}`);
  }
  if (inferredPixels > legacyPixels) {
    throw new Error(`Adaptive plan regressed inference area for ${testCase.width}x${testCase.height}`);
  }

  const reduction = ((legacyPixels - inferredPixels) / legacyPixels) * 100;
  console.log(JSON.stringify({
    input: `${testCase.width}x${testCase.height}`,
    tile: testCase.tileSize,
    counts: `${plan.x.count}x${plan.y.count}`,
    adaptiveTile: `${plan.x.tileSize}x${plan.y.tileSize}`,
    legacyPixels,
    inferredPixels,
    reductionPercent: Number(reduction.toFixed(1)),
  }));
}
