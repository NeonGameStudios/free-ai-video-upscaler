const assert = require('node:assert/strict');
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

const { calculateTilePlan, calculateTileCopyRegion } = tileModule.exports;

function buildRegions(testCase) {
  const { width, height, tileSize, padding, scale } = testCase;
  const plan = calculateTilePlan(width, height, tileSize, padding);
  const outputWidth = width * scale;
  const outputHeight = height * scale;
  const effectivePadding = Math.min(padding, Math.floor(plan.overlap / 2));
  const regions = [];

  for (let ty = 0; ty < plan.y.count; ty++) {
    for (let tx = 0; tx < plan.x.count; tx++) {
      const sourceX = Math.min(
        tx * plan.x.step,
        Math.max(0, width - plan.x.tileSize),
      );
      const sourceY = Math.min(
        ty * plan.y.step,
        Math.max(0, height - plan.y.tileSize),
      );
      const isLeftEdge = sourceX === 0;
      const isTopEdge = sourceY === 0;
      const isRightEdge = sourceX + plan.x.tileSize >= width;
      const isBottomEdge = sourceY + plan.y.tileSize >= height;
      const cropX = isLeftEdge ? 0 : effectivePadding * scale;
      const cropY = isTopEdge ? 0 : effectivePadding * scale;
      const cropEndX = isRightEdge
        ? plan.x.tileSize * scale
        : (plan.x.tileSize - effectivePadding) * scale;
      const cropEndY = isBottomEdge
        ? plan.y.tileSize * scale
        : (plan.y.tileSize - effectivePadding) * scale;
      const tileOutputX = sourceX * scale;
      const tileOutputY = sourceY * scale;
      const region = calculateTileCopyRegion({
        tileOutputX,
        tileOutputY,
        cropX,
        cropY,
        width: cropEndX - cropX,
        height: cropEndY - cropY,
      });

      assert.equal(region.sourceX, cropX, `${testCase.name}: source X crop changed`);
      assert.equal(region.sourceY, cropY, `${testCase.name}: source Y crop changed`);
      assert.equal(
        region.destinationX,
        sourceX * scale + cropX,
        `${testCase.name}: cropped pixels lost their global X coordinate`,
      );
      assert.equal(
        region.destinationY,
        sourceY * scale + cropY,
        `${testCase.name}: cropped pixels lost their global Y coordinate`,
      );
      assert.ok(region.width > 0 && region.height > 0, `${testCase.name}: empty copy region`);
      assert.ok(region.sourceX >= 0 && region.sourceY >= 0, `${testCase.name}: negative source origin`);
      assert.ok(
        region.sourceX + region.width <= plan.x.tileSize * scale
          && region.sourceY + region.height <= plan.y.tileSize * scale,
        `${testCase.name}: crop exceeds the model output tile`,
      );
      assert.ok(
        region.destinationX >= 0
          && region.destinationY >= 0
          && region.destinationX + region.width <= outputWidth
          && region.destinationY + region.height <= outputHeight,
        `${testCase.name}: copy exceeds the output canvas`,
      );

      regions.push(region);
    }
  }

  assert.ok(regions.length > 1, `${testCase.name}: regression case must exercise multiple tiles`);
  assert.ok(
    regions.some(region => region.sourceX > 0 || region.sourceY > 0),
    `${testCase.name}: regression case must exercise a cropped interior tile`,
  );

  return { plan, regions, outputWidth, outputHeight };
}

function assertFullCoverage(testCase, result) {
  const { regions, outputWidth, outputHeight } = result;
  const yBoundaries = [...new Set(regions.flatMap(region => [
    region.destinationY,
    region.destinationY + region.height,
  ]))].sort((a, b) => a - b);

  assert.equal(yBoundaries[0], 0, `${testCase.name}: top border is uncovered`);
  assert.equal(
    yBoundaries[yBoundaries.length - 1],
    outputHeight,
    `${testCase.name}: bottom border is uncovered`,
  );

  for (let boundaryIndex = 0; boundaryIndex < yBoundaries.length - 1; boundaryIndex++) {
    const y = yBoundaries[boundaryIndex];
    const nextY = yBoundaries[boundaryIndex + 1];
    if (nextY <= y) continue;

    const intervals = regions
      .filter(region => region.destinationY <= y && region.destinationY + region.height > y)
      .map(region => [region.destinationX, region.destinationX + region.width])
      .sort((a, b) => a[0] - b[0] || a[1] - b[1]);

    assert.ok(intervals.length > 0, `${testCase.name}: uncovered horizontal band at y=${y}`);
    let coveredUntil = 0;
    for (const [start, end] of intervals) {
      assert.ok(
        start <= coveredUntil,
        `${testCase.name}: gap from x=${coveredUntil} to x=${start} at y=${y}`,
      );
      coveredUntil = Math.max(coveredUntil, end);
    }
    assert.equal(
      coveredUntil,
      outputWidth,
      `${testCase.name}: right border is uncovered at y=${y}`,
    );
  }
}

const cases = [
  { name: '4x-1080p', width: 1920, height: 1080, tileSize: 512, padding: 32, scale: 4 },
  { name: '4x-mixed-edge', width: 640, height: 360, tileSize: 512, padding: 32, scale: 4 },
  { name: '2x-odd-clamped-edge', width: 853, height: 479, tileSize: 256, padding: 24, scale: 2 },
  { name: '1x-fixed-window', width: 640, height: 360, tileSize: 126, padding: 14, scale: 1 },
];

for (const testCase of cases) {
  const result = buildRegions(testCase);
  assertFullCoverage(testCase, result);
  console.log(JSON.stringify({
    case: testCase.name,
    input: `${testCase.width}x${testCase.height}`,
    output: `${result.outputWidth}x${result.outputHeight}`,
    tiles: `${result.plan.x.count}x${result.plan.y.count}`,
    tileShape: `${result.plan.x.tileSize}x${result.plan.y.tileSize}`,
  }));
}
