const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const typescript = require('typescript');

const sourcePath = path.resolve(__dirname, '../src/types/worker-messages.ts');
const source = fs.readFileSync(sourcePath, 'utf8');
const compiled = typescript.transpileModule(source, {
  compilerOptions: {
    target: typescript.ScriptTarget.ES2020,
    module: typescript.ModuleKind.CommonJS,
  },
}).outputText;
const messageModule = { exports: {} };
new Function('module', 'exports', compiled)(messageModule, messageModule.exports);

const {
  ceilToEven,
  resolveInferenceResolution,
  resolveOutputResolution,
} = messageModule.exports;

assert.equal(ceilToEven(1), 2);
assert.equal(ceilToEven(2), 2);
assert.equal(ceilToEven(3), 4);
assert.equal(ceilToEven(4.1), 6);
assert.equal(ceilToEven(Number.NaN), 2);

const outputCases = [
  {
    name: 'odd-1x-source',
    source: { width: 1921, height: 1081 },
    scale: 1,
    targetHeight: 1081,
    expected: { width: 1922, height: 1082 },
  },
  {
    name: 'odd-cleanup-source',
    source: { width: 641, height: 481 },
    scale: 1,
    targetHeight: 481,
    expected: { width: 642, height: 482 },
  },
  {
    name: 'odd-high-resolution-auto',
    source: { width: 3841, height: 2161 },
    scale: 4,
    targetHeight: 2161,
    expected: { width: 3842, height: 2162 },
  },
  {
    name: 'omitted-target-conservative-fallback',
    source: { width: 1920, height: 1080 },
    scale: 4,
    targetHeight: undefined,
    expected: { width: 1920, height: 1080 },
  },
  {
    name: 'preset-does-not-exceed-native-scale',
    source: { width: 640, height: 360 },
    scale: 4,
    targetHeight: 2160,
    expected: { width: 2560, height: 1440 },
  },
  {
    name: 'capped-odd-source',
    source: { width: 1921, height: 1081 },
    scale: 4,
    targetHeight: 1080,
    expected: { width: 1920, height: 1080 },
  },
];

for (const testCase of outputCases) {
  const actual = resolveOutputResolution(
    testCase.source,
    testCase.scale,
    testCase.targetHeight,
  );
  assert.deepEqual(actual, testCase.expected, testCase.name);
  console.log(JSON.stringify({ case: testCase.name, output: actual }));
}

const inferenceCases = [
  {
    name: 'cleanup-keeps-native-odd-input',
    source: { width: 641, height: 481 },
    scale: 1,
    encode: { width: 642, height: 482 },
    expected: { width: 641, height: 481 },
  },
  {
    name: 'scaled-source-rounds-inference-up',
    source: { width: 641, height: 481 },
    scale: 4,
    encode: { width: 642, height: 482 },
    expected: { width: 161, height: 121 },
  },
  {
    name: 'native-model-output-keeps-source-input',
    source: { width: 641, height: 481 },
    scale: 2,
    encode: { width: 1282, height: 962 },
    expected: { width: 641, height: 481 },
  },
  {
    name: 'capped-output-never-undershoots',
    source: { width: 1921, height: 1081 },
    scale: 4,
    encode: { width: 1920, height: 1080 },
    expected: { width: 480, height: 270 },
  },
];

for (const testCase of inferenceCases) {
  const actual = resolveInferenceResolution(
    testCase.source,
    testCase.scale,
    testCase.encode,
  );
  assert.deepEqual(actual, testCase.expected, testCase.name);
  console.log(JSON.stringify({ case: testCase.name, inference: actual }));
}
