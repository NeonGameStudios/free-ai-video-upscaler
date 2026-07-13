const fs = require('node:fs');
const path = require('node:path');
const typescript = require('typescript');

const sourcePath = path.resolve(__dirname, '../src/onnx-webgpu-compat.ts');
const source = fs.readFileSync(sourcePath, 'utf8');
const compiled = typescript.transpileModule(source, {
  compilerOptions: {
    target: typescript.ScriptTarget.ES2020,
    module: typescript.ModuleKind.CommonJS,
  },
}).outputText;
const compatModule = { exports: {} };
new Function('require', 'module', 'exports', compiled)(require, compatModule, compatModule.exports);
const { rewritePReluForWebGPU } = compatModule.exports;

const modelDir = path.resolve(__dirname, '../public/models');
const modelNames = [
  '2x_AnimeJaNai_SD_V1beta34_Compact.onnx',
  '2x_AnimeJaNai_HD_V3_Compact.onnx',
  '2x_AnimeJaNai_HD_V3_UltraCompact.onnx',
  '2x_AnimeJaNai_HD_V3_SuperUltraCompact.onnx',
];

for (const modelName of modelNames) {
  const original = fs.readFileSync(path.join(modelDir, modelName));
  const rewritten = rewritePReluForWebGPU(original.buffer.slice(
    original.byteOffset,
    original.byteOffset + original.byteLength
  ));
  if (rewritten.rewrittenNodes === 0) {
    throw new Error(`${modelName}: no PRelu nodes were rewritten`);
  }
  if (rewritten.data.byteLength === 0) {
    throw new Error(`${modelName}: rewritten model is empty`);
  }
  if (process.argv.includes('--write')) {
    const outputDir = path.resolve(__dirname, '../dist/models');
    fs.mkdirSync(outputDir, { recursive: true });
    fs.writeFileSync(
      path.join(outputDir, modelName.replace(/\.onnx$/i, '.webgpu.onnx')),
      Buffer.from(rewritten.data)
    );
  }
  console.log(JSON.stringify({
    model: modelName,
    originalBytes: original.byteLength,
    rewrittenBytes: rewritten.data.byteLength,
    rewrittenNodes: rewritten.rewrittenNodes,
  }));
}
