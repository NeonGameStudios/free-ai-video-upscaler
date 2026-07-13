const projectConfig = require('../webpack.config.js')({}, { mode: 'development' });
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyWebpackPlugin = require('copy-webpack-plugin');

module.exports = {
  ...projectConfig,
  entry: './scripts/clip-regression.ts',
  output: {
    ...projectConfig.output,
    filename: 'clip-regression.js',
  },
  plugins: [
    new HtmlWebpackPlugin({
      filename: 'clip-regression.html',
      templateContent: '<!doctype html><meta charset="utf-8"><title>Clip regression</title>',
    }),
    new CopyWebpackPlugin({
      patterns: [
        { from: 'node_modules/onnxruntime-web/dist/*.wasm', to: '[name][ext]' },
        { from: 'node_modules/onnxruntime-web/dist/*.jsep.*', to: '[name][ext]' },
        { from: 'public/models', to: 'models' },
      ],
    }),
  ],
  mode: 'development',
  devtool: false,
};
