const projectConfig = require('../webpack.config.js')({}, { mode: 'development' });
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyWebpackPlugin = require('copy-webpack-plugin');

module.exports = {
  ...projectConfig,
  entry: './scripts/encoding-benchmark.ts',
  output: {
    ...projectConfig.output,
    filename: 'encoding-benchmark.[contenthash].js',
  },
  module: {
    ...projectConfig.module,
    rules: [
      ...projectConfig.module.rules,
      {
        test: /\.mp4$/,
        type: 'asset/inline',
      },
    ],
  },
  plugins: [
    new HtmlWebpackPlugin({
      filename: 'encoding-benchmark.html',
      templateContent: '<!doctype html><meta charset="utf-8"><title>Encoding benchmark</title>',
    }),
    new CopyWebpackPlugin({
      patterns: [
        { from: 'node_modules/onnxruntime-web/dist/*.wasm', to: '[name][ext]' },
        { from: 'node_modules/onnxruntime-web/dist/*.jsep.*', to: '[name][ext]' },
        { from: 'public/models', to: 'models' },
      ],
    }),
  ],
  devServer: projectConfig.devServer,
  mode: 'development',
  devtool: false,
};
