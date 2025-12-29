const path = require('path');
const webpack = require('webpack');

const CopyWebpackPlugin = require('copy-webpack-plugin');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');


module.exports = {
    entry: ["./src/index.ts", './src/worker.ts'],
    output: {
        libraryExport: "default",
        path: path.resolve(__dirname, './dist'),
        filename: "main.js"
    },
    module: {

        rules: [
            {
                test: /\.(ts|js)$/,
                exclude: /node_modules/,
                use: [
                    {
                        loader: "babel-loader"
                    },
                    {
                        loader: "ts-loader",
                        options: {
                            allowTsInNodeModules: false
                        }
                    }
                ],
            },

            {
                test: /\.css$/i,
                use: ["style-loader", "css-loader", "postcss-loader"],
            }

        ],

    },

    plugins: [

        new HtmlWebpackPlugin({
            template: 'src/index.html'
        }),

        new CleanWebpackPlugin({
            cleanStaleWebpackAssets: false
        }),
        new CopyWebpackPlugin({
            patterns: [
                { from: "src/*.js", to: path.basename('[name].js') },
                { from: "src/img/*.svg", to: path.basename('[name].svg') },
                { from: "src/img/*.png", to: path.basename('[name].png') },
                // Copy ONNX Runtime WASM files for WebGPU support
                {
                    from: "node_modules/onnxruntime-web/dist/*.wasm",
                    to: "[name][ext]"
                },
                {
                    from: "node_modules/onnxruntime-web/dist/*.jsep.*",
                    to: "[name][ext]"
                },
                // Copy models directory
                {
                    from: "public/models",
                    to: "models",
                    noErrorOnMissing: true
                }
            ]
        })

    ],
    resolve: {
        extensions: [".ts", ".tsx", ".js", ".css"],
        fallback: {
            // ONNX Runtime Web may need these polyfills
            "path": false,
            "fs": false
        }
    },

    // Ignore warnings about ONNX Runtime dynamic imports
    ignoreWarnings: [
        {
            module: /onnxruntime-web/,
        },
    ],

    devServer: {
        static: {
            directory: path.join(__dirname, 'dist'),
        },
        compress: true,
        port: 8080,
        allowedHosts: "all",
        // Required for WebGPU - must use HTTPS
        https: false, // Set to true with certs for production
        headers: {
            // Required headers for SharedArrayBuffer (needed by ONNX Runtime)
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Embedder-Policy": "require-corp"
        }
    },

    mode: 'development'

};
