const path = require('path');

const CopyWebpackPlugin = require('copy-webpack-plugin');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');


/**
 * Webpack is invoked by both `webpack` (production) and `webpack serve`
 * (development).  Keeping the mode in the CLI scripts means webpack can
 * apply the correct optimisations without baking development defaults into
 * production builds.
 */
module.exports = (env, argv = {}) => {
    const mode = argv.mode || process.env.NODE_ENV || 'development';
    const isProduction = mode === 'production';

    return {
    // `worker.ts` is discovered by the Worker(new URL(...)) expression in
    // index.ts.  Listing it here as a second entry compiled the worker twice
    // and emitted an unnecessary copy in the main bundle.
    entry: './src/index.ts',
    output: {
        libraryExport: "default",
        path: path.resolve(__dirname, './dist'),
        filename: "main.js",
        // Worker child compilations and any future async imports get a
        // deterministic filename instead of colliding with main.js.
        chunkFilename: "[name].js",
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
            },

            // Load WGSL shaders as raw strings
            {
                test: /\.wgsl$/,
                type: 'asset/source',
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
                },
                // Copy FFmpeg.wasm core files for local hosting
                {
                    from: "node_modules/@ffmpeg/core/dist/esm/*",
                    to: "ffmpeg/[name][ext]"
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

    // Ignore warnings about dynamic imports in ONNX Runtime and FFmpeg
    ignoreWarnings: [
        {
            module: /onnxruntime-web/,
        },
        {
            module: /@ffmpeg\/ffmpeg/,
        },
    ],

    devServer: {
        static: [
            { directory: path.join(__dirname, 'dist') },
            // Keep local regression clips available to the optional benchmark
            // page without copying them into production builds.
            { directory: path.join(__dirname, 'test-clips'), publicPath: '/test-clips' },
        ],
        compress: true,
        host: '0.0.0.0',  // Listen on all interfaces (required for Docker)
        port: 8080,
        allowedHosts: "all",
        // Required for WebGPU - must use HTTPS
        https: false, // Set to true with certs for production
        headers: {
            // Required headers for SharedArrayBuffer (needed by ONNX Runtime and FFmpeg.wasm)
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Embedder-Policy": "require-corp"
        },
        client: {
            overlay: {
                errors: true,
                warnings: false,  // Don't show warnings in overlay
            },
        },
    },

    mode,
    // Source maps are useful while iterating locally but add a large amount
    // of startup/download work to the deployed bundle.
    devtool: isProduction ? false : 'eval-cheap-module-source-map',
    optimization: {
        // Production mode enables minification and tree-shaking.  Explicitly
        // retaining these defaults documents the intended build behaviour.
        minimize: isProduction,
        moduleIds: 'deterministic',
        chunkIds: 'deterministic',
    },
    };
};
