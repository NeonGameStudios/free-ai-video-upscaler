/**
 * Free AI Video Upscaler - Main Application
 *
 * Browser-based video upscaling using Real-ESRGAN and Real-CUGAN models.
 * All processing happens locally in the browser using WebGPU acceleration.
 */

import Alpine from 'alpinejs';
import ImageCompare from './lib/image-compare-viewer.min';
import {
  AVAILABLE_MODELS,
  OUTPUT_FORMATS,
  RESOLUTION_PRESETS,
  getModelInfo,
  getFormatInfo,
  getResolutionPreset,
} from './types/worker-messages';
import type {
  WorkerRequestMessage,
  WorkerResponseMessage,
  ModelType,
  ModelInfo,
  DenoiseLevel,
  OutputFormat,
  OutputResolution,
  ModelConfig,
} from './types/worker-messages';

import 'bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import "./index.css";
import "./lib/image-compare-viewer.min.css";

// Web Worker for video processing
const worker = new Worker(new URL('./worker.ts', import.meta.url));

// Canvas and video elements
let upscaled_canvas: HTMLCanvasElement;
let original_canvas: HTMLCanvasElement;
let video: HTMLVideoElement;
let previewBitmap: ImageBitmap | null = null;

// Current settings
let currentModel: ModelType = 'realesrgan-anime-fast';
let currentDenoiseLevel: DenoiseLevel = 0;
let currentOutputFormat: OutputFormat = 'mp4';
let currentOutputResolution: OutputResolution = 'auto';

// Video data
let download_name: string;
let inputFileHandle: FileSystemFileHandle;

// Declare global window functions for Alpine to call and File System Access API
declare global {
    interface Window {
        chooseFile: (e?: Event) => Promise<void>;
        initRecording: () => Promise<void>;
        fullScreenPreview: (e?: Event) => Promise<void>;
        onModelChange: (modelId: string) => Promise<void>;
        onDenoiseChange: (level: number) => void;
        onFormatChange: (format: string) => void;
        onResolutionChange: (resolution: string) => void;
        showSaveFilePicker: (options?: any) => Promise<FileSystemFileHandle>;
        showOpenFilePicker: (options?: any) => Promise<FileSystemFileHandle[]>;
    }
}

document.addEventListener("DOMContentLoaded", index);

//===================  Initial Load ===========================

/**
 * Main initialization function called on page load.
 */
async function index(): Promise<void> {
    Alpine.store('state', 'init');

    // Initialize settings stores
    Alpine.store('models', AVAILABLE_MODELS);
    Alpine.store('formats', OUTPUT_FORMATS);
    Alpine.store('resolutions', RESOLUTION_PRESETS);

    Alpine.store('selectedModel', currentModel);
    Alpine.store('selectedDenoise', currentDenoiseLevel);
    Alpine.store('selectedFormat', currentOutputFormat);
    Alpine.store('selectedResolution', currentOutputResolution);

    // Get initial model info
    const modelInfo = getModelInfo(currentModel);
    Alpine.store('currentScale', modelInfo?.scale || 4);
    Alpine.store('supportsDenoising', modelInfo?.supportsDenoising || false);

    Alpine.start();
    document.body.style.display = "block";

    upscaled_canvas = document.getElementById("upscaled") as HTMLCanvasElement;
    original_canvas = document.getElementById('original') as HTMLCanvasElement;

    if (!("VideoEncoder" in window)) return showUnsupported("WebCodecs");

    if (!window.showSaveFilePicker) return showUnsupported("File Write System API");

    worker.postMessage({ cmd: 'isSupported' } satisfies WorkerRequestMessage);

    // Set up global functions
    window.chooseFile = chooseFile;
    window.onModelChange = onModelChange;
    window.onDenoiseChange = onDenoiseChange;
    window.onFormatChange = onFormatChange;
    window.onResolutionChange = onResolutionChange;
}

/**
 * Show unsupported browser feature message.
 */
function showUnsupported(text: string): void {
    Alpine.store('component', text);
    Alpine.store('state', 'unsupported');
}

/**
 * Prompt user to choose a video file using File System Access API.
 */
async function chooseFile(e?: Event): Promise<void> {
    try {
        const [fileHandle] = await window.showOpenFilePicker({
            types: [{
                description: 'Video Files',
                accept: { 'video/mp4': ['.mp4'] }
            }],
            multiple: false
        });

        await loadVideo(fileHandle);
    } catch (e) {
        // User cancelled file picker
        console.log('File selection cancelled');
    }
}

//===================  Settings Handlers ===========================

/**
 * Handle model selection change.
 */
async function onModelChange(modelId: string): Promise<void> {
    const modelInfo = getModelInfo(modelId as ModelType);
    if (!modelInfo) return;

    currentModel = modelId as ModelType;
    Alpine.store('selectedModel', currentModel);
    Alpine.store('currentScale', modelInfo.scale);
    Alpine.store('supportsDenoising', modelInfo.supportsDenoising);

    // Reset denoise level if model doesn't support it
    if (!modelInfo.supportsDenoising) {
        currentDenoiseLevel = 0;
        Alpine.store('selectedDenoise', 0);
    }

    // Update output dimensions display
    if (video) {
        updateOutputDimensions();
    }

    // If we have a preview, switch the model
    if (previewBitmap && Alpine.store('state') === 'preview') {
        Alpine.store('state', 'loading');
        Alpine.store('loading_message', 'Switching model...');

        const modelConfig = getModelConfig();

        worker.postMessage({
            cmd: 'switchModel',
            data: {
                bitmap: previewBitmap,
                modelConfig
            }
        } satisfies WorkerRequestMessage);
    }
}

/**
 * Handle denoise level change.
 */
function onDenoiseChange(level: number): void {
    currentDenoiseLevel = level as DenoiseLevel;
    Alpine.store('selectedDenoise', currentDenoiseLevel);
}

/**
 * Handle output format change.
 */
function onFormatChange(format: string): void {
    currentOutputFormat = format as OutputFormat;
    Alpine.store('selectedFormat', currentOutputFormat);
    updateDownloadName();
}

/**
 * Handle output resolution change.
 */
function onResolutionChange(resolution: string): void {
    currentOutputResolution = resolution as OutputResolution;
    Alpine.store('selectedResolution', currentOutputResolution);
    updateOutputDimensions();
}

/**
 * Get current model configuration.
 */
function getModelConfig(): ModelConfig {
    const modelInfo = getModelInfo(currentModel);
    return {
        modelPath: `/models/${modelInfo?.modelFile || 'realesrgan-anime-fast.onnx'}`,
        scale: modelInfo?.scale || 4,
        tileSize: 256,
        tilePadding: 16,
        denoiseLevel: modelInfo?.supportsDenoising ? currentDenoiseLevel : undefined,
    };
}

/**
 * Update output dimensions display based on current settings.
 */
function updateOutputDimensions(): void {
    if (!video) return;

    const modelInfo = getModelInfo(currentModel);
    const scale = modelInfo?.scale || 4;
    const resPreset = getResolutionPreset(currentOutputResolution);

    let outputWidth = video.videoWidth * scale;
    let outputHeight = video.videoHeight * scale;

    // Apply resolution limit if not auto
    if (resPreset && resPreset.maxHeight !== null && outputHeight > resPreset.maxHeight) {
        const aspectRatio = outputWidth / outputHeight;
        outputHeight = resPreset.maxHeight;
        outputWidth = Math.round(outputHeight * aspectRatio);
    }

    Alpine.store('outputWidth', outputWidth);
    Alpine.store('outputHeight', outputHeight);
}

/**
 * Update download filename based on current settings.
 */
function updateDownloadName(): void {
    if (!video) return;

    const formatInfo = getFormatInfo(currentOutputFormat);
    const modelInfo = getModelInfo(currentModel);
    const baseName = (Alpine.store('filename') as string)?.split(".")[0] || 'video';

    download_name = `${baseName}-upscaled-${modelInfo?.scale || 4}x${formatInfo?.extension || '.mp4'}`;
    Alpine.store('download_name', download_name);
}

//===================  Preview ===========================

/**
 * Load video file from FileSystemFileHandle.
 */
async function loadVideo(fileHandle: FileSystemFileHandle): Promise<void> {
    Alpine.store('state', 'loading');
    Alpine.store('loading_message', 'Loading video...');

    // Store the file handle for later processing
    inputFileHandle = fileHandle;

    // Get the file to create a preview
    const file = await fileHandle.getFile();

    // Set up initial filename
    Alpine.store('filename', file.name);
    updateDownloadName();

    // Read file for preview setup
    const arrayBuffer = await file.arrayBuffer();
    await setupPreview(arrayBuffer);
}

/**
 * Set up the preview UI with before/after comparison.
 */
async function setupPreview(data: ArrayBuffer): Promise<void> {
    video = document.createElement('video');

    const fileBlob = new Blob([data], { type: "video/mp4" });

    video.src = URL.createObjectURL(fileBlob);

    const imageCompare = document.getElementById('image-compare-outer') as HTMLElement;

    video.onloadeddata = async function () {
        Alpine.store('width', video.videoWidth);
        Alpine.store('height', video.videoHeight);

        // Update output dimensions
        updateOutputDimensions();

        const modelInfo = getModelInfo(currentModel);
        const scale = modelInfo?.scale || 4;

        // Set canvas dimensions
        upscaled_canvas.width = video.videoWidth * scale;
        upscaled_canvas.height = video.videoHeight * scale;
        original_canvas.width = video.videoWidth * scale;
        original_canvas.height = video.videoHeight * scale;

        imageCompare.style.height = '318px';
        imageCompare.style.width = `${Math.round(video.videoWidth / video.videoHeight * 318)}px`;
        imageCompare.style.margin = 'auto';
        imageCompare.style.position = 'relative';

        new ImageCompare(document.getElementById('image-compare')).mount();
        video.currentTime = video.duration * 0.2 || 0;

        if (video.requestVideoFrameCallback) {
            video.requestVideoFrameCallback(showPreview);
        } else {
            requestAnimationFrame(showPreview);
        }
    };

    async function showPreview() {
        const fullScreenButton = document.getElementById('full-screen');

        window.initRecording = initRecording;
        window.fullScreenPreview = fullScreenPreview;

        // Store bitmap for model switching
        previewBitmap = await createImageBitmap(video);

        const upscaled = upscaled_canvas.transferControlToOffscreen();
        const original = original_canvas.transferControlToOffscreen();

        Alpine.store('loading_message', 'Loading AI model...');

        const modelConfig = getModelConfig();

        worker.postMessage({
            cmd: "init",
            data: {
                bitmap: previewBitmap,
                upscaled,
                original,
                resolution: {
                    width: video.videoWidth,
                    height: video.videoHeight
                },
                modelConfig
            }
        } satisfies WorkerRequestMessage, [upscaled, original]);

        function setFullScreenLocation() {
            const containerWidth = Math.round(video.videoWidth / video.videoHeight * 318);
            const containerHeight = 318;

            // Position at bottom-right of the preview container (with small padding)
            fullScreenButton!.style.left = `${imageCompare.offsetLeft + containerWidth - 20}px`;
            fullScreenButton!.style.top = `${imageCompare.offsetTop + containerHeight - 20}px`;
        }

        setTimeout(setFullScreenLocation, 20);
        setTimeout(setFullScreenLocation, 60);
        setTimeout(setFullScreenLocation, 200);

        imageCompare.addEventListener('fullscreenchange', function () {
            if (!document.fullscreenElement) {
                // Reset canvas styles
                upscaled_canvas.style.width = ``;
                upscaled_canvas.style.height = ``;
                original_canvas.style.width = ``;
                original_canvas.style.height = ``;

                // Reset container styles to original preview dimensions
                const imageCompareOuter = document.getElementById('image-compare-outer')!;
                const imageCompareInner = document.getElementById('image-compare')!;

                // Reset outer container
                imageCompareOuter.style.width = ``;
                imageCompareOuter.style.height = ``;
                imageCompareOuter.style.backgroundColor = ``;
                imageCompareOuter.style.display = ``;
                imageCompareOuter.style.justifyContent = ``;
                imageCompareOuter.style.alignItems = ``;

                // Reset inner container to original preview size
                imageCompareInner.style.height = '318px';
                imageCompareInner.style.width = `${Math.round(video.videoWidth / video.videoHeight * 318)}px`;
                imageCompareInner.style.margin = 'auto';
                imageCompareInner.style.position = 'relative';
            }
        });

        let bitrate = getBitrate();

        const estimated_size = (bitrate / 8) * video.duration + (128 / 8) * video.duration; // Assume 128 kbps audio

        if (estimated_size > 1900 * 1024 * 1024) {
            Alpine.store('target', 'writer');
        } else {
            Alpine.store('target', 'blob');
        }

        const quota = (await navigator.storage.estimate()).quota;

        if (estimated_size > quota!) {
            return showError(`The video is too big. It would output a file of ${humanFileSize(estimated_size)} but the browser can only write files up to ${humanFileSize(quota!)}`);
        }

        Alpine.store('size', humanFileSize(estimated_size));

        function canvasFullScreen() {
            // Calculate aspect ratios
            const videoAspectRatio = video.videoWidth / video.videoHeight;
            const screenAspectRatio = window.innerWidth / window.innerHeight;

            let displayWidth, displayHeight;

            const imageCompareOuter = document.getElementById('image-compare-outer')!;
            const imageCompareInner = document.getElementById('image-compare')!;

            // If video is wider than screen, fit to width (letterbox on top/bottom)
            if (videoAspectRatio > screenAspectRatio) {
                displayWidth = window.innerWidth;
                displayHeight = window.innerWidth / videoAspectRatio;
            }
            // If video is taller than screen, fit to height (pillarbox on sides)
            else {
                displayWidth = window.innerHeight * videoAspectRatio;
                displayHeight = window.innerHeight;
            }

            // Style the outer container to fill screen with black background and center content
            imageCompareOuter.style.width = `${window.innerWidth}px`;
            imageCompareOuter.style.height = `${window.innerHeight}px`;
            imageCompareOuter.style.backgroundColor = 'black';
            imageCompareOuter.style.display = 'flex';
            imageCompareOuter.style.justifyContent = 'center';
            imageCompareOuter.style.alignItems = 'center';

            // Size the inner container to maintain aspect ratio
            imageCompareInner.style.width = `${displayWidth}px`;
            imageCompareInner.style.height = `${displayHeight}px`;

            // Let the canvases fill their parent container
            upscaled_canvas.style.width = `${displayWidth}px`;
            upscaled_canvas.style.height = `${displayHeight}px`;
            original_canvas.style.width = `${displayWidth}px`;
            original_canvas.style.height = `${displayHeight}px`;
        }

        async function fullScreenPreview(e: Event) {
            imageCompare.requestFullscreen();
            setTimeout(canvasFullScreen, 20);
            setTimeout(canvasFullScreen, 60);
            setTimeout(canvasFullScreen, 200);
        }
    }
}

/**
 * Handle messages from the video processing worker.
 */
worker.onmessage = function (event: MessageEvent<WorkerResponseMessage>) {
    if (event.data.cmd === 'isSupported') {
        const supported = event.data.data;
        if (!supported) return showUnsupported("WebGPU");

    } else if (event.data.cmd === 'modelLoading') {
        Alpine.store('loading_message', 'Loading AI model...');

    } else if (event.data.cmd === 'modelLoaded') {
        Alpine.store('state', 'preview');

    } else if (event.data.cmd === 'progress') {
        Alpine.store('progress', event.data.data);
        Alpine.store('state', 'processing');

    } else if (event.data.cmd === 'process') {
        // Processing started

    } else if (event.data.cmd === 'error') {
        showError(event.data.data);

    } else if (event.data.cmd === 'eta') {
        Alpine.store('eta', event.data.data);

    } else if (event.data.cmd === 'finished') {
        Alpine.store('state', 'complete');
        if (event.data.data) {
            const formatInfo = getFormatInfo(currentOutputFormat);
            const blob = new Blob([event.data.data], { type: formatInfo?.mimeType || "video/mp4" });
            Alpine.store('download_url', window.URL.createObjectURL(blob));
        }
    }
};

//===================  Process ===========================

/**
 * Start the video upscaling process.
 */
async function initRecording(): Promise<void> {
    Alpine.store('state', 'loading');
    Alpine.store('loading_message', 'Preparing to process...');

    let bitrate = getBitrate();
    const estimated_size = (bitrate / 8) * video.duration + (128 / 8) * video.duration; // Assume 128 kbps audio

    let outputHandle: FileSystemFileHandle | undefined;

    // Max Blob size - 10 MB (for testing, should be much higher in production)
    if (estimated_size > 1900 * 1024 * 1024) {
        try {
            outputHandle = await showFilePicker();
        } catch (e) {
            console.warn("User aborted request");
            return Alpine.store('state', 'preview');
        }
    }

    const resPreset = getResolutionPreset(currentOutputResolution);

    worker.postMessage({
        cmd: "process",
        inputHandle: inputFileHandle,
        outputHandle,
        settings: {
            outputFormat: currentOutputFormat,
            outputResolution: currentOutputResolution,
            targetHeight: resPreset?.maxHeight || undefined,
        }
    } satisfies WorkerRequestMessage);
}

/**
 * Display error message to user.
 */
function showError(message: string): void {
    Alpine.store('state', 'error');
    Alpine.store('error', message);
}

/**
 * Calculate target bitrate based on video resolution.
 */
function getBitrate(): number {
    const modelInfo = getModelInfo(currentModel);
    const scale = modelInfo?.scale || 4;
    // Upscaling = scale^2 pixels, adjust bitrate accordingly
    return 5e6 * (video.videoWidth * video.videoHeight * scale * scale) / (1280 * 720);
}

/**
 * Format bytes into human-readable file size.
 */
function humanFileSize(bytes: number, si: boolean = false, dp: number = 1): string {
    const thresh = si ? 1000 : 1024;

    if (Math.abs(bytes) < thresh) {
        return bytes + ' B';
    }

    const units = si
        ? ['kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
        : ['KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB'];
    let u = -1;
    const r = 10 ** dp;

    do {
        bytes /= thresh;
        ++u;
    } while (Math.round(Math.abs(bytes) * r) / r >= thresh && u < units.length - 1);

    return bytes.toFixed(dp) + ' ' + units[u];
}

/**
 * Show native file picker for saving output video.
 */
async function showFilePicker(): Promise<FileSystemFileHandle> {
    const formatInfo = getFormatInfo(currentOutputFormat);

    const handle = await window.showSaveFilePicker({
        startIn: 'downloads',
        suggestedName: download_name,
        types: [{
            description: 'Video File',
            accept: { [formatInfo?.mimeType || 'video/mp4']: [formatInfo?.extension || '.mp4'] }
        }],
    });

    return handle;
}
