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
  resolveOutputResolution,
} from './types/worker-messages';
import type {
  WorkerRequestMessage,
  WorkerResponseMessage,
  ModelInfo,
  ModelCategory,
  ModelType,
  DenoiseLevel,
  OutputFormat,
  OutputResolution,
  ModelConfig,
} from './types/worker-messages';
import { isModelAvailable } from './model-loader';

// Extended model info with availability status for UI
interface ModelInfoWithAvailability extends ModelInfo {
  available: boolean;
}

interface ModelGroupWithAvailability {
    id: ModelCategory;
    name: string;
    models: ModelInfoWithAvailability[];
}

const MODEL_CATEGORY_LABELS: Record<ModelCategory, string> = {
    upscale: 'Upscaling Models',
    cleanup: 'Same-Resolution Cleanup',
};

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
let videoObjectUrl: string | null = null;
let downloadUrl: string | null = null;
// Incremented whenever a new file is selected.  Preview callbacks can outlive
// the video element that created them, so use this token to ignore stale work.
let previewGeneration = 0;

// Current settings
let currentModel: ModelType = 'realesr-animevideov3';
let currentDenoiseLevel: DenoiseLevel = 0;
let currentOutputFormat: OutputFormat = 'mp4';
let currentOutputResolution: OutputResolution = 'auto';
let upscaleOutputResolutionBeforeCleanup: OutputResolution = 'auto';
const MAX_CLEANUP_PREVIEW_HEIGHT = 144;
const AUTO_OUTPUT_HEIGHT_CAP = detectAutoOutputHeightCap();
const AUDIO_BITRATE_BPS = 128_000;
// Blob-backed output remains resident in browser memory until the download is
// released. Keep the convenience path deliberately small and stream larger
// files through the File System Access API instead.
const MAX_IN_MEMORY_OUTPUT_BYTES = 192 * 1024 * 1024;

/**
 * Choose a conservative automatic output cap from coarse browser hardware
 * signals. `deviceMemory` is Chromium-only, so hardwareConcurrency provides a
 * deterministic fallback and unknown environments use the safest cap.
 */
function detectAutoOutputHeightCap(): number {
    if (typeof navigator === 'undefined') return 1080;

    const deviceMemory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
    const hardwareThreads = navigator.hardwareConcurrency || 0;

    if (typeof deviceMemory === 'number' && Number.isFinite(deviceMemory)) {
        if (deviceMemory >= 8 && hardwareThreads >= 8) return 2160;
        if (deviceMemory >= 4 && hardwareThreads >= 4) return 1440;
        return 1080;
    }

    if (hardwareThreads >= 8) return 1440;
    return 1080;
}

function getAutoOutputLabel(): string {
    if (AUTO_OUTPUT_HEIGHT_CAP >= 2160) return '4K';
    if (AUTO_OUTPUT_HEIGHT_CAP >= 1440) return '1440p';
    return '1080p';
}

// Video data
let download_name: string;
let inputFileHandle: FileSystemFileHandle | null = null;
let inputFile: File | null = null;  // Used for remuxed MKV files

function isMatroskaFilename(filename: string): boolean {
    const ext = filename.toLowerCase().split('.').pop();
    return ext === 'mkv' || ext === 'matroska';
}

function getBaseName(filename: string): string {
    return filename.replace(/\.[^/.]+$/, '');
}

/**
 * Probe the native MediaBunny/WebCodecs path before downloading FFmpeg. This
 * only reads container metadata and asks the browser whether the primary
 * video track is decodable; the original file remains available for the
 * worker when the probe succeeds.
 */
async function canDecodeNativeContainer(file: File): Promise<boolean> {
    try {
        const { BlobSource, Input, MATROSKA, WEBM } = await import('mediabunny');
        const input = new Input({
            formats: [MATROSKA, WEBM],
            source: new BlobSource(file),
        });

        try {
            const track = await input.getPrimaryVideoTrack();
            return !!track && await track.canDecode();
        } finally {
            input.dispose();
        }
    } catch (error) {
        console.debug('Native Matroska/WebM probe failed; using FFmpeg fallback:', error);
        return false;
    }
}

/**
 * HTMLVideoElement is still used for the preview UI. A native WebCodecs
 * decoder can support a container that the element cannot render, so verify a
 * decoded frame as well before committing to the native input handle.
 */
async function canPreviewVideo(data: Blob): Promise<boolean> {
    const candidate = document.createElement('video');
    const objectUrl = URL.createObjectURL(data);

    return new Promise((resolve) => {
        let settled = false;
        const finish = (value: boolean) => {
            if (settled) return;
            settled = true;
            candidate.onloadeddata = null;
            candidate.onerror = null;
            candidate.removeAttribute('src');
            candidate.load();
            URL.revokeObjectURL(objectUrl);
            resolve(value);
        };

        const timeout = setTimeout(() => finish(false), 5000);
        candidate.muted = true;
        // We wait for loadeddata (not just metadata) so a container that has
        // readable headers but an unsupported codec is sent through FFmpeg.
        candidate.preload = 'auto';
        candidate.onloadeddata = () => {
            clearTimeout(timeout);
            finish(true);
        };
        candidate.onerror = () => {
            clearTimeout(timeout);
            finish(false);
        };
        candidate.src = objectUrl;
        candidate.load();
    });
}

// Declare global window functions for Alpine to call and File System Access API
declare global {
    interface Window {
        chooseFile: (e?: Event) => Promise<void>;
        initRecording: () => Promise<void>;
        cancelRecording: () => void;
        fullScreenPreview: (e?: Event) => Promise<void>;
        onModelChange: (modelId: string) => Promise<void>;
        onDenoiseChange: (level: number) => Promise<void>;
        onFormatChange: (format: string) => void;
        onResolutionChange: (resolution: string) => void;
        showSaveFilePicker?: (options?: any) => Promise<FileSystemFileHandle>;
        showOpenFilePicker?: (options?: any) => Promise<FileSystemFileHandle[]>;
    }
}

document.addEventListener("DOMContentLoaded", index);

//===================  Initial Load ===========================

/**
 * Main initialization function called on page load.
 */
async function index(): Promise<void> {
    // Expose functions to window immediately for onclick handlers
    window.initRecording = initRecording;
    window.cancelRecording = cancelRecording;
    window.chooseFile = chooseFile;

    Alpine.store('state', 'init');

    // Add availability status to models for UI
    const modelsWithAvailability: ModelInfoWithAvailability[] = AVAILABLE_MODELS.map(model => ({
        ...model,
        available: isModelAvailable(model.id)
    }));
    const modelGroups: ModelGroupWithAvailability[] = (Object.keys(MODEL_CATEGORY_LABELS) as ModelCategory[])
        .map(category => ({
            id: category,
            name: MODEL_CATEGORY_LABELS[category],
            models: modelsWithAvailability.filter(model => model.category === category),
        }))
        .filter(group => group.models.length > 0);

    // Initialize settings stores
    Alpine.store('models', modelsWithAvailability);
    Alpine.store('modelGroups', modelGroups);
    Alpine.store('formats', OUTPUT_FORMATS);
    Alpine.store('resolutions', RESOLUTION_PRESETS.map(preset => (
        preset.id === 'auto'
            ? { ...preset, name: `Auto (hardware cap: ${getAutoOutputLabel()})` }
            : preset
    )));

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

    worker.postMessage({ cmd: 'isSupported' } satisfies WorkerRequestMessage);

    // Set up global functions
    window.chooseFile = chooseFile;
    window.cancelRecording = cancelRecording;
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
 * Return true only for an explicit browser-picker cancellation. Permission,
 * security, and I/O failures should remain visible to the user.
 */
function isAbortError(error: unknown): boolean {
    return typeof error === 'object'
        && error !== null
        && 'name' in error
        && (error as { name?: unknown }).name === 'AbortError';
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

/**
 * Use a normal file input when showOpenFilePicker is unavailable (notably in
 * Safari and Firefox). File objects still remain blob-backed and are streamed
 * by MediaBunny; this does not read the whole input into an ArrayBuffer.
 */
function chooseFileWithInput(): void {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.mp4,.m4v,.webm,.mkv,video/mp4,video/webm,video/x-matroska';
    input.hidden = true;

    const cleanup = () => input.remove();
    input.addEventListener('cancel', cleanup, { once: true });
    input.addEventListener('change', () => {
        const file = input.files?.[0];
        cleanup();
        if (!file) return;

        void loadVideo(file).catch(error => {
            console.error('Failed to open input file:', error);
            showError(`Failed to open video: ${errorMessage(error)}`);
        });
    }, { once: true });

    document.body.append(input);
    input.click();
}

/**
 * Prompt the user for a video through File System Access when available, with
 * a regular file-input fallback for browsers that do not expose that API.
 */
async function chooseFile(e?: Event): Promise<void> {
    if (typeof window.showOpenFilePicker !== 'function') {
        chooseFileWithInput();
        return;
    }

    try {
        const [fileHandle] = await window.showOpenFilePicker({
            types: [{
                description: 'Video Files (MP4, WebM, MKV)',
                accept: {
                    'video/mp4': ['.mp4', '.m4v'],
                    'video/webm': ['.webm'],
                    'video/x-matroska': ['.mkv'],
                }
            }],
            multiple: false
        });

        await loadVideo(fileHandle);
    } catch (error) {
        if (isAbortError(error)) {
            console.log('File selection cancelled');
            return;
        }

        console.error('Failed to open input file:', error);
        showError(`Failed to open video: ${errorMessage(error)}`);
    }
}

//===================  Settings Handlers ===========================

/**
 * Handle model selection change.
 */
async function onModelChange(modelId: string): Promise<void> {
    const modelInfo = getModelInfo(modelId as ModelType);
    if (!modelInfo) return;

    // Reject selection of unavailable models
    if (!isModelAvailable(modelId as ModelType)) {
        Alpine.store('selectedModel', currentModel); // Reset to previous
        return;
    }

    const previousModelInfo = getModelInfo(currentModel);
    const enteringCleanup = previousModelInfo?.scale !== 1 && modelInfo.scale === 1;
    const leavingCleanup = previousModelInfo?.scale === 1 && modelInfo.scale !== 1;

    if (enteringCleanup) {
        upscaleOutputResolutionBeforeCleanup = currentOutputResolution;
    }

    currentModel = modelId as ModelType;
    Alpine.store('selectedModel', currentModel);
    Alpine.store('currentScale', modelInfo.scale);
    Alpine.store('supportsDenoising', modelInfo.supportsDenoising);

    // Reset denoise level if model doesn't support it
    if (!modelInfo.supportsDenoising) {
        currentDenoiseLevel = 0;
        Alpine.store('selectedDenoise', 0);
    }

    if (modelInfo.scale === 1) {
        currentOutputResolution = 'source';
        Alpine.store('selectedResolution', currentOutputResolution);
    } else if (leavingCleanup) {
        currentOutputResolution = upscaleOutputResolutionBeforeCleanup;
        Alpine.store('selectedResolution', currentOutputResolution);
    }

    // Update output dimensions display
    if (video) {
        updateOutputDimensions();
        updateOutputEstimateAndTarget();
    }
    updateDownloadName();

    // If we have a preview, switch the model
    if (previewBitmap && Alpine.store('state') === 'preview') {
        Alpine.store('state', 'loading');
        Alpine.store('loading_message', 'Switching model...');

        const modelConfig = getModelConfig();

        worker.postMessage({
            cmd: 'switchModel',
            data: {
                bitmap: previewBitmap,
                modelConfig,
                targetHeight: getPreviewTargetHeight(),
            }
        } satisfies WorkerRequestMessage);
    }
}

/**
 * Handle denoise level change.
 */
async function onDenoiseChange(level: number): Promise<void> {
    currentDenoiseLevel = level as DenoiseLevel;
    Alpine.store('selectedDenoise', currentDenoiseLevel);

    const modelInfo = getModelInfo(currentModel);
    if (!modelInfo?.supportsDenoising) return;

    if (previewBitmap && Alpine.store('state') === 'preview') {
        Alpine.store('state', 'loading');
        Alpine.store('loading_message', 'Switching denoise level...');

        worker.postMessage({
            cmd: 'switchModel',
            data: {
                bitmap: previewBitmap,
                modelConfig: getModelConfig(),
                targetHeight: getPreviewTargetHeight(),
            }
        } satisfies WorkerRequestMessage);
    }
}

/**
 * Handle output format change.
 */
function onFormatChange(format: string): void {
    currentOutputFormat = format as OutputFormat;
    Alpine.store('selectedFormat', currentOutputFormat);
    updateDownloadName();
    updateOutputEstimateAndTarget();
}

/**
 * Handle output resolution change.
 */
function onResolutionChange(resolution: string): void {
    currentOutputResolution = resolution as OutputResolution;
    if ((getModelInfo(currentModel)?.scale || 1) !== 1) {
        upscaleOutputResolutionBeforeCleanup = currentOutputResolution;
    }
    Alpine.store('selectedResolution', currentOutputResolution);
    updateOutputDimensions();
    updateDownloadName();
    updateOutputEstimateAndTarget();

    if (previewBitmap && Alpine.store('state') === 'preview') {
        Alpine.store('state', 'loading');
        Alpine.store('loading_message', 'Updating preview resolution...');
        worker.postMessage({
            cmd: 'renderPreview',
            data: {
                bitmap: previewBitmap,
                targetHeight: getPreviewTargetHeight(),
            },
        } satisfies WorkerRequestMessage);
    }
}

/**
 * Get current model configuration.
 */
function getModelConfig(): ModelConfig {
    const modelInfo = getModelInfo(currentModel);
    const scale = modelInfo?.scale || 4;
    const isScunet = currentModel.startsWith('scunet-');
    const isSwinirJpeg = currentModel.startsWith('swinir-jpeg');

    return {
        modelId: currentModel,
        scale,
        tileSize: isSwinirJpeg ? 126 : isScunet ? 256 : scale === 2 ? 1024 : 512,
        tilePadding: isSwinirJpeg ? 14 : 32,
        inputWidth: isSwinirJpeg ? 126 : undefined,
        inputHeight: isSwinirJpeg ? 126 : undefined,
        inputMultiple: isScunet ? 64 : undefined,
        denoiseLevel: modelInfo?.supportsDenoising ? currentDenoiseLevel : undefined,
    };
}

/**
 * Update output dimensions display based on current settings.
 */
function updateOutputDimensions(): void {
    if (!video?.videoWidth || !video.videoHeight) return;

    const modelInfo = getModelInfo(currentModel);
    const scale = modelInfo?.scale || 4;
    const { width: outputWidth, height: outputHeight } = resolveOutputResolution(
        { width: video.videoWidth, height: video.videoHeight },
        scale,
        getTargetHeight(),
        AUTO_OUTPUT_HEIGHT_CAP,
    );

    Alpine.store('outputWidth', outputWidth);
    Alpine.store('outputHeight', outputHeight);
    Alpine.store('currentScale', Number((outputHeight / video.videoHeight).toFixed(2)));
}

/**
 * Update download filename based on current settings.
 */
function updateDownloadName(): void {
    const formatInfo = getFormatInfo(currentOutputFormat);
    const modelInfo = getModelInfo(currentModel);
    const baseName = (Alpine.store('filename') as string)?.split(".")[0] || 'video';
    const resolutionSuffix = currentOutputResolution === 'source' ? 'source' : `${modelInfo?.scale || 4}x`;

    download_name = `${baseName}-upscaled-${resolutionSuffix}${formatInfo?.extension || '.mp4'}`;
    Alpine.store('download_name', download_name);
}

function getTargetHeight(): number | undefined {
    if (!video?.videoHeight) return undefined;

    if (currentOutputResolution === 'source') {
        return video.videoHeight;
    }

    if (currentOutputResolution === 'auto') {
        const modelInfo = getModelInfo(currentModel);
        const nativeHeight = video.videoHeight * (modelInfo?.scale || 4);
        // Auto never enlarges beyond the model's native scale and never
        // downscales a source that is already taller than the hardware cap.
        return Math.min(
            nativeHeight,
            Math.max(video.videoHeight, AUTO_OUTPUT_HEIGHT_CAP)
        );
    }

    return getResolutionPreset(currentOutputResolution)?.maxHeight || undefined;
}

function getPreviewTargetHeight(): number | undefined {
    const targetHeight = getTargetHeight();
    const modelInfo = getModelInfo(currentModel);

    if (modelInfo?.category === 'cleanup') {
        return Math.min(targetHeight || MAX_CLEANUP_PREVIEW_HEIGHT, MAX_CLEANUP_PREVIEW_HEIGHT);
    }

    return targetHeight;
}

function clearMediaObjectUrls(): void {
    previewGeneration += 1;

    previewBitmap?.close();
    previewBitmap = null;

    if (video) {
        // Detach callbacks before clearing the source.  This avoids a queued
        // `loadeddata` event from rendering a preview for the previous file.
        video.onloadeddata = null;
        video.onerror = null;
        video.pause();
        video.removeAttribute('src');
        video.load();
    }

    if (videoObjectUrl) {
        URL.revokeObjectURL(videoObjectUrl);
        videoObjectUrl = null;
    }

    if (downloadUrl) {
        URL.revokeObjectURL(downloadUrl);
        downloadUrl = null;
        Alpine.store('download_url', '');
    }

    // A canvas can only be transferred to an OffscreenCanvas once.  Replace
    // the DOM canvases between files so subsequent previews do not throw
    // InvalidStateError, while releasing the old browser-side backing stores.
    upscaled_canvas = replacePreviewCanvas(upscaled_canvas);
    original_canvas = replacePreviewCanvas(original_canvas);

    inputFileHandle = null;
    inputFile = null;
}

function replacePreviewCanvas(canvas: HTMLCanvasElement): HTMLCanvasElement {
    if (!canvas?.parentNode) return canvas;

    const replacement = canvas.cloneNode(false) as HTMLCanvasElement;
    // Do not carry over a previous fullscreen size or a large content
    // allocation. The worker sets the OffscreenCanvas dimensions during init.
    replacement.removeAttribute('width');
    replacement.removeAttribute('height');
    replacement.style.width = '';
    replacement.style.height = '';
    canvas.replaceWith(replacement);
    return replacement;
}

//===================  Preview ===========================

/**
 * Load a video from either File System Access or the regular file-input
 * fallback.
 * Uses native Matroska/WebM support when available and lazily falls back to
 * FFmpeg remuxing only when the browser cannot decode the container.
 */
async function loadVideo(source: FileSystemFileHandle | File): Promise<void> {
    Alpine.store('state', 'loading');
    Alpine.store('loading_message', 'Loading video...');
    clearMediaObjectUrls();

    // Get the file to check format and create preview
    const fileHandle = source instanceof File ? null : source;
    const file = source instanceof File ? source : await source.getFile();
    const originalFilename = file.name;

    // Set up initial filename (use base name, will add extension based on output format)
    Alpine.store('filename', originalFilename);

    if (isMatroskaFilename(originalFilename)) {
        const nativeContainer = await canDecodeNativeContainer(file);
        const nativePreview = nativeContainer && await canPreviewVideo(file);

        if (nativePreview) {
            Alpine.store('loading_message', 'Loading native Matroska input...');
            // Keep the original handle so the worker can stream the source
            // without creating an intermediate MP4 or duplicating the file.
            inputFileHandle = fileHandle;
            inputFile = fileHandle ? null : file;
        } else {
            Alpine.store('loading_message', 'Converting MKV to MP4...');

            try {
                // Keep FFmpeg and its large loader out of the initial bundle.
                const { remuxToMp4 } = await import('./remux');
                const arrayBuffer = await remuxToMp4(file, (message) => {
                    Alpine.store('loading_message', message);
                });

                // Store remuxed file directly (can't use virtual FileSystemFileHandle with postMessage)
                const mp4Blob = new Blob([arrayBuffer], { type: 'video/mp4' });
                inputFile = new File([mp4Blob], getBaseName(originalFilename) + '.mp4', { type: 'video/mp4' });
                inputFileHandle = null;  // No handle for remuxed files
            } catch (e) {
                console.error('Remux failed:', e);
                showError(`Failed to convert MKV: ${e}`);
                return;
            }
        }
    } else {
        // MP4 and WebM inputs can stream from a file handle or remain backed by
        // the File selected through the standard input fallback.
        inputFileHandle = fileHandle;
        inputFile = fileHandle ? null : file;
    }

    updateDownloadName();
    // Keep normal files as Blob-backed object URLs.  Reading the complete
    // file into an ArrayBuffer here duplicated potentially gigabytes of data
    // before MediaBunny streamed it from the file handle during processing.
    await setupPreview(inputFile || file);
}

/**
 * Set up the preview UI with before/after comparison.
 */
async function setupPreview(data: Blob): Promise<void> {
    const generation = previewGeneration;
    console.log('setupPreview called, creating video element');
    video = document.createElement('video');
    console.log('video element created:', video);

    videoObjectUrl = URL.createObjectURL(data);
    video.src = videoObjectUrl;

    const imageCompare = document.getElementById('image-compare-outer') as HTMLElement;

    video.onloadeddata = async function () {
        if (generation !== previewGeneration) return;

        console.log('video.onloadeddata fired, videoWidth:', video.videoWidth);
        Alpine.store('width', video.videoWidth);
        Alpine.store('height', video.videoHeight);

        // Update output dimensions
        updateOutputDimensions();

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
        if (generation !== previewGeneration) return;

        const fullScreenButton = document.getElementById('full-screen');

        window.initRecording = initRecording;
        window.fullScreenPreview = fullScreenPreview;

        // Store bitmap for model switching
        const bitmap = await createImageBitmap(video);

        if (generation !== previewGeneration) {
            // A newer file may already have installed its own preview bitmap;
            // only close the bitmap created by this stale callback.
            bitmap.close();
            return;
        }

        previewBitmap?.close();
        previewBitmap = bitmap;

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
                modelConfig,
                targetHeight: getPreviewTargetHeight(),
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

        updateOutputEstimateAndTarget();

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
        if (!supported) return showUnsupported("ONNX Runtime WebAssembly");

    } else if (event.data.cmd === 'modelLoading') {
        const progress = event.data.data;
        if (progress < 100) {
            Alpine.store('loading_message', `Downloading AI model... ${progress}%`);
        } else {
            Alpine.store('loading_message', 'Initializing AI model...');
        }

    } else if (event.data.cmd === 'status') {
        Alpine.store('loading_message', event.data.data);

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

    } else if (event.data.cmd === 'timing') {
        // Keep the latest averaged stage timings available for diagnostics
        // without updating the visible progress UI on every video frame.
        Alpine.store('timing', event.data.data);

    } else if (event.data.cmd === 'pipeline') {
        Alpine.store('pipeline', event.data.data);

    } else if (event.data.cmd === 'finished') {
        Alpine.store('state', 'complete');
        if (event.data.data) {
            const formatInfo = getFormatInfo(currentOutputFormat);
            const blob = new Blob([event.data.data], { type: formatInfo?.mimeType || "video/mp4" });
            if (downloadUrl) {
                URL.revokeObjectURL(downloadUrl);
            }
            downloadUrl = URL.createObjectURL(blob);
            Alpine.store('download_url', downloadUrl);
        }

    } else if (event.data.cmd === 'cancelled') {
        Alpine.store('progress', 0);
        Alpine.store('eta', '');
        Alpine.store('state', 'preview');
    }
};

//===================  Process ===========================

/**
 * Start the video upscaling process.
 */
async function initRecording(): Promise<void> {
    console.log('initRecording called');

    if (!video || !video.videoWidth) {
        console.error('Video not loaded yet');
        return;
    }

    Alpine.store('state', 'loading');
    Alpine.store('loading_message', 'Preparing to process...');

    updateOutputDimensions();
    const estimatedSize = estimateOutputSize();
    updateOutputEstimateAndTarget(estimatedSize);

    let outputHandle: FileSystemFileHandle | undefined;

    if (shouldStreamOutput(estimatedSize)) {
        if (typeof window.showSaveFilePicker !== 'function') {
            showError(
                'This output must be streamed to a file, but this browser does not support the save-file picker. '
                + 'Choose a smaller output or use Chrome/Edge for large or unknown-duration jobs.'
            );
            return;
        }

        try {
            outputHandle = await showFilePicker();
        } catch (error) {
            if (isAbortError(error)) {
                console.log('Output selection cancelled');
                Alpine.store('state', 'preview');
                return;
            }

            console.error('Failed to choose output location:', error);
            showError(`Failed to choose output location: ${errorMessage(error)}`);
            return;
        }
    }

    // Pass either inputHandle (for regular files) or inputFile (for remuxed MKV)
    const message: WorkerRequestMessage = {
        cmd: "process",
        inputHandle: inputFileHandle || undefined,
        inputFile: inputFile || undefined,
        outputHandle,
        settings: {
            outputFormat: currentOutputFormat,
            outputResolution: currentOutputResolution,
            targetHeight: getTargetHeight(),
        }
    };

    worker.postMessage(message);
}

function cancelRecording(): void {
    Alpine.store('state', 'loading');
    Alpine.store('loading_message', 'Stopping...');
    worker.postMessage({ cmd: 'cancel' } satisfies WorkerRequestMessage);
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
    const outputWidth = (Alpine.store('outputWidth') as number) || video.videoWidth * scale;
    const outputHeight = (Alpine.store('outputHeight') as number) || video.videoHeight * scale;

    return 5e6 * (outputWidth * outputHeight) / (1280 * 720);
}

function estimateOutputSize(): number | null {
    if (!video || !Number.isFinite(video.duration) || video.duration <= 0) {
        return null;
    }

    return ((getBitrate() + AUDIO_BITRATE_BPS) / 8) * video.duration;
}

function shouldStreamOutput(estimatedSize: number | null): boolean {
    // Unknown-duration inputs must use the bounded-memory streaming path.
    return estimatedSize === null || estimatedSize > MAX_IN_MEMORY_OUTPUT_BYTES;
}

function updateOutputEstimateAndTarget(estimatedSize: number | null = estimateOutputSize()): void {
    const streamOutput = shouldStreamOutput(estimatedSize);
    Alpine.store('target', streamOutput ? 'writer' : 'blob');
    Alpine.store('size', estimatedSize === null ? 'Unknown (streaming)' : humanFileSize(estimatedSize));
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
    const picker = window.showSaveFilePicker;

    if (typeof picker !== 'function') {
        throw new Error('The browser does not support streamed file output');
    }

    const handle = await picker.call(window, {
        startIn: 'downloads',
        suggestedName: download_name,
        types: [{
            description: 'Video File',
            accept: { [formatInfo?.mimeType || 'video/mp4']: [formatInfo?.extension || '.mp4'] }
        }],
    });

    return handle;
}
