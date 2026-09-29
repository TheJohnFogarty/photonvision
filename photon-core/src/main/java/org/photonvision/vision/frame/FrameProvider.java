/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.vision.frame;

import java.util.function.Supplier;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.opencv.ImageRotationMode;
import org.photonvision.vision.opencv.Releasable;
import org.photonvision.vision.pipe.impl.CropPipe;
import org.photonvision.vision.pipe.impl.HSVPipe;
import org.photonvision.vision.pipeline.AdvancedPipelineSettings;

public abstract class FrameProvider implements Supplier<Frame>, Releasable {
    protected int sequenceID = 0;
    private final CropPipe cropPipe = new CropPipe();

    /** How much the cropped-away area is dimmed in the input stream's context image. */
    private static final double CONTEXT_DIM_FACTOR = 0.35;

    // Escape hatch to allow us to synchronously (from the main vision thread) run
    // extra
    // setup/callbacks once cscore connects to our underlying device for the first
    // time
    public boolean cameraPropertiesCached = false;

    protected void onCameraConnected() {
        cameraPropertiesCached = true;
    }

    /** Internal provider for if the camera is currently connected. */
    protected abstract boolean checkCameraConnected();

    /** Checks if the camera is currently connected. Also handles connection events. */
    public boolean isConnected() {
        boolean connected = this.checkCameraConnected();

        if (!cameraPropertiesCached && connected) {
            onCameraConnected();
        }

        return connected;
    }

    /**
     * Returns if the camera has connected at some point. This is not if it is currently connected.
     */
    public boolean hasConnected() {
        return cameraPropertiesCached;
    }

    public abstract String getName();

    /** Ask the camera to produce a certain kind of processed image (e.g. HSV or greyscale) */
    public abstract void requestFrameThresholdType(FrameThresholdType type);

    /** Ask the camera for grayscale where the camera format allows it. */
    public void requestGrayscaleInput(boolean grayscaleInput) {}

    /** Ask the camera to rotate frames it outputs */
    public abstract void requestFrameRotation(ImageRotationMode rotationMode);

    /** Ask the camera to provide either the input, output, or both frames. */
    public abstract void requestFrameCopies(boolean copyInput, boolean copyOutput);

    /** Ask the camera to rotate frames it outputs */
    public abstract void requestHsvSettings(HSVPipe.HSVParams params);

    /** Ask the camera to block for new frames (true) or use latest available (false) */
    public abstract void requestBlockForFrames(boolean blockForFrames);

    public final void setCropParams(AdvancedPipelineSettings settings) {
        // A pixel bound is never negative. Dropping the sign rather than trusting it keeps a garbage
        // bound (a value that overflowed on its way in, say) from being read as a sliver of a crop
        // one pixel from the origin.
        int xLow =
                Math.max(0, Math.min(settings.staticCropX.getFirst(), settings.staticCropX.getSecond()));
        int xHigh =
                Math.max(0, Math.max(settings.staticCropX.getFirst(), settings.staticCropX.getSecond()));
        int yLow =
                Math.max(0, Math.min(settings.staticCropY.getFirst(), settings.staticCropY.getSecond()));
        int yHigh =
                Math.max(0, Math.max(settings.staticCropY.getFirst(), settings.staticCropY.getSecond()));

        int width = xHigh - xLow;
        int height = yHigh - yLow;

        Rect cropRegion;
        if (width <= 0 || height <= 0) {
            cropRegion = null;
        } else {
            cropRegion = new Rect(xLow, yLow, width, height);
        }

        if (!settings.staticCropEnabled) {
            cropRegion = null;
        }

        cropPipe.setParams(new CropPipe.CropPipeParams(cropRegion, settings));
    }

    /**
     * Statically crop the frame to the configured region, in place. A no-op crop returns the frame
     * unchanged.
     *
     * @param frame The frame to crop.
     * @param keepContext Whether to also keep a dimmed full-frame context image for the input stream.
     * @return The cropped frame, carrying properties derived for the cropped size.
     */
    public final Frame cropFrame(Frame frame, boolean keepContext) {
        var reference = !frame.colorImage.getMat().empty() ? frame.colorImage : frame.processedImage;
        Rect effectiveCrop =
                cropPipe.effectiveCrop(reference.getMat().cols(), reference.getMat().rows());
        if (effectiveCrop == null) {
            // Cropping is a no-op, so the cached cropped properties can never be reused; don't hold
            // their native calibration memory alive until a crop happens to come along again.
            cropPipe.releaseCachedProperties();
            return frame;
        }

        CVMat contextImage = null;
        try {
            if (keepContext && !frame.colorImage.getMat().empty()) {
                contextImage = new CVMat();
                Mat sourceCrop = null;
                Mat contextCrop = null;
                try {
                    frame.colorImage.getMat().convertTo(contextImage.getMat(), -1, CONTEXT_DIM_FACTOR, 0);
                    sourceCrop = frame.colorImage.getMat().submat(effectiveCrop);
                    contextCrop = contextImage.getMat().submat(effectiveCrop);
                    sourceCrop.copyTo(contextCrop);
                } finally {
                    if (sourceCrop != null) sourceCrop.release();
                    if (contextCrop != null) contextCrop.release();
                }
            }

            boolean cropped = cropInPlace(frame.colorImage);
            cropped |= cropInPlace(frame.processedImage);
            if (!cropped) {
                return frame;
            }

            var croppedFrame =
                    new Frame(
                            frame.sequenceID,
                            frame.colorImage,
                            frame.processedImage,
                            frame.type,
                            frame.timestampNanos,
                            frame.frameStaticProperties != null
                                    ? cropPipe.croppedProperties(frame.frameStaticProperties, effectiveCrop)
                                    : null);
            croppedFrame.contextColorImage = contextImage;
            contextImage = null; // Ownership passes to the returned frame only after success.
            return croppedFrame;
        } finally {
            if (contextImage != null) contextImage.release();
        }
    }

    private boolean cropInPlace(CVMat image) {
        var result = cropPipe.run(image);
        if (result.output == null) {
            return false;
        }

        Mat cropped = null;
        try {
            cropped = result.output.getMat().clone();
            cropped.copyTo(image.getMat());
            return true;
        } finally {
            result.output.release();
            if (cropped != null) cropped.release();
        }
    }

    @Override
    public void release() {
        cropPipe.release();
    }
}
