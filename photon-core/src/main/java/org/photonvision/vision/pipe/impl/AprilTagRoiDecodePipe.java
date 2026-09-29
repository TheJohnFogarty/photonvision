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

package org.photonvision.vision.pipe.impl;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipe.CVPipe;
import org.photonvision.vision.pipeline.AprilTagPipelineSettings;
import org.wpilib.vision.apriltag.AprilTagDetection;

/**
 * Decodes model-proposed regions into unique, accepted detections in the input image's coordinates.
 */
public class AprilTagRoiDecodePipe
        extends CVPipe<
                AprilTagRoiDecodePipe.Input, AprilTagRoiDecodePipe.Output, AprilTagPipelineSettings> {
    private static final int DECIMATE_3_THRESHOLD = 160 * 160;
    private static final int DECIMATE_6_THRESHOLD = 320 * 320;

    private final AprilTagDetectionPipe detector;
    private final CropPipe cropPipe = new CropPipe();
    private final PadRectPipe padRectPipe = new PadRectPipe();

    /** Borrows the configured detector shared with full-frame detection; the caller owns it. */
    public AprilTagRoiDecodePipe(AprilTagDetectionPipe detector) {
        this.detector = detector;
    }

    public record Input(CVMat image, List<NeuralNetworkPipeResult> regions) {}

    /** Raw detections suppress full-frame fallback even when quality filtering rejects them all. */
    public record Output(
            List<AprilTagDetection> detections,
            boolean hadRawDetections,
            List<NeuralNetworkPipeResult> cropRegions) {}

    @Override
    protected Output process(Input input) {
        var detectionsById = new LinkedHashMap<Integer, AprilTagDetection>();
        List<NeuralNetworkPipeResult> cropRegions = new ArrayList<>();
        boolean hadRawDetections = false;
        var image = input.image().getMat();
        var config = detector.getParams().detectorParams();
        float defaultDecimate = params.decimate;
        padRectPipe.setParams(params.mlPadding);

        try {
            for (var region : input.regions()) {
                var padded = padRectPipe.run(region.bbox().boundingRect()).output;
                cropPipe.setParams(new CropPipe.CropPipeParams(padded, params));
                try (var croppedImage = cropPipe.run(input.image()).output) {
                    if (padded.width * padded.height >= DECIMATE_6_THRESHOLD) {
                        config.quadDecimate = 6;
                    } else if (padded.width * padded.height >= DECIMATE_3_THRESHOLD) {
                        config.quadDecimate = 3;
                    } else {
                        config.quadDecimate = defaultDecimate;
                    }
                    detector.setConfig(config);

                    var detections = detector.run(croppedImage != null ? croppedImage : input.image()).output;
                    hadRawDetections |= !detections.isEmpty();
                    var cropRect = cropPipe.effectiveCrop(image.cols(), image.rows());
                    var roiRect = cropRect != null ? cropRect : new Rect(0, 0, image.cols(), image.rows());
                    cropRegions.add(
                            new NeuralNetworkPipeResult(
                                    new Rect2d(roiRect.x, roiRect.y, roiRect.width, roiRect.height),
                                    region.classIdx(),
                                    region.confidence()));
                    double offsetX = cropRect != null ? cropRect.x : 0;
                    double offsetY = cropRect != null ? cropRect.y : 0;

                    for (var detection : detections) {
                        if (detection.getDecisionMargin() < params.decisionMargin) continue;
                        if (detection.getHamming() > params.hammingDist) continue;

                        var existing = detectionsById.get(detection.getId());
                        if (existing != null && !(detection.getDecisionMargin() > existing.getDecisionMargin()))
                            continue;

                        var corners = detection.getCorners().clone();
                        for (int i = 0; i < corners.length; i += 2) {
                            corners[i] += offsetX;
                            corners[i + 1] += offsetY;
                        }

                        // Pose uses corners and homography in the same frame: H_frame = T * H_roi.
                        var homography = detection.getHomography();
                        var frameHomography = homography.clone();
                        for (int j = 0; j < 3; j++) {
                            frameHomography[j] += offsetX * homography[6 + j];
                            frameHomography[3 + j] += offsetY * homography[6 + j];
                        }

                        detectionsById.put(
                                detection.getId(),
                                new AprilTagDetection(
                                        detection.getFamily(),
                                        detection.getId(),
                                        detection.getHamming(),
                                        detection.getDecisionMargin(),
                                        frameHomography,
                                        detection.getCenterX() + offsetX,
                                        detection.getCenterY() + offsetY,
                                        corners));
                    }
                }
            }
        } finally {
            // Full-frame fallback and the next frame must use the configured default decimation.
            config.quadDecimate = defaultDecimate;
            detector.setConfig(config);
        }

        return new Output(new ArrayList<>(detectionsById.values()), hadRawDetections, cropRegions);
    }

    @Override
    public void release() {
        cropPipe.release();
    }
}
