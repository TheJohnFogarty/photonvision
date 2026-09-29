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

package org.photonvision.vision.pipeline;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.stream.IntStream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.vision.calibration.CameraCalibrationCoefficients;
import org.photonvision.vision.calibration.CameraLensModel;
import org.photonvision.vision.calibration.JsonMatOfDouble;
import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipe.impl.AprilTagDetectionPipe;
import org.photonvision.vision.pipe.impl.NeuralNetworkPipeResult;
import org.photonvision.vision.pipe.impl.ObjectDetectionPipe;
import org.photonvision.vision.target.TrackedTarget;
import org.wpilib.vision.apriltag.AprilTagDetection;

class MLTagDeduplicationTest {
    @BeforeAll
    static void loadLibraries() {
        LoadJNI.loadLibraries();
        ConfigManager.getInstance().load();
    }

    private record Observation(int id, float margin, int hamming, double centerX) {}

    private static AprilTagPipelineSettings settings() {
        var settings = new AprilTagPipelineSettings();
        settings.mltagEnabled = true;
        settings.mltagFallbackEnabled = false;
        settings.mlPadding = 0;
        settings.solvePNPEnabled = false;
        settings.hammingDist = 0;
        settings.decisionMargin = 35;
        return settings;
    }

    private static AprilTagPipeline pipeline(
            AprilTagPipelineSettings settings, Observation... observations) {
        // Each overlapping proposal has a different origin, aligned to the detector's tile grid.
        var proposals =
                IntStream.range(0, observations.length)
                        .mapToObj(i -> new NeuralNetworkPipeResult(new Rect2d(16 + 4 * i, 16, 100, 80), 0, 0.9))
                        .toList();
        var model =
                new ObjectDetectionPipe() {
                    @Override
                    protected List<NeuralNetworkPipeResult> process(CVMat in) {
                        return proposals;
                    }
                };
        var detector =
                new AprilTagDetectionPipe() {
                    private int index;

                    @Override
                    protected List<AprilTagDetection> process(CVMat in) {
                        var observation = observations[index];
                        double x = observation.centerX() - (16 + 4 * index++);
                        double y = 60 - 16;
                        return List.of(
                                new AprilTagDetection(
                                        "tag36h11",
                                        observation.id(),
                                        observation.hamming(),
                                        observation.margin(),
                                        new double[] {10, 0, x, 0, 10, y, 0, 0, 1},
                                        x,
                                        y,
                                        new double[] {x - 10, y + 10, x + 10, y + 10, x + 10, y - 10, x - 10, y - 10}));
                    }
                };
        return new AprilTagPipeline(settings, model, detector);
    }

    private static Frame frame(CameraCalibrationCoefficients calibration) {
        return new Frame(
                1,
                new CVMat(new Mat(120, 160, CvType.CV_8UC1, new Scalar(128))),
                new CVMat(new Mat(120, 160, CvType.CV_8UC1, new Scalar(128))),
                FrameThresholdType.GREYSCALE,
                new FrameStaticProperties(160, 120, 70, calibration));
    }

    @Test
    void overlappingProposalsKeepStrongestObservationEvenWithAWeakerLaterResult() {
        try (var pipeline =
                        pipeline(
                                settings(),
                                new Observation(1, 50, 0, 60),
                                new Observation(1, 90, 0, 61),
                                new Observation(1, 70, 0, 62));
                var frame = frame(null);
                var result = pipeline.run(frame, QuirkyCamera.DefaultCamera)) {
            assertEquals(1, result.targets.size());
            assertEquals(51, result.targets.getFirst().getTargetCorners().getFirst().x, 1e-9);
            assertEquals(
                    3, result.mlROIs.size(), "Every model proposal remains available for the overlay");
        }
    }

    @Test
    void higherMarginRejectedObservationsDoNotSuppressAValidTag() {
        try (var pipeline =
                        pipeline(
                                settings(),
                                new Observation(1, 150, 1, 60),
                                new Observation(1, 70, 0, 61),
                                new Observation(1, 200, 1, 62));
                var frame = frame(null);
                var result = pipeline.run(frame, QuirkyCamera.DefaultCamera)) {
            assertEquals(1, result.targets.size());
            assertEquals(51, result.targets.getFirst().getTargetCorners().getFirst().x, 1e-9);
        }
    }

    @Test
    void distinctIdsKeepTheirFirstSeenOrderWhenAnEarlierIdIsReplaced() {
        try (var pipeline =
                        pipeline(
                                settings(),
                                new Observation(1, 50, 0, 60),
                                new Observation(2, 80, 0, 85),
                                new Observation(1, 90, 0, 61));
                var frame = frame(null);
                var result = pipeline.run(frame, QuirkyCamera.DefaultCamera)) {
            assertEquals(
                    List.of(1, 2), result.targets.stream().map(TrackedTarget::getFiducialId).toList());
            assertEquals(51, result.targets.getFirst().getTargetCorners().getFirst().x, 1e-9);
            assertEquals(75, result.targets.getLast().getTargetCorners().getFirst().x, 1e-9);
        }
    }

    @Test
    void fullFrameFallbackPreservesSeparateTagsWithTheSameId() {
        var settings = settings();
        settings.mltagFallbackEnabled = true;
        var model =
                new ObjectDetectionPipe() {
                    @Override
                    protected List<NeuralNetworkPipeResult> process(CVMat in) {
                        return List.of();
                    }
                };
        var detector =
                new AprilTagDetectionPipe() {
                    @Override
                    protected List<AprilTagDetection> process(CVMat in) {
                        assertEquals(160, in.getMat().cols());
                        assertEquals(120, in.getMat().rows());
                        return IntStream.of(50, 90)
                                .mapToObj(
                                        x ->
                                                new AprilTagDetection(
                                                        "tag36h11",
                                                        1,
                                                        0,
                                                        x,
                                                        new double[] {10, 0, x, 0, 10, 60, 0, 0, 1},
                                                        x,
                                                        60,
                                                        new double[] {x - 10, 70, x + 10, 70, x + 10, 50, x - 10, 50}))
                                .toList();
                    }
                };
        try (var pipeline = new AprilTagPipeline(settings, model, detector);
                var frame = frame(null);
                var result = pipeline.run(frame, QuirkyCamera.DefaultCamera)) {
            assertEquals(
                    List.of(1, 1), result.targets.stream().map(TrackedTarget::getFiducialId).toList());
            assertEquals(40, result.targets.getFirst().getTargetCorners().getFirst().x, 1e-9);
            assertEquals(80, result.targets.getLast().getTargetCorners().getFirst().x, 1e-9);
            assertTrue(result.mlROIs.isEmpty());
        }
    }

    @Test
    void duplicateObservationsCannotProduceAMultiTagPose() {
        assertTrue(ConfigManager.getInstance().getConfig().getFieldLayout().getTagPose(1).isPresent());
        var settings = settings();
        settings.solvePNPEnabled = true;
        settings.doMultiTarget = true;
        try (var calibration =
                        new CameraCalibrationCoefficients(
                                new Size(160, 120),
                                new JsonMatOfDouble(3, 3, new double[] {200, 0, 80, 0, 200, 60, 0, 0, 1}),
                                new JsonMatOfDouble(1, 5, new double[5]),
                                new double[0],
                                List.of(),
                                new Size(),
                                0,
                                CameraLensModel.LENSMODEL_OPENCV);
                var pipeline =
                        pipeline(settings, new Observation(1, 60, 0, 60), new Observation(1, 90, 0, 60));
                var frame = frame(calibration);
                var result = pipeline.run(frame, QuirkyCamera.DefaultCamera)) {
            assertTrue(result.multiTagResult.isEmpty(), "Two observations of one tag are not two tags");
            assertEquals(1, result.targets.size());
        }
    }
}
