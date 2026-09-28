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

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.photonvision.common.LoadJNI;
import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipe.CVPipe;
import org.photonvision.vision.pipe.impl.AprilTagDetectionPipe;
import org.photonvision.vision.pipe.impl.NeuralNetworkPipeResult;
import org.photonvision.vision.pipe.impl.ObjectDetectionPipe;
import org.photonvision.vision.target.TrackedTarget;
import org.wpilib.vision.apriltag.AprilTagDetection;

class MLTagResourceTest {
    @BeforeAll
    static void loadLibraries() {
        LoadJNI.loadLibraries();
    }

    private static Frame frame() {
        return new Frame(
                1,
                new CVMat(new Mat(120, 160, CvType.CV_8UC1, new Scalar(128))),
                new CVMat(new Mat(120, 160, CvType.CV_8UC1, new Scalar(128))),
                FrameThresholdType.GREYSCALE,
                new FrameStaticProperties(160, 120, 70, null));
    }

    private static AprilTagPipelineSettings settings() {
        var settings = new AprilTagPipelineSettings();
        settings.mltagEnabled = true;
        settings.mltagFallbackEnabled = false;
        settings.mlPadding = 0;
        settings.solvePNPEnabled = false;
        return settings;
    }

    private static ObjectDetectionPipe model(Rect2d bounds) {
        return new ObjectDetectionPipe() {
            @Override
            protected List<NeuralNetworkPipeResult> process(CVMat in) {
                return List.of(new NeuralNetworkPipeResult(bounds, 0, 0.9));
            }
        };
    }

    private static class RecordingDetector extends AprilTagDetectionPipe {
        final List<CVMat> inputs = new ArrayList<>();
        boolean fail;
        List<AprilTagDetection> detections = List.of();

        @Override
        protected List<AprilTagDetection> process(CVMat in) {
            inputs.add(in);
            assertNotNull(in, "A full-frame proposal must use the original image");
            assertFalse(in.getMat().empty());
            if (fail) throw new IllegalStateException("detection failed");
            return detections;
        }
    }

    @Test
    void releasesEveryMlCropAfterDetection() {
        var detector = new RecordingDetector();
        detector.detections =
                List.of(
                        new AprilTagDetection(
                                "tag36h11",
                                1,
                                0,
                                100,
                                new double[] {1, 0, 0, 0, 1, 0, 0, 0, 1},
                                15,
                                15,
                                new double[] {5, 5, 25, 5, 25, 25, 5, 25}));
        try (var pipeline =
                new AprilTagPipeline(settings(), model(new Rect2d(20, 20, 40, 40)), detector)) {
            for (int i = 0; i < 20; i++) {
                try (var frame = frame();
                        var result = pipeline.run(frame, QuirkyCamera.DefaultCamera)) {
                    assertEquals(1, result.mlROIs.size());
                    assertEquals(1, result.targets.size());
                    assertTrue(detector.inputs.getLast().isReleased(), "ML crop must be released each frame");
                    assertFalse(frame.processedImage.isReleased(), "The caller's input must remain alive");
                }
            }
        } finally {
            detector.inputs.stream().filter(it -> it != null && !it.isReleased()).forEach(CVMat::release);
        }
    }

    @Test
    void releasesMlCropWhenDetectionThrows() {
        var detector = new RecordingDetector();
        detector.fail = true;
        try (var pipeline =
                        new AprilTagPipeline(settings(), model(new Rect2d(20, 20, 40, 40)), detector);
                var frame = frame()) {
            assertThrows(
                    IllegalStateException.class, () -> pipeline.run(frame, QuirkyCamera.DefaultCamera));
            assertTrue(detector.inputs.getFirst().isReleased());
            assertFalse(frame.processedImage.isReleased());
            assertEquals(
                    pipeline.getSettings().decimate, detector.getParams().detectorParams().quadDecimate);
        } finally {
            detector.inputs.stream().filter(it -> it != null && !it.isReleased()).forEach(CVMat::release);
        }
    }

    @Test
    void fullFrameMlProposalBorrowsInputWithoutReleasingIt() {
        var detector = new RecordingDetector();
        try (var pipeline =
                        new AprilTagPipeline(settings(), model(new Rect2d(0, 0, 160, 120)), detector);
                var frame = frame();
                var result = pipeline.run(frame, QuirkyCamera.DefaultCamera)) {
            assertSame(frame.processedImage, detector.inputs.getFirst());
            assertFalse(frame.processedImage.isReleased());
            assertEquals(1, result.mlROIs.size());
        }
    }

    @Test
    void failingFullFrameDetectionDoesNotReleaseBorrowedInput() {
        var detector = new RecordingDetector();
        detector.fail = true;
        try (var pipeline =
                        new AprilTagPipeline(settings(), model(new Rect2d(0, 0, 160, 120)), detector);
                var frame = frame()) {
            assertThrows(
                    IllegalStateException.class, () -> pipeline.run(frame, QuirkyCamera.DefaultCamera));
            assertSame(frame.processedImage, detector.inputs.getFirst());
            assertFalse(frame.processedImage.isReleased());
            assertEquals(
                    pipeline.getSettings().decimate, detector.getParams().detectorParams().quadDecimate);
        }
    }

    @Test
    void poseFailureReleasesAlreadyCollectedMlRegions() throws ReflectiveOperationException {
        var detector = new RecordingDetector();
        detector.detections =
                List.of(
                        new AprilTagDetection(
                                "tag36h11",
                                1,
                                0,
                                100,
                                new double[] {1, 0, 0, 0, 1, 0, 0, 0, 1},
                                15,
                                15,
                                new double[] {5, 5, 25, 5, 25, 25, 5, 25}));
        var settings = settings();
        settings.solvePNPEnabled = true;
        settings.doMultiTarget = false;
        try (var pipeline =
                        new AprilTagPipeline(settings, model(new Rect2d(20, 20, 40, 40)), detector);
                var frame = frame()) {
            // The input is deliberately uncalibrated, so pose estimation fails after ROI collection.
            assertThrows(
                    NullPointerException.class, () -> pipeline.run(frame, QuirkyCamera.DefaultCamera));

            // Observe the real collector's retained output without adding a production API for tests.
            var collectorField = AprilTagPipeline.class.getDeclaredField("collect2dMLROIsPipe");
            collectorField.setAccessible(true);
            var resultField = CVPipe.class.getDeclaredField("result");
            resultField.setAccessible(true);
            var collectionResult = (CVPipe.CVPipeResult<?>) resultField.get(collectorField.get(pipeline));
            var regions = (List<?>) collectionResult.output;
            assertEquals(1, regions.size());
            for (var region : regions) {
                var target = (TrackedTarget) region;
                try {
                    assertTrue(
                            target.getShape().contour.mat.empty(), "Failed frames must release ML regions");
                } finally {
                    target.release();
                }
            }
            assertFalse(frame.processedImage.isReleased());
        }
    }
}
