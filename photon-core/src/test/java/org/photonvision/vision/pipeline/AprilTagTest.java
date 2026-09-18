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

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.opencv.core.Rect;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.util.TestUtils;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.provider.FileFrameProvider;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipe.impl.PadRectPipe;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.photonvision.vision.target.TargetModel;
import org.wpilib.math.geometry.Transform3d;
import org.wpilib.math.geometry.Translation3d;

public class AprilTagTest {
    @BeforeEach
    public void setup() {
        LoadJNI.loadLibraries();
        ConfigManager.getInstance().load();
    }

    @Test
    public void testApriltagFacingCamera() {
        Transform3d pose;
        try (var pipeline = new AprilTagPipeline()) {
            pipeline.getSettings().inputShouldShow = true;
            pipeline.getSettings().outputShouldDraw = true;
            pipeline.getSettings().solvePNPEnabled = true;
            pipeline.getSettings().cornerDetectionAccuracyPercentage = 4;
            pipeline.getSettings().cornerDetectionUseConvexHulls = true;
            pipeline.getSettings().targetModel = TargetModel.kAprilTag6p5in_36h11;
            pipeline.getSettings().tagFamily = AprilTagFamily.kTag36h11;

            try (var frameProvider =
                    new FileFrameProvider(
                            TestUtils.getApriltagImagePath(TestUtils.ApriltagTestImages.kTag1_640_480, false),
                            TestUtils.WPI2020Image.FOV,
                            TestUtils.get2020LifeCamCoeffs(false))) {
                frameProvider.requestFrameThresholdType(pipeline.getThresholdType());

                try (CVPipelineResult pipelineResult =
                        pipeline.run(frameProvider.get(), QuirkyCamera.DefaultCamera)) {
                    TestUtils.printTestResultsWithLocation(pipelineResult);

                    // Draw on input
                    var outputPipe = new OutputStreamPipeline();
                    var ret =
                            outputPipe.process(
                                    pipelineResult.inputAndOutputFrame,
                                    pipeline.getSettings(),
                                    pipelineResult.targets);

                    TestUtils.showImage(
                            ret.inputAndOutputFrame.processedImage.getMat(), "Pipeline output", 999999);

                    // these numbers are not *accurate*, but they are known and expected
                    var target = pipelineResult.targets.get(0);

                    // Test corner order
                    var corners = target.getTargetCorners();
                    assertEquals(260, corners.get(0).x, 10);
                    assertEquals(245, corners.get(0).y, 10);
                    assertEquals(315, corners.get(1).x, 10);
                    assertEquals(245, corners.get(1).y, 10);
                    assertEquals(315, corners.get(2).x, 10);
                    assertEquals(190, corners.get(2).y, 10);
                    assertEquals(260, corners.get(3).x, 10);
                    assertEquals(190, corners.get(3).y, 10);

                    pose = target.getBestCameraToTarget3d();
                }
            }
        }
        // Test pose estimate translation
        assertEquals(2, pose.getTranslation().getX(), 0.2);
        assertEquals(0.1, pose.getTranslation().getY(), 0.2);
        assertEquals(0.0, pose.getTranslation().getZ(), 0.2);

        // Test pose estimate rotation
        // We expect the object axes to be in NWU, with the x-axis coming out of the tag
        // This visible tag is facing the camera almost parallel, so in world space:

        // The object's X axis should be (-1, 0, 0)
        assertEquals(-1, new Translation3d(1, 0, 0).rotateBy(pose.getRotation()).getX(), 0.1);
        // The object's Y axis should be (0, -1, 0)
        assertEquals(-1, new Translation3d(0, 1, 0).rotateBy(pose.getRotation()).getY(), 0.1);
        // The object's Z axis should be (0, 0, 1)
        assertEquals(1, new Translation3d(0, 0, 1).rotateBy(pose.getRotation()).getZ(), 0.1);
    }

    @Test
    public void testApriltagDistorted() {
        try (var pipeline = new AprilTagPipeline()) {
            pipeline.getSettings().inputShouldShow = true;
            pipeline.getSettings().outputShouldDraw = true;
            pipeline.getSettings().solvePNPEnabled = true;
            pipeline.getSettings().cornerDetectionAccuracyPercentage = 4;
            pipeline.getSettings().cornerDetectionUseConvexHulls = true;
            pipeline.getSettings().tagFamily = AprilTagFamily.kTag16h5;

            try (var frameProvider =
                    new FileFrameProvider(
                            TestUtils.getApriltagImagePath(TestUtils.ApriltagTestImages.kTag_corner_1280, false),
                            TestUtils.WPI2020Image.FOV,
                            TestUtils.getCoeffs(TestUtils.LIMELIGHT_480P_CAL_FILE, false))) {
                frameProvider.requestFrameThresholdType(pipeline.getThresholdType());

                try (CVPipelineResult pipelineResult =
                        pipeline.run(frameProvider.get(), QuirkyCamera.DefaultCamera)) {
                    TestUtils.printTestResultsWithLocation(pipelineResult);

                    // Draw on input
                    var outputPipe = new OutputStreamPipeline();
                    var ret =
                            outputPipe.process(
                                    pipelineResult.inputAndOutputFrame,
                                    pipeline.getSettings(),
                                    pipelineResult.targets);

                    TestUtils.showImage(
                            ret.inputAndOutputFrame.processedImage.getMat(), "Pipeline output", 999999);

                    // these numbers are not *accurate*, but they are known and expected
                    var pose = pipelineResult.targets.get(0).getBestCameraToTarget3d();
                    assertEquals(4.14, pose.getTranslation().getX(), 0.2);
                    assertEquals(2, pose.getTranslation().getY(), 0.2);
                    assertEquals(0.0, pose.getTranslation().getZ(), 0.2);
                }
            }
        }
    }

    @Test
    public void testManyDetections() {
        // Given a 36h11 pipeline
        try (var pipeline = new AprilTagPipeline()) {
            pipeline.getSettings().inputShouldShow = true;
            pipeline.getSettings().outputShouldDraw = true;
            pipeline.getSettings().solvePNPEnabled = true;
            pipeline.getSettings().cornerDetectionAccuracyPercentage = 4;
            pipeline.getSettings().cornerDetectionUseConvexHulls = true;
            pipeline.getSettings().tagFamily = AprilTagFamily.kTag36h11;
            pipeline.getSettings().outputMaximumTargets = 3; // bogus

            // when we have a picture with 280 targets
            try (var frameProvider =
                    new FileFrameProvider(
                            TestUtils.getApriltagImagePath(
                                    TestUtils.ApriltagTestImages.k36h11_stress_test, false),
                            TestUtils.WPI2020Image.FOV,
                            TestUtils.getCoeffs(TestUtils.LIMELIGHT_480P_CAL_FILE, false))) {
                frameProvider.requestFrameThresholdType(pipeline.getThresholdType());

                try (CVPipelineResult pipelineResult =
                        pipeline.run(frameProvider.get(), QuirkyCamera.DefaultCamera)) {
                    // the pipeline will only give us Byte.MAX_VALUE many
                    assertEquals(Byte.MAX_VALUE, pipelineResult.targets.size());
                }
            }
        }
    }

    @Test
    public void testMlCropRegionDoesNotLeak() {
        try (var pipeline = new AprilTagPipeline()) {
            pipeline.getSettings().tagFamily = AprilTagFamily.kTag36h11;
            pipeline.getSettings().decimate = 1;
            pipeline.getSettings().mltagEnabled = false;

            try (var frameProvider =
                    new FileFrameProvider(
                            TestUtils.getApriltagImagePath(TestUtils.ApriltagTestImages.kTag1_640_480, false),
                            TestUtils.WPI2020Image.FOV,
                            TestUtils.get2020LifeCamCoeffs(false))) {
                frameProvider.requestFrameThresholdType(pipeline.getThresholdType());

                // A normal run initialises the detector params (setPipeParamsImpl) and gives us
                // the full-frame reference detection: tag centre is near (287, 217) in this image.
                var frame = frameProvider.get();
                try (CVPipelineResult ref = pipeline.run(frame, QuirkyCamera.DefaultCamera)) {
                    assertEquals(1, ref.targets.size());
                    double refCx =
                            ref.targets.get(0).getTargetCorners().stream()
                                    .mapToDouble(p -> p.x)
                                    .average()
                                    .orElseThrow();
                    double refCy =
                            ref.targets.get(0).getTargetCorners().stream()
                                    .mapToDouble(p -> p.y)
                                    .average()
                                    .orElseThrow();

                    // Region tightly around the tag (tag spans roughly x 260-315, y 190-245), padded.
                    var region = new Rect(200, 140, 180, 160);

                    int matsBefore = CVMat.getMatCount();
                    for (int i = 0; i < 25; i++) {
                        var result = pipeline.detectInRegion(frame, region);
                        assertEquals(1, result.output.size(), "cropped region should still find the tag");
                        var det = result.output.get(0);
                        assertEquals(
                                refCx, det.getCenterX(), 3.0, "centre X must be in full-frame coordinates");
                        assertEquals(
                                refCy, det.getCenterY(), 3.0, "centre Y must be in full-frame coordinates");
                    }
                    assertEquals(matsBefore, CVMat.getMatCount(), "detectInRegion must release every crop");

                    // A region covering the whole frame is a no-op crop (CropPipe returns null); must not
                    // NPE and must still detect with zero offset.
                    var whole = pipeline.detectInRegion(frame, new Rect(0, 0, 640, 480));
                    assertEquals(1, whole.output.size());
                    assertEquals(refCx, whole.output.get(0).getCenterX(), 3.0);
                    assertEquals(matsBefore, CVMat.getMatCount());

                    // A region padded past the frame edge (negative origin) is clamped, not rejected.
                    var overhang = pipeline.detectInRegion(frame, new Rect(-50, -50, 400, 350));
                    assertTrue(overhang.output.size() >= 1);
                    assertEquals(matsBefore, CVMat.getMatCount());
                }
            }
        }
    }

    @Test
    public void testMlPaddingRestoresQuietZone() {
        try (var pipeline = new AprilTagPipeline()) {
            pipeline.getSettings().tagFamily = AprilTagFamily.kTag36h11;
            pipeline.getSettings().decimate = 1;
            pipeline.getSettings().mltagEnabled = false;

            try (var frameProvider =
                    new FileFrameProvider(
                            TestUtils.getApriltagImagePath(TestUtils.ApriltagTestImages.kTag1_640_480, false),
                            TestUtils.WPI2020Image.FOV,
                            TestUtils.get2020LifeCamCoeffs(false))) {
                frameProvider.requestFrameThresholdType(pipeline.getThresholdType());

                var frame = frameProvider.get();
                try (CVPipelineResult ref = pipeline.run(frame, QuirkyCamera.DefaultCamera)) {
                    assertEquals(1, ref.targets.size());
                    var corners = ref.targets.get(0).getTargetCorners();
                    double minX = corners.stream().mapToDouble(p -> p.x).min().orElseThrow();
                    double maxX = corners.stream().mapToDouble(p -> p.x).max().orElseThrow();
                    double minY = corners.stream().mapToDouble(p -> p.y).min().orElseThrow();
                    double maxY = corners.stream().mapToDouble(p -> p.y).max().orElseThrow();
                    double refCx = corners.stream().mapToDouble(p -> p.x).average().orElseThrow();
                    double refCy = corners.stream().mapToDouble(p -> p.y).average().orElseThrow();

                    // Tight to the tag border — no quiet zone. The detector should miss.
                    var tight =
                            new Rect(
                                    (int) Math.round(minX),
                                    (int) Math.round(minY),
                                    (int) Math.round(maxX - minX),
                                    (int) Math.round(maxY - minY));
                    var tightResult = pipeline.detectInRegion(frame, tight);
                    assertEquals(
                            0, tightResult.output.size(), "a crop with no quiet zone should miss the tag");

                    var padPipe = new PadRectPipe();
                    padPipe.setParams(0.15);
                    var padded = padPipe.run(tight).output;
                    var paddedResult = pipeline.detectInRegion(frame, padded);
                    assertEquals(
                            1, paddedResult.output.size(), "default crop padding should restore the quiet zone");
                    assertEquals(
                            refCx,
                            paddedResult.output.get(0).getCenterX(),
                            3.0,
                            "centre X must be in full-frame coordinates");
                    assertEquals(
                            refCy,
                            paddedResult.output.get(0).getCenterY(),
                            3.0,
                            "centre Y must be in full-frame coordinates");
                }
            }
        }
    }
}
