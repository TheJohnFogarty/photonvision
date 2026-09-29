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

import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.util.math.MathUtils;
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
import org.wpilib.math.geometry.Rotation3d;
import org.wpilib.math.geometry.Transform3d;
import org.wpilib.math.geometry.Translation3d;
import org.wpilib.vision.apriltag.AprilTagDetection;

class MLTagPoseTest {
    private static final double FX = 800, FY = 800, CX = 640, CY = 360;
    private static final double TAG_SIZE = 0.1651;
    private static final double[] IDEAL_CORNERS = {-1, 1, 1, 1, 1, -1, -1, -1};
    private static final Transform3d TAG_POSE =
            new Transform3d(
                    new Translation3d(1, 0.4, 3), new Rotation3d(0, Math.toRadians(35), Math.toRadians(10)));

    @BeforeAll
    static void loadLibraries() {
        LoadJNI.loadLibraries();
        ConfigManager.getInstance().load();
    }

    @ParameterizedTest
    @CsvSource({
        // Static crop origin, requested ML ROI, effective aligned ML origin.
        "0, 0, 800, 400, 200, 160, 800, 400",
        "0, 0, 803, 403, 200, 160, 800, 400",
        "0, 0, 0, 0, 1280, 720, 0, 0",
        "0, 0, -20, -12, 1100, 600, 0, 0",
        "160, 80, 640, 320, 200, 160, 640, 320"
    })
    void mlCropPreservesSingleTagPositionAndRotation(
            int staticX,
            int staticY,
            int roiX,
            int roiY,
            int roiWidth,
            int roiHeight,
            int effectiveX,
            int effectiveY) {
        var originalCalibration = calibration();
        var frameCalibration =
                staticX == 0 && staticY == 0
                        ? originalCalibration
                        : originalCalibration.cropCoefficients(
                                new Rect(staticX, staticY, 1280 - staticX, 720 - staticY));
        var frameH = translated(physicalHomography(), -staticX, -staticY);
        var roiH = translated(frameH, -effectiveX, -effectiveY);
        var unchangedRoiH = roiH.clone();
        var center = project(roiH, new double[] {0, 0});
        var roiDetection =
                new AprilTagDetection(
                        "tag36h11", 1, 0, 100, roiH, center[0], center[1], project(roiH, IDEAL_CORNERS));

        // Inject exact detections to isolate coordinate mapping from ML and corner-finding noise.
        // The actual pipeline performs cropping, remapping and native pose estimation.
        var model =
                new ObjectDetectionPipe() {
                    @Override
                    protected List<NeuralNetworkPipeResult> process(CVMat in) {
                        return List.of(
                                new NeuralNetworkPipeResult(new Rect2d(roiX, roiY, roiWidth, roiHeight), 0, 0.9));
                    }
                };
        var detector =
                new AprilTagDetectionPipe() {
                    @Override
                    protected List<AprilTagDetection> process(CVMat in) {
                        assertFalse(in.getMat().empty());
                        return List.of(roiDetection);
                    }
                };
        var settings = new AprilTagPipelineSettings();
        settings.mltagEnabled = true;
        settings.mltagFallbackEnabled = false;
        settings.mlPadding = 0;
        settings.solvePNPEnabled = true;
        settings.doMultiTarget = false;
        settings.numIterations = 40;
        var expected =
                MathUtils.convertOpenCVtoPhotonTransform(MathUtils.convertApriltagtoOpenCV(TAG_POSE));

        try (var pipeline = new AprilTagPipeline(settings, model, detector)) {
            // Reuse the same detection to ensure remapping does not mutate detector-owned arrays.
            for (int attempt = 0; attempt < 2; attempt++) {
                try (var frame =
                                new Frame(
                                        attempt,
                                        new CVMat(
                                                new Mat(720 - staticY, 1280 - staticX, CvType.CV_8UC1, new Scalar(128))),
                                        new CVMat(
                                                new Mat(720 - staticY, 1280 - staticX, CvType.CV_8UC1, new Scalar(128))),
                                        FrameThresholdType.GREYSCALE,
                                        new FrameStaticProperties(
                                                1280 - staticX, 720 - staticY, 70, frameCalibration));
                        var result = pipeline.run(frame, QuirkyCamera.DefaultCamera)) {
                    assertEquals(1, result.targets.size());
                    var target = result.targets.getFirst();
                    var actual = target.getBestCameraToTarget3d();
                    assertEquals(
                            0,
                            actual.getTranslation().getDistance(expected.getTranslation()),
                            1e-4,
                            "An ML crop must preserve the known tag position");
                    assertEquals(
                            0,
                            actual.getRotation().relativeTo(expected.getRotation()).getAngle(),
                            1e-4,
                            "An ML crop must preserve the known tag rotation");
                    var expectedCorners = project(frameH, IDEAL_CORNERS);
                    for (int i = 0; i < 4; i++) {
                        assertEquals(expectedCorners[2 * i], target.getTargetCorners().get(i).x, 1e-9);
                        assertEquals(expectedCorners[2 * i + 1], target.getTargetCorners().get(i).y, 1e-9);
                    }
                    assertArrayEquals(
                            unchangedRoiH,
                            roiDetection.getHomography(),
                            0,
                            "Mapping must not change the detector's homography");
                }
            }
        } finally {
            if (frameCalibration != originalCalibration) frameCalibration.release();
            originalCalibration.release();
        }
    }

    private static double[] physicalHomography() {
        var r1 = new Translation3d(TAG_SIZE / 2, 0, 0).rotateBy(TAG_POSE.getRotation());
        var r2 = new Translation3d(0, TAG_SIZE / 2, 0).rotateBy(TAG_POSE.getRotation());
        var t = TAG_POSE.getTranslation();
        return new double[] {
            FX * r1.getX() + CX * r1.getZ(),
            FX * r2.getX() + CX * r2.getZ(),
            FX * t.getX() + CX * t.getZ(),
            FY * r1.getY() + CY * r1.getZ(),
            FY * r2.getY() + CY * r2.getZ(),
            FY * t.getY() + CY * t.getZ(),
            r1.getZ(),
            r2.getZ(),
            t.getZ()
        };
    }

    private static double[] translated(double[] h, double x, double y) {
        var out = h.clone();
        for (int j = 0; j < 3; j++) {
            out[j] += x * h[6 + j];
            out[3 + j] += y * h[6 + j];
        }
        return out;
    }

    private static double[] project(double[] h, double[] points) {
        var out = new double[points.length];
        for (int i = 0; i < points.length; i += 2) {
            double x = points[i], y = points[i + 1];
            double w = h[6] * x + h[7] * y + h[8];
            out[i] = (h[0] * x + h[1] * y + h[2]) / w;
            out[i + 1] = (h[3] * x + h[4] * y + h[5]) / w;
        }
        return out;
    }

    private static CameraCalibrationCoefficients calibration() {
        return new CameraCalibrationCoefficients(
                new Size(1280, 720),
                new JsonMatOfDouble(3, 3, new double[] {FX, 0, CX, 0, FY, CY, 0, 0, 1}),
                new JsonMatOfDouble(1, 5, new double[5]),
                new double[0],
                List.of(),
                new Size(),
                0,
                CameraLensModel.LENSMODEL_OPENCV);
    }
}
