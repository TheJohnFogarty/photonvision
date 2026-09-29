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

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.opencv.core.Core;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.photonvision.common.LoadJNI;
import org.photonvision.vision.opencv.CVMat;
import org.wpilib.math.geometry.Rotation3d;
import org.wpilib.math.geometry.Translation3d;
import org.wpilib.vision.apriltag.AprilTagPoseEstimator;

class AprilTagRoiGeometryTest {
    @BeforeAll
    static void loadLibraries() {
        LoadJNI.loadLibraries();
    }

    @ParameterizedTest
    @CsvSource({
        // Parent static-crop origin, requested model region. No effective crop origins supplied.
        "0, 0, 840, 410, 150, 140",
        "0, 0, 843, 413, 150, 140",
        "0, 0, 0, 0, 1280, 720",
        "0, 0, -20, -12, 1100, 600",
        "160, 80, 680, 330, 150, 140"
    })
    void croppingPreservesDecodedGeometryAndPose(
            int staticX, int staticY, int x, int y, int width, int height) {
        try (var fixture = new AprilTagImageFixture();
                var calibration = AprilTagImageFixture.calibration();
                var pose = new AprilTagPoseEstimatorPipe()) {
            renderTiltedTag(fixture);
            var fullDetections = fixture.fullframe();
            assertEquals(1, fullDetections.size(), "The rendered tag must really decode");
            pose.setParams(
                    new AprilTagPoseEstimatorPipe.AprilTagPoseEstimatorPipeParams(
                            new AprilTagPoseEstimator.Config(0.1651, 800, 800, 640, 360), calibration, 40));
            var expectedPose = pose.run(fullDetections.getFirst()).output.pose1;
            var parentRect = new Rect(staticX, staticY, 1280 - staticX, 720 - staticY);
            try (var parentImage = new CVMat(fixture.image.getMat().submat(parentRect));
                    var parentCalibration = calibration.cropCoefficients(parentRect)) {
                pose.setParams(
                        new AprilTagPoseEstimatorPipe.AprilTagPoseEstimatorPipeParams(
                                new AprilTagPoseEstimator.Config(0.1651, 800, 800, 640 - staticX, 360 - staticY),
                                parentCalibration,
                                40));
                var output =
                        fixture.pipe.run(
                                        new AprilTagRoiDecodePipe.Input(
                                                parentImage, AprilTagImageFixture.regions(new Rect(x, y, width, height))))
                                .output;
                assertEquals(1, output.detections().size());
                var detection = output.detections().getFirst();
                var h = detection.getHomography();
                // AprilTag's documented ideal tag coordinates, independently projected through H.
                double[] ideal = {-1, 1, 1, 1, 1, -1, -1, -1};
                for (int i = 0; i < 4; i++) {
                    double u = ideal[2 * i], v = ideal[2 * i + 1];
                    double w = h[6] * u + h[7] * v + h[8];
                    assertEquals(detection.getCornerX(i), (h[0] * u + h[1] * v + h[2]) / w, 0.001);
                    assertEquals(detection.getCornerY(i), (h[3] * u + h[4] * v + h[5]) / w, 0.001);
                    assertEquals(
                            fullDetections.getFirst().getCornerX(i) - staticX, detection.getCornerX(i), 1.0);
                    assertEquals(
                            fullDetections.getFirst().getCornerY(i) - staticY, detection.getCornerY(i), 1.0);
                }
                var actualPose = pose.run(detection).output.pose1;
                assertEquals(
                        0,
                        expectedPose.getTranslation().getDistance(actualPose.getTranslation()),
                        0.01,
                        "Cropping must preserve position within native corner-finding tolerance");
                assertEquals(
                        0,
                        expectedPose.getRotation().relativeTo(actualPose.getRotation()).getAngle(),
                        Math.toRadians(1),
                        "Cropping must preserve orientation");
            }
        }
    }

    private static void renderTiltedTag(AprilTagImageFixture fixture) {
        fixture.drawTag(1, new Rect(0, 0, 200, 200), false, 1);
        // The generated bitmap includes a white cell around the eight-cell black tag border.
        // Project its outer edges from a tilted physical plane; this only constructs input pixels.
        var rotation = new Rotation3d(0, Math.toRadians(35), Math.toRadians(10));
        var points = new Point[4];
        double[] corners = {-1, -1, 1, -1, 1, 1, -1, 1};
        for (int i = 0; i < 4; i++) {
            var p =
                    new Translation3d(corners[2 * i] * 0.1651 * 0.625, corners[2 * i + 1] * 0.1651 * 0.625, 0)
                            .rotateBy(rotation)
                            .plus(new Translation3d(0.5, 0.2, 1.5));
            points[i] = new Point(800 * p.getX() / p.getZ() + 640, 800 * p.getY() / p.getZ() + 360);
        }
        var source =
                new MatOfPoint2f(
                        new Point(0, 0), new Point(200, 0), new Point(200, 200), new Point(0, 200));
        var dest = new MatOfPoint2f(points);
        var transform = Imgproc.getPerspectiveTransform(source, dest);
        try (var rendered = new CVMat()) {
            Imgproc.warpPerspective(
                    fixture.image.getMat(),
                    rendered.getMat(),
                    transform,
                    new Size(1280, 720),
                    Imgproc.INTER_NEAREST,
                    Core.BORDER_CONSTANT,
                    new Scalar(255));
            fixture.image.copyFrom(rendered);
        } finally {
            source.release();
            dest.release();
            transform.release();
        }
    }
}
