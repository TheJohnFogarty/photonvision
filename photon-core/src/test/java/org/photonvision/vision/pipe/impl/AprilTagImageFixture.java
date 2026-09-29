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

import java.util.Arrays;
import java.util.List;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.calibration.CameraCalibrationCoefficients;
import org.photonvision.vision.calibration.CameraLensModel;
import org.photonvision.vision.calibration.JsonMatOfDouble;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipeline.AprilTagPipelineSettings;
import org.wpilib.vision.apriltag.AprilTagDetection;
import org.wpilib.vision.apriltag.AprilTagDetector;
import org.wpilib.vision.apriltag.AprilTagImageGenerator;

/** Real tag pixels and a real decoder, shared by ROI geometry, selection and lifetime tests. */
final class AprilTagImageFixture implements AutoCloseable {
    static final int WIDTH = 1280;
    static final int HEIGHT = 720;
    final CVMat image = new CVMat(new Mat(HEIGHT, WIDTH, CvType.CV_8UC1, new Scalar(255)));
    final AprilTagPipelineSettings settings = new AprilTagPipelineSettings();
    final AprilTagDetectionPipe detector = new AprilTagDetectionPipe();
    final AprilTagRoiDecodePipe pipe = new AprilTagRoiDecodePipe(detector);

    AprilTagImageFixture() {
        settings.mlPadding = 0;
        settings.decimate = 1;
        settings.decisionMargin = 1;
        settings.hammingDist = 0;
        configure();
    }

    void drawTag(int id, Rect bounds, boolean corruptBit, double contrast) {
        Mat tag = new Mat();
        Mat enlarged = new Mat();
        Mat destination = image.getMat().submat(bounds);
        try (var raw = AprilTagImageGenerator.generate36h11AprilTagImage(id)) {
            int width = raw.getWidth();
            int height = raw.getHeight();
            int stride = raw.getStride() > 0 ? raw.getStride() : width;
            var pixels = new byte[width * height];
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    pixels[y * width + x] = raw.getData().get(y * stride + x);
                }
            }
            if (corruptBit) {
                // The central cell is data, safely inside the black border and white quiet zone.
                int center = (height / 2) * width + width / 2;
                pixels[center] = (byte) (255 - Byte.toUnsignedInt(pixels[center]));
            }
            tag.create(height, width, CvType.CV_8UC1);
            tag.put(0, 0, pixels);
            Imgproc.resize(
                    tag, enlarged, new Size(bounds.width, bounds.height), 0, 0, Imgproc.INTER_NEAREST);
            enlarged.convertTo(destination, -1, contrast, 127.5 * (1 - contrast));
        } finally {
            destination.release();
            enlarged.release();
            tag.release();
        }
    }

    static List<NeuralNetworkPipeResult> regions(Rect... regions) {
        return Arrays.stream(regions)
                .map(r -> new NeuralNetworkPipeResult(new Rect2d(r.x, r.y, r.width, r.height), 0, 0.9))
                .toList();
    }

    static CameraCalibrationCoefficients calibration() {
        return new CameraCalibrationCoefficients(
                new Size(WIDTH, HEIGHT),
                new JsonMatOfDouble(3, 3, new double[] {800, 0, 640, 0, 800, 360, 0, 0, 1}),
                new JsonMatOfDouble(1, 5, new double[5]),
                new double[0],
                List.of(),
                new Size(),
                0,
                CameraLensModel.LENSMODEL_OPENCV,
                new CameraCalibrationCoefficients.OptimizationInputs(List.of()));
    }

    AprilTagRoiDecodePipe.Output decode(Rect... regions) {
        configure();
        return pipe.run(new AprilTagRoiDecodePipe.Input(image, regions(regions))).output;
    }

    List<AprilTagDetection> fullframe() {
        configure();
        return detector.run(image).output;
    }

    private void configure() {
        var config = new AprilTagDetector.Config();
        config.numThreads = 1;
        config.quadDecimate = settings.decimate;
        config.refineEdges = true;
        var thresholds = new AprilTagDetector.QuadThresholdParameters();
        thresholds.minClusterPixels = 5;
        detector.setParams(
                new AprilTagDetectionPipe.AprilTagDetectionPipeParams(
                        AprilTagFamily.kTag36h11, config, thresholds));
        pipe.setParams(settings);
    }

    @Override
    public void close() {
        pipe.release();
        detector.release();
        image.release();
    }
}
