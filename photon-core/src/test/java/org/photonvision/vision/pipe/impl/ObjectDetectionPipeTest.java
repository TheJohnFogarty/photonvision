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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Family;
import org.photonvision.common.configuration.NeuralNetworkModelsSettings.ModelProperties;
import org.photonvision.vision.objects.Model;
import org.photonvision.vision.objects.ObjectDetector;
import org.photonvision.vision.opencv.CVMat;

class ObjectDetectionPipeTest {
    @BeforeAll
    static void loadLibraries() {
        LoadJNI.loadLibraries();
    }

    @Test
    void failedReplacementCanReloadThePreviousModel() {
        var original = new TestModel(false);
        var failing = new TestModel(true);
        try (var pipe = new ObjectDetectionPipe();
                var frame = new CVMat(Mat.zeros(8, 8, CvType.CV_8UC3))) {
            select(pipe, original);
            pipe.run(frame);
            select(pipe, failing);
            assertThrows(IllegalStateException.class, () -> pipe.run(frame));
            assertEquals(1, original.detectors.get(0).releaseCount);

            select(pipe, original);
            pipe.run(frame);
            assertEquals(2, original.detectors.size());
            assertEquals(1, original.detectors.get(1).detectionCount);
        }
        assertEquals(1, original.detectors.get(0).releaseCount);
        assertEquals(1, original.detectors.get(1).releaseCount);
    }

    @Test
    void repeatedFailedReplacementDoesNotReleasePreviousDetectorAgain() {
        var original = new TestModel(false);
        var failing = new TestModel(true);
        try (var pipe = new ObjectDetectionPipe();
                var frame = new CVMat(Mat.zeros(8, 8, CvType.CV_8UC3))) {
            select(pipe, original);
            pipe.run(frame);
            select(pipe, failing);
            assertThrows(IllegalStateException.class, () -> pipe.run(frame));
            assertThrows(IllegalStateException.class, () -> pipe.run(frame));
            assertEquals(1, original.detectors.get(0).releaseCount);
        }
        assertEquals(1, original.detectors.get(0).releaseCount);
    }

    private static void select(ObjectDetectionPipe pipe, Model model) {
        pipe.setParams(new ObjectDetectionPipe.ObjectDetectionPipeParams(0.5, 0.5, model));
    }

    private static class TestModel implements Model {
        private final boolean failLoading;
        private final List<TestDetector> detectors = new ArrayList<>();

        TestModel(boolean failLoading) {
            this.failLoading = failLoading;
        }

        @Override
        public ObjectDetector load() {
            if (failLoading) throw new IllegalStateException("Model load failed");
            var detector = new TestDetector(this);
            detectors.add(detector);
            return detector;
        }

        @Override
        public Path getPath() {
            return Path.of("test-model");
        }

        @Override
        public String getNickname() {
            return "Test model";
        }

        @Override
        public Family getFamily() {
            return Family.RUBIK;
        }

        @Override
        public ModelProperties getProperties() {
            return null;
        }
    }

    private static class TestDetector implements ObjectDetector {
        private final Model model;
        private int releaseCount;
        private int detectionCount;

        TestDetector(Model model) {
            this.model = model;
        }

        @Override
        public Model getModel() {
            return model;
        }

        @Override
        public List<String> getClasses() {
            return List.of("tag");
        }

        @Override
        public List<NeuralNetworkPipeResult> detect(Mat frame, double nms, double confidence) {
            if (releaseCount > 0) throw new IllegalStateException("Detector was released");
            detectionCount++;
            return List.of();
        }

        @Override
        public void release() {
            releaseCount++;
        }
    }
}
