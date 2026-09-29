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
import org.opencv.core.Rect;
import org.photonvision.common.LoadJNI;
import org.photonvision.vision.opencv.CVMat;

class AprilTagRoiResourceTest {
    @BeforeAll
    static void loadLibraries() {
        LoadJNI.loadLibraries();
    }

    @ParameterizedTest
    @CsvSource({"180, 180, 200, 200", "0, 0, 1280, 720"})
    void repeatedDecodingReleasesCropsAndKeepsInputUsable(int x, int y, int width, int height) {
        try (var fixture = new AprilTagImageFixture()) {
            fixture.drawTag(1, new Rect(200, 200, 160, 160), false, 1);
            int allocated = CVMat.getMatCount();
            for (int i = 0; i < 20; i++) {
                assertEquals(1, fixture.decode(new Rect(x, y, width, height)).detections().size());
                assertEquals(allocated, CVMat.getMatCount(), "No crop wrapper may outlive decoding");
                assertFalse(fixture.image.getMat().empty(), "The borrowed frame must remain usable");
            }
        }
    }

    @ParameterizedTest
    @CsvSource({"180, 180, 200, 200", "0, 0, 1280, 720"})
    void releasedDetectorFailureCleansCropsAndRestoresDecimation(
            int x, int y, int width, int height) {
        var fixture = new AprilTagImageFixture();
        // A real, safely rejected dependency-lifetime error, without entering JNI with invalid data.
        // Close the borrowed native detector once here; the remaining resources are closed below.
        fixture.detector.release();
        try {
            int allocated = CVMat.getMatCount();
            var failure =
                    assertThrows(
                            RuntimeException.class,
                            () ->
                                    fixture.pipe.run(
                                            new AprilTagRoiDecodePipe.Input(
                                                    fixture.image,
                                                    AprilTagImageFixture.regions(new Rect(x, y, width, height)))));
            assertEquals("Apriltag detector was released!", failure.getMessage());
            assertEquals(allocated, CVMat.getMatCount(), "A failure must not retain the crop");
            assertFalse(fixture.image.getMat().empty());
            // Both proposals exceed adaptive-decimation thresholds; restoring 1 is meaningful.
            assertEquals(
                    fixture.settings.decimate, fixture.detector.getParams().detectorParams().quadDecimate);
        } finally {
            fixture.pipe.release();
            fixture.image.release();
        }
    }
}
