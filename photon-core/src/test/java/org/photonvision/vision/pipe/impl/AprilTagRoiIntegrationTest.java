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

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Rect;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.estimation.TargetModel;
import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipeline.AprilTagPipeline;
import org.photonvision.vision.target.TrackedTarget;

class AprilTagRoiIntegrationTest {
    @BeforeAll
    static void loadLibraries() {
        LoadJNI.loadLibraries();
        ConfigManager.getInstance().load();
    }

    @Test
    void fullFrameFallbackPreservesSeparateTagsWithTheSameId() {
        try (var fixture = new AprilTagImageFixture()) {
            fixture.drawTag(1, new Rect(200, 200, 160, 160), false, 1);
            fixture.drawTag(1, new Rect(600, 200, 160, 160), false, 1);
            fixture.settings.mltagEnabled = true;
            fixture.settings.mltagFallbackEnabled = true;
            fixture.settings.solvePNPEnabled = false;
            // Empty model input produces no proposals through the normal ObjectDetectionPipe.
            try (var pipeline = new AprilTagPipeline(fixture.settings);
                    var frame =
                            new Frame(
                                    1,
                                    new CVMat(),
                                    new CVMat(fixture.image.getMat().clone()),
                                    FrameThresholdType.GREYSCALE,
                                    new FrameStaticProperties(1280, 720, 70, null));
                    var result = pipeline.run(frame, QuirkyCamera.DefaultCamera)) {
                assertEquals(
                        List.of(1, 1), result.targets.stream().map(TrackedTarget::getFiducialId).toList());
                assertTrue(result.mlROIs.isEmpty());
            }
        }
    }

    @Test
    void onlyDistinctDecodedTagsAdmitMultiTagPose() {
        var field = ConfigManager.getInstance().getConfig().getFieldLayout();
        assertTrue(field.getTagPose(1).isPresent());
        assertTrue(field.getTagPose(2).isPresent());
        try (var fixture = new AprilTagImageFixture();
                var calibration = AprilTagImageFixture.calibration();
                var multiTag = new MultiTargetPNPPipe()) {
            fixture.drawTag(1, new Rect(200, 200, 160, 160), false, 1);
            multiTag.setParams(
                    new MultiTargetPNPPipe.MultiTargetPNPPipeParams(
                            calibration, field, TargetModel.kAprilTag36h11));
            var properties = new FrameStaticProperties(1280, 720, 70, calibration);
            var regions =
                    new Rect[] {
                        new Rect(180, 180, 200, 200), new Rect(184, 184, 200, 200), new Rect(580, 180, 200, 200)
                    };
            for (boolean addDistinctTag : new boolean[] {false, true}) {
                if (addDistinctTag) fixture.drawTag(2, new Rect(600, 200, 160, 160), false, 1);
                var targets = new ArrayList<TrackedTarget>();
                try {
                    for (var detection : fixture.decode(regions).detections()) {
                        targets.add(
                                new TrackedTarget(
                                        detection,
                                        null,
                                        new TrackedTarget.TargetCalculationParameters(
                                                false, null, null, null, null, properties)));
                    }
                    assertEquals(
                            addDistinctTag,
                            multiTag.run(targets).output.isPresent(),
                            "Overlapping observations of one ID cannot count as two field tags");
                } finally {
                    targets.forEach(TrackedTarget::release);
                }
            }
        }
    }
}
