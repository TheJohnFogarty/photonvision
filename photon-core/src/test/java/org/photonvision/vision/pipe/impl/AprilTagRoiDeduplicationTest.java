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

import java.util.Comparator;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.opencv.core.Rect;
import org.photonvision.common.LoadJNI;
import org.wpilib.vision.apriltag.AprilTagDetection;

class AprilTagRoiDeduplicationTest {
    private static final Rect LEFT_TAG = new Rect(160, 160, 100, 100);
    private static final Rect RIGHT_TAG = new Rect(400, 160, 100, 100);
    private static final Rect LEFT_ROI = new Rect(144, 144, 132, 132);
    private static final Rect RIGHT_ROI = new Rect(384, 144, 132, 132);

    @BeforeAll
    static void loadLibraries() {
        LoadJNI.loadLibraries();
    }

    @Test
    void overlappingRegionsReturnOneObservationOfTheTag() {
        try (var fixture = new AprilTagImageFixture()) {
            fixture.drawTag(1, LEFT_TAG, false, 1);
            var otherRegion = new Rect(136, 136, 148, 148);
            assertEquals(1, fixture.decode(LEFT_ROI).detections().size());
            assertEquals(1, fixture.decode(otherRegion).detections().size());
            var output = fixture.decode(LEFT_ROI, otherRegion);
            assertEquals(List.of(1), output.detections().stream().map(AprilTagDetection::getId).toList());
            assertTrue(output.hadRawDetections());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void strongestObservationWinsInEitherRegionOrder(boolean strongestFirst) {
        try (var fixture = new AprilTagImageFixture()) {
            fixture.drawTag(1, LEFT_TAG, false, 0.4);
            fixture.drawTag(1, RIGHT_TAG, false, 1);
            var raw =
                    fixture.fullframe().stream()
                            .sorted(Comparator.comparingDouble(AprilTagDetection::getCenterX))
                            .toList();
            assertEquals(2, raw.size());
            assertTrue(raw.getLast().getDecisionMargin() > raw.getFirst().getDecisionMargin());
            var output =
                    strongestFirst
                            ? fixture.decode(RIGHT_ROI, LEFT_ROI)
                            : fixture.decode(LEFT_ROI, RIGHT_ROI);
            assertEquals(1, output.detections().size());
            assertEquals(450, output.detections().getFirst().getCenterX(), 1);
        }
    }

    @Test
    void higherMarginWithRejectedHammingCannotSuppressValidObservation() {
        try (var fixture = new AprilTagImageFixture()) {
            fixture.drawTag(1, LEFT_TAG, false, 0.4);
            fixture.drawTag(1, RIGHT_TAG, true, 1);
            var raw =
                    fixture.fullframe().stream()
                            .sorted(Comparator.comparingDouble(AprilTagDetection::getCenterX))
                            .toList();
            assertEquals(2, raw.size());
            assertEquals(0, raw.getFirst().getHamming());
            assertEquals(1, raw.getLast().getHamming());
            assertEquals(1, raw.getLast().getId());
            assertTrue(raw.getLast().getDecisionMargin() > raw.getFirst().getDecisionMargin());
            var output = fixture.decode(RIGHT_ROI, LEFT_ROI);
            assertEquals(1, output.detections().size());
            assertEquals(210, output.detections().getFirst().getCenterX(), 1);
            assertEquals(0, output.detections().getFirst().getHamming());
        }
    }

    @Test
    void distinctIdsRetainFirstSeenOrderWhenAnObservationIsReplaced() {
        try (var fixture = new AprilTagImageFixture()) {
            fixture.drawTag(2, LEFT_TAG, false, 0.4);
            fixture.drawTag(1, RIGHT_TAG, false, 1);
            fixture.drawTag(2, new Rect(640, 160, 100, 100), false, 1);
            var output = fixture.decode(LEFT_ROI, RIGHT_ROI, new Rect(624, 144, 132, 132));
            assertEquals(690, output.detections().getFirst().getCenterX(), 1);
            assertEquals(
                    List.of(2, 1), output.detections().stream().map(AprilTagDetection::getId).toList());
        }
    }

    @Test
    void rejectedTagsRemainDistinguishableFromAnEmptyDecode() {
        try (var fixture = new AprilTagImageFixture()) {
            var empty = fixture.decode(LEFT_ROI);
            assertTrue(empty.detections().isEmpty());
            assertFalse(empty.hadRawDetections());
            fixture.drawTag(1, LEFT_TAG, true, 1);
            var raw = fixture.fullframe();
            assertEquals(1, raw.size());
            assertEquals(1, raw.getFirst().getHamming());
            var output = fixture.decode(LEFT_ROI);
            assertTrue(output.detections().isEmpty());
            assertTrue(output.hadRawDetections());
        }
    }
}
