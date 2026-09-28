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

package org.photonvision.common.dataflow.networktables;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.networktables.NTTopicSet;
import org.wpilib.networktables.NetworkTableInstance;

class NTTopicSetResourceTest {
    @BeforeAll
    static void loadLibraries() {
        LoadJNI.loadLibraries();
    }

    @Test
    void removingCameraClosesRequestAndResultPublishers() {
        try (var instance = NetworkTableInstance.create()) {
            var topics = new NTTopicSet();
            topics.subTable = instance.getTable("resource-cleanup-camera");
            for (int i = 0; i < 3; i++) {
                try {
                    topics.updateEntries();
                    assertTrue(topics.subTable.getTopic("driverModeRequest").exists());
                    assertTrue(topics.subTable.getTopic("fpsLimitRequest").exists());
                    assertTrue(topics.subTable.getTopic("enabledRequest").exists());
                } finally {
                    topics.removeEntries();
                }
                for (var name :
                        new String[] {
                            "driverModeRequest", "fpsLimitRequest", "enabledRequest", "result_proto"
                        }) {
                    assertFalse(topics.subTable.getTopic(name).exists(), "Publisher still owns " + name);
                }
            }
        }
    }
}
