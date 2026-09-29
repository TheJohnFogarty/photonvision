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

package org.photonvision.vision.processes;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.CameraConfiguration;
import org.photonvision.vision.camera.PVCameraInfo;
import org.photonvision.vision.camera.USBCameras.GenericUSBCameraSettables;
import org.photonvision.vision.camera.USBCameras.USBCameraSource;
import org.wpilib.util.Alert;
import org.wpilib.vision.camera.UsbCamera;
import org.wpilib.vision.camera.UsbCameraInfo;
import org.wpilib.vision.stream.CameraServer;

class VisionModuleLifecycleTest {
    @BeforeAll
    static void loadLibraries() {
        // These failures happen before TimeSync starts, so targeting JNI is not required.
        LoadJNI.loadLibraries();
    }

    private static class TrackingUsbSource extends USBCameraSource {
        UsbCamera openedCamera;
        int releaseCount;

        TrackingUsbSource(CameraConfiguration config) {
            super(config);
        }

        @Override
        protected GenericUSBCameraSettables createSettables(
                CameraConfiguration config, UsbCamera camera) {
            openedCamera = camera;
            return super.createSettables(config, camera);
        }

        @Override
        public void release() {
            releaseCount++;
            super.release();
        }
    }

    @Test
    void failedAlertAllocationClosesNativeCameraAndSinkOnEveryRetry() {
        var info =
                PVCameraInfo.fromUsbCameraInfo(
                        new UsbCameraInfo(
                                99,
                                "/dev/photonvision-lifecycle-test-camera",
                                "Lifecycle cleanup test",
                                new String[] {"/dev/photonvision-lifecycle-test-camera"},
                                0,
                                0));
        try (var manager = new VisionModuleManager()) {
            for (int attempt = 0; attempt < 3; attempt++) {
                var config = new CameraConfiguration(info);
                // Starting in driver mode avoids broadcasting a pipeline change (and starting TimeSync).
                config.currentPipelineIndex = PipelineManager.DRIVERMODE_INDEX;
                var source = new TrackingUsbSource(config);
                // CameraServer returns the same sink that USBFrameProvider owns.
                var sink = CameraServer.getVideo(source.openedCamera);
                try (var occupied =
                        new Alert(
                                "PhotonAlerts",
                                source.getFrameProvider().getName(),
                                "occupied",
                                Alert.Level.MEDIUM)) {
                    var failure = assertThrows(RuntimeException.class, () -> manager.addSource(source));
                    assertTrue(
                            failure.getMessage().contains("Alert already allocated"),
                            "Failure must occur at alert allocation, not earlier initialization");
                    assertEquals(1, source.releaseCount);
                    assertEquals(0, source.openedCamera.getHandle(), "Camera handle must be closed");
                    assertEquals(0, sink.getHandle(), "Capture sink handle must be closed");
                    assertTrue(manager.getModules().isEmpty());
                } finally {
                    // Keep the test itself leak-free when a regression prevents manager cleanup.
                    if (source.releaseCount == 0) source.close();
                }
            }
        }
    }
}
