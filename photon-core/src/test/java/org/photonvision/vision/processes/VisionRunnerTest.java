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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junitpioneer.jupiter.cartesian.CartesianTest;
import org.junitpioneer.jupiter.cartesian.CartesianTest.Enum;
import org.junitpioneer.jupiter.cartesian.CartesianTest.Values;
import org.opencv.core.CvType;
import org.opencv.core.Rect;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.util.numbers.IntegerCouple;
import org.photonvision.jni.LibraryLoader;
import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameProvider;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.opencv.ImageRotationMode;
import org.photonvision.vision.pipe.impl.HSVPipe;
import org.photonvision.vision.pipeline.*;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.wpilib.util.Alert;

public class VisionRunnerTest {
    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
    }

    @Test
    public void failedInitializationDoesNotAllocateAlert() {
        var provider = new RecordingFrameProvider();
        var failure =
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                new VisionRunner(
                                        provider,
                                        () -> null,
                                        result -> {},
                                        QuirkyCamera.DefaultCamera,
                                        () -> {
                                            throw new IllegalStateException("injected settings failure");
                                        },
                                        () -> -1,
                                        () -> true,
                                        () -> false));
        assertEquals("injected settings failure", failure.getMessage());
        // Failed settings initialization must leave the alert ID available.
        try (var replacement =
                new Alert("PhotonAlerts", provider.getName(), "replacement", Alert.Level.MEDIUM)) {
            replacement.set(false);
        }
    }

    @Test
    public void closeReleasesAlertForSameCamera() {
        var provider = new RecordingFrameProvider();
        for (int attempt = 0; attempt < 3; attempt++) {
            try (var runner =
                    new VisionRunner(
                            provider,
                            () -> null,
                            result -> {},
                            QuirkyCamera.DefaultCamera,
                            () -> {},
                            () -> -1,
                            () -> true,
                            () -> false)) {
                runner.close();
                // Closing must release the exact identifier, not merely deactivate the alert.
                try (var replacement =
                        new Alert("PhotonAlerts", provider.getName(), "replacement", Alert.Level.MEDIUM)) {
                    replacement.set(false);
                }
            }
        }
    }

    private static Frame observedFrame(CountDownLatch released) {
        return observedFrame(released, new FrameStaticProperties(64, 64, 70, null));
    }

    private static Frame observedFrame(CountDownLatch released, FrameStaticProperties properties) {
        var frame =
                new Frame(-1, new CVMat(), new CVMat(), FrameThresholdType.NONE, 0, properties) {
                    @Override
                    public void release() {
                        super.release();
                        released.countDown();
                    }
                };
        frame.processedImage.getMat().create(64, 64, CvType.CV_8UC1);
        return frame;
    }

    @Test
    public void pipelineChangeReleasesCapturedFrame() throws InterruptedException {
        assumeTrue(LibraryLoader.loadTargeting(), "Running the vision loop requires TimeSync JNI");
        ConfigManager.getInstance().load();
        var released = new CountDownLatch(1);
        var frame = observedFrame(released);
        var provider =
                new RecordingFrameProvider() {
                    @Override
                    public Frame get() {
                        Thread.currentThread().interrupt(); // One iteration is enough for this ownership check.
                        return frame;
                    }
                };
        var reads = new AtomicInteger();
        try (var first = new ObjectDetectionStubPipeline();
                var second = new ObjectDetectionStubPipeline();
                var runner =
                        new VisionRunner(
                                provider,
                                () -> reads.getAndIncrement() == 0 ? first : second,
                                CVPipelineResult::release,
                                QuirkyCamera.DefaultCamera,
                                () -> {},
                                () -> -1,
                                () -> true,
                                () -> false)) {
            runner.startProcess();
            assertTrue(released.await(5, TimeUnit.SECONDS), "Discarded frame must be released");
        } finally {
            frame.release();
        }
    }

    @Test
    public void cropFailureReleasesFrameAndContinues() throws InterruptedException {
        assumeTrue(LibraryLoader.loadTargeting(), "Running the vision loop requires TimeSync JNI");
        ConfigManager.getInstance().load();
        var released = new CountDownLatch(2);
        var properties =
                new FrameStaticProperties(64, 64, 70, null) {
                    @Override
                    public FrameStaticProperties crop(Rect cropRect) {
                        throw new IllegalArgumentException("injected crop failure");
                    }
                };
        var frames =
                java.util.List.of(observedFrame(released, properties), observedFrame(released, properties));
        var captures = new AtomicInteger();
        var provider =
                new RecordingFrameProvider() {
                    @Override
                    public Frame get() {
                        int capture = captures.incrementAndGet();
                        if (capture == 2) Thread.currentThread().interrupt();
                        return frames.get(capture - 1);
                    }
                };
        var pipeline = new ObjectDetectionStubPipeline();
        pipeline.getSettings().staticCropX = new IntegerCouple(0, 16);
        pipeline.getSettings().staticCropY = new IntegerCouple(0, 16);
        try (pipeline;
                var runner =
                        new VisionRunner(
                                provider,
                                () -> pipeline,
                                CVPipelineResult::release,
                                QuirkyCamera.DefaultCamera,
                                () -> {},
                                () -> -1,
                                () -> true,
                                () -> false)) {
            runner.startProcess();
            assertTrue(released.await(5, TimeUnit.SECONDS), "Both failed frames must be released");
            assertEquals(2, captures.get(), "A crop failure must not kill the vision loop");
        } finally {
            frames.forEach(Frame::release);
        }
    }

    /**
     * Stand-in for ObjectDetectionPipeline since we cannot load JNI libraries in some environments
     * {@link VisionRunner#configureFrameProviderForPipeline}.
     */
    private static final class ObjectDetectionStubPipeline
            extends CVPipeline<CVPipelineResult, AdvancedPipelineSettings> {
        ObjectDetectionStubPipeline() {
            super(ObjectDetectionPipeline.PROCESSING_TYPE);
            this.settings = new ObjectDetectionPipelineSettings();
        }

        @Override
        protected void setPipeParamsImpl() {}

        @Override
        protected CVPipelineResult process(Frame frame, AdvancedPipelineSettings settings) {
            throw new UnsupportedOperationException();
        }
    }

    // TODO consider some sort of test fixture
    private static class RecordingFrameProvider extends FrameProvider {
        boolean copyInput;
        boolean copyOutput;

        @Override
        protected boolean checkCameraConnected() {
            return true;
        }

        @Override
        public String getName() {
            return "recording";
        }

        @Override
        public void requestFrameThresholdType(FrameThresholdType type) {}

        @Override
        public void requestFrameRotation(ImageRotationMode rotationMode) {}

        @Override
        public void requestFrameCopies(boolean copyInput, boolean copyOutput) {
            this.copyInput = copyInput;
            this.copyOutput = copyOutput;
        }

        @Override
        public void requestHsvSettings(HSVPipe.HSVParams params) {}

        @Override
        public void requestBlockForFrames(boolean blockForFrames) {}

        @Override
        public Frame get() {
            return new Frame();
        }

        @Override
        public void release() {}
    }

    /**
     * Pipelines under test, with needsColor/needsProcessed hardcoded from whether {@code process}
     * reads {@code frame.colorImage} or {@code frame.processedImage} as input.
     *
     * <p>Aruco's {@code debugThreshold} color-copy behavior is covered separately.
     */
    @SuppressWarnings("rawtypes")
    public enum PipelineUnderTest {
        APRILTAG(AprilTagPipeline::new, false, true),

        ARUCO(ArucoPipeline::new, false, true),
        // When Aruco is used with debug threshold, we need to copy the color image
        ARUCO_DEBUG(
                () -> {
                    ArucoPipeline pipeline = new ArucoPipeline();
                    pipeline.getSettings().debugThreshold = true;
                    return pipeline;
                },
                true,
                true),

        REFLECTIVE(ReflectivePipeline::new, false, true),
        COLORED_SHAPE(ColoredShapePipeline::new, false, true),
        // Real ObjectDetectionPipeline constructs ObjectDetectionPipe, which eagerly Model::load()s
        // the default NN and requires detector JNI unavailable in CI.
        OBJECT_DETECTION(ObjectDetectionStubPipeline::new, true, false),
        FOCUS(FocusPipeline::new, true, false),
        DRIVER_MODE(DriverModePipeline::new, true, false),
        CALIBRATE_3D(Calibrate3dPipeline::new, true, false);

        private final Supplier<? extends CVPipeline> factory;
        private final boolean needsColor;
        private final boolean needsProcessed;

        PipelineUnderTest(
                Supplier<? extends CVPipeline> factory, boolean needsColor, boolean needsProcessed) {
            this.factory = factory;
            this.needsColor = needsColor;
            this.needsProcessed = needsProcessed;
        }

        CVPipeline create() {
            return factory.get();
        }
    }

    @CartesianTest
    public void testFrameCopyRequests(
            @Enum PipelineUnderTest pipelineUnderTest,
            @Values(booleans = {true, false}) boolean inputShouldShow,
            @Values(booleans = {true, false}) boolean outputShouldShow) {
        try (var pipeline = pipelineUnderTest.create()) {
            var provider = new RecordingFrameProvider();
            pipeline.getSettings().inputShouldShow = inputShouldShow;
            pipeline.getSettings().outputShouldShow = outputShouldShow;

            VisionRunner.configureFrameProviderForPipeline(provider, pipeline);

            boolean expectedCopyInput = pipelineUnderTest.needsColor || inputShouldShow;
            boolean expectedCopyOutput = pipelineUnderTest.needsProcessed || outputShouldShow;

            assertEquals(expectedCopyInput, provider.copyInput);
            assertEquals(expectedCopyOutput, provider.copyOutput);
        }
    }
}
