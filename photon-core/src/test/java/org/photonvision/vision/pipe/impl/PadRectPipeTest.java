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

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Rect;
import org.photonvision.common.LoadJNI;

public class PadRectPipeTest {
    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
    }

    private static Rect pad(Rect in, double padding) {
        var pipe = new PadRectPipe();
        pipe.setParams(padding);
        return pipe.run(in).output;
    }

    @Test
    public void zeroPaddingLeavesTheRectUnchanged() {
        var in = new Rect(10, 20, 100, 50);
        assertEquals(in, pad(in, 0.0));
    }

    @Test
    public void halfPaddingGrowsEachSideByHalfTheAxis() {
        // 0.5 of 100x50 is 50px horizontally and 25px vertically on each side.
        var padded = pad(new Rect(10, 20, 100, 50), 0.5);
        assertEquals(new Rect(-40, -5, 200, 100), padded);
    }

    @Test
    public void paddingIsCeiledSoAFractionalPixelStillGrowsTheBox() {
        // 3 * 0.1 = 0.3, which truncates to 0 but ceils to 1. The detector needs that extra
        // pixel: without it a 0.01-step of the Crop Padding slider would be a no-op on small boxes.
        var padded = pad(new Rect(10, 20, 3, 3), 0.1);
        assertEquals(new Rect(9, 19, 5, 5), padded);
    }
}
