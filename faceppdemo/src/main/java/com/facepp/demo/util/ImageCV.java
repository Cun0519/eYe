package com.facepp.demo.util;

import android.graphics.Bitmap;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

public class ImageCV {
    Bitmap outputBitmap;
    Mat originMat;
    Mat outputMat;

    private void init(Bitmap originBitmap) {
        outputBitmap = originBitmap.copy(originBitmap.getConfig(), true);
        originMat = new Mat();
        outputMat = new Mat();

        //bitmap to mat
        Utils.bitmapToMat(originBitmap, originMat);
    }
}
