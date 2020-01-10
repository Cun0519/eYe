package com.facepp.demo.util;

import android.graphics.Bitmap;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class ImageCV {
    public Bitmap convert2Grey(Bitmap originBitmap) {
        Mat firstMat = new Mat();
        Mat temp = new Mat();
        Mat lastMat = new Mat();
        Bitmap outputBitmap = originBitmap.copy(originBitmap.getConfig(), true);
        Utils.bitmapToMat(originBitmap, firstMat);
        Imgproc.cvtColor(firstMat, temp, Imgproc.COLOR_RGB2BGR);
        Imgproc.cvtColor(temp, lastMat, Imgproc.COLOR_BGR2GRAY);
        Utils.matToBitmap(lastMat, outputBitmap);
        return outputBitmap;
    }
}
