package com.facepp.demo.util;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.SimpleBlobDetector;
import org.opencv.imgproc.Imgproc;

public class ImageCV {

    public Bitmap removeSpot(Bitmap originBitmap) {

        Bitmap outputBitmap = originBitmap.copy(originBitmap.getConfig(), true);
        Mat originMat = new Mat();
        Mat outputMat = new Mat();
        MatOfKeyPoint points = new MatOfKeyPoint();
        SimpleBlobDetector simpleBlobDetector = SimpleBlobDetector.create();

        Utils.bitmapToMat(originBitmap, originMat);
        simpleBlobDetector.detect(originMat, points);
        Features2d.drawKeypoints(originMat, points, outputMat, new Scalar(0, 0, 255));
        Utils.matToBitmap(outputMat, outputBitmap);
        return outputBitmap;
    }

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
