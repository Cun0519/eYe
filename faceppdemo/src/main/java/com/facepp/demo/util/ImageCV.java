package com.facepp.demo.util;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class ImageCV {

    //加载native lib
    static{
        System.loadLibrary("native-lib");
    }

    Bitmap outputBitmap_l, outputBitmap_r;
    Mat originMat_l, originMat_r;
    Mat outputMat_l, outputMat_r;

    public Bitmap[] process(Bitmap originBitmap_l, Bitmap originBitmap_r) {

        //返回值
        Bitmap[] returnBitmapArray = new Bitmap[2];

        //初始化
        init(originBitmap_l, originBitmap_r);

        //NDK处理过程
        int[] centroid = imageCVProcess(originMat_l.getNativeObjAddr(), originMat_r.getNativeObjAddr());

        Utils.matToBitmap(originMat_l, outputBitmap_l);
        Utils.matToBitmap(originMat_r, outputBitmap_r);

        Log.d("imageCV", "\n" +
                "centroid_L: " + centroid[0] + " " + centroid[1] + "\n" +
                "centroid_R: " + centroid[2] + " " + centroid[3]);

        returnBitmapArray[0] = outputBitmap_l;
        returnBitmapArray[1] = outputBitmap_r;

        return returnBitmapArray;
    }

    private void init(Bitmap originBitmap_l, Bitmap originBitmap_r) {
        outputBitmap_l = originBitmap_l.copy(originBitmap_l.getConfig(), true);
        outputBitmap_r = originBitmap_r.copy(originBitmap_r.getConfig(), true);
        originMat_l = new Mat();
        outputMat_l = new Mat();
        originMat_r = new Mat();
        outputMat_r = new Mat();

        //bitmap to mat
        Utils.bitmapToMat(originBitmap_l, originMat_l);
        Utils.bitmapToMat(originBitmap_r, originMat_r);
    }

    private native int[] imageCVProcess(long mat_Addr_l, long mat_Addr_r);
}