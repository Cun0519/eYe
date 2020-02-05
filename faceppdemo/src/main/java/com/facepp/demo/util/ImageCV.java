package com.facepp.demo.util;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class ImageCV {

    //加载native lib
    static{
        System.loadLibrary("native-lib");
    }

    private static final String TAG = "ImageCV";

    Bitmap outputBitmap_L, outputBitmap_R;
    Mat originMat_L, originMat_R;
    Mat outputMat_L, outputMat_R;

    public Bitmap[] process(Bitmap l_mBitmap, Bitmap r_mBitmap) {

        //返回值
        Bitmap[] returnBitmapArray = new Bitmap[2];

        //初始化
        init(l_mBitmap, r_mBitmap);

        //NDK处理过程
        int[] centroid = imageCVProcess(originMat_L.getNativeObjAddr(), originMat_L.getNativeObjAddr());

        Utils.matToBitmap(originMat_L, outputBitmap_L);
        Utils.matToBitmap(originMat_R, outputBitmap_R);

        Log.d(TAG, " centroid_L: " + centroid[0] + " " + centroid[1] + " centroid_R: " + centroid[2] + " " + centroid[3]);

        returnBitmapArray[0] = outputBitmap_L;
        returnBitmapArray[1] = outputBitmap_R;

        return returnBitmapArray;
    }

    private void init(Bitmap l_mBitmap, Bitmap r_mBitmap) {
        outputBitmap_L = l_mBitmap.copy(l_mBitmap.getConfig(), true);
        outputBitmap_R = r_mBitmap.copy(r_mBitmap.getConfig(), true);
        originMat_L = new Mat();
        outputMat_L = new Mat();
        originMat_R = new Mat();
        outputMat_R = new Mat();

        //bitmap to mat
        Utils.bitmapToMat(l_mBitmap, originMat_L);
        Utils.bitmapToMat(r_mBitmap, originMat_R);
    }

    private native int[] imageCVProcess(long mat_Addr_l, long mat_Addr_r);
}