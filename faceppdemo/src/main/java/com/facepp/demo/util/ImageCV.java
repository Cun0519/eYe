package com.facepp.demo.util;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

public class ImageCV {

    //加载native lib
    static{
        System.loadLibrary("native-lib");
    }

    private static final String TAG = "ImageCV";

    Mat originMat_L, originMat_R;

    //求虹膜中心坐标
    public int[] process(Bitmap l_mBitmap, Bitmap r_mBitmap) {

        //初始化
        originMat_L = new Mat();
        originMat_R = new Mat();
        //bitmap to mat
        Utils.bitmapToMat(l_mBitmap, originMat_L);
        Utils.bitmapToMat(r_mBitmap, originMat_R);

        //NDK处理过程
        int[] returnArray = imageCVProcess(originMat_L.getNativeObjAddr(), originMat_R.getNativeObjAddr());

        Log.d(TAG, " centroid_L: " + returnArray[0] + " " + returnArray[1] + " centroid_R: " + returnArray[2] + " " + returnArray[3]);

        return returnArray;
    }

    private native int[] imageCVProcess(long mat_Addr_l, long mat_Addr_r);
}