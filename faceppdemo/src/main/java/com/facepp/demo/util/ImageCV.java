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

    private static final String TAG = "ImageCV";

    Mat originMat_L, originMat_R;

    //Singleton
    private volatile static ImageCV instance;

    private ImageCV() {
        //初始化
        originMat_L = new Mat();
        originMat_R = new Mat();
    }

    public static ImageCV getInstance() {
        if (instance == null) {
            synchronized (ImageCV.class) {
                if (instance == null) {
                    instance = new ImageCV();
                }
            }
        }

        return instance;
    }

    //求虹膜中心坐标
    public int[] process(Bitmap l_mBitmap, Bitmap r_mBitmap) {

        //bitmap to mat
        Utils.bitmapToMat(l_mBitmap, originMat_L);
        Utils.bitmapToMat(r_mBitmap, originMat_R);

        DebugCV.saveMat(originMat_L, "L_IN");
        DebugCV.saveMat(originMat_R, "R_IN");

        //NDK处理过程
        int[] returnArray = imageCVProcess(originMat_L.getNativeObjAddr(), originMat_R.getNativeObjAddr());

        DebugCV.saveMat(originMat_L, returnArray[0], returnArray[1], "L_OUT");
        DebugCV.saveMat(originMat_R, returnArray[2], returnArray[3], "R_OUT");
        Log.d(TAG, " center_L: " + returnArray[0] + " " + returnArray[1] + " center_R: " + returnArray[2] + " " + returnArray[3]);

        return returnArray;
    }

    private native int[] imageCVProcess(long mat_Addr_l, long mat_Addr_r);
}