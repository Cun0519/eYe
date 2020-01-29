package com.facepp.demo.util;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

public class ImageCV {

    //加载native lib
    static{
        System.loadLibrary("imagecv-native-lib");
    }

    Bitmap outputBitmap_l, outputBitmap_r;
    Mat originMat_l, originMat_r;
    Mat outputMat_l, outputMat_r;

    long originMat_Addr_l, originMat_Addr_r;

    public void process(Bitmap originBitmap_l, Bitmap originBitmap_r) {
        //初始化
        init(originBitmap_l, originBitmap_r);

        //NDK处理过程
        int debugNum = imageCVProcess(originMat_Addr_r, originMat_Addr_r);
        Log.d("imageCVProcess", "debugNum: " + debugNum);
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

        //获取Mat地址
        originMat_Addr_l = originMat_l.getNativeObjAddr();
        originMat_Addr_r = originMat_r.getNativeObjAddr();
    }

    private native int imageCVProcess(long mat_Addr_l, long mat_Addr_r);
}