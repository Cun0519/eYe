package com.facepp.demo.util;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class DebugCV {

    public static void saveMat(Mat inputImg, String name) {
        Bitmap bitmap = Bitmap.createBitmap(inputImg.cols(), inputImg.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(inputImg, bitmap);
        bitmap = reverseImage(bitmap);
        saveBitmap(bitmap, name);
    }

    public static void saveMat(Mat inputImg, int pointX, int pointY, String name) {
        int crossLength = 10;
        Bitmap bitmap = Bitmap.createBitmap(inputImg.cols(), inputImg.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(inputImg, bitmap);
        bitmap = reverseImage(bitmap);

        for (int x = pointX - crossLength; x < pointX + crossLength; x++) {
            bitmap.setPixel(x, pointY, Color.GREEN);
        }
        for (int y = pointY - crossLength; y < pointY + crossLength; y++) {
            bitmap.setPixel(pointX, y, Color.GREEN);
        }
        saveBitmap(bitmap, name);
    }
    private static int saveBitmap(Bitmap bmp, String eye) {

        //生成文件夹路径
        String root = "/sdcard";
        String dirName = "/cunxie_Demo";
        File appDir = new File(root, dirName);
        if (!appDir.exists()) {
            appDir.mkdirs();
        }

        String fileName = eye + ".jpg";

        //获取文件
        File file = new File(appDir, fileName);
        FileOutputStream fos = null;
        try {
            fos = new FileOutputStream(file);
            bmp.compress(Bitmap.CompressFormat.JPEG, 100, fos);
            fos.flush();
            fos.close();

            return 1;
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                if (fos != null) {
                    fos.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return -1;
    }

    private static Bitmap reverseImage(Bitmap originBitmap) {
        int w = originBitmap.getWidth();
        int h = originBitmap.getHeight();
        android.graphics.Matrix m = new android.graphics.Matrix();
        //垂直翻转
        m.setScale(1, -1);
        Bitmap reverseBitmap = Bitmap.createBitmap(originBitmap, 0, 0, w, h, m, true);
        return reverseBitmap;
    }

}
