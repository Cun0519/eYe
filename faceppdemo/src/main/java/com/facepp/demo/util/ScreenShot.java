package com.facepp.demo.util;

import android.graphics.Bitmap;
import android.opengl.GLES20;
import android.opengl.GLException;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import javax.microedition.khronos.opengles.GL10;

public class ScreenShot {

    private ByteBuffer l_mScreenShotBuffer, r_mScreenShotBuffer;
    private Bitmap l_mBitmap, r_mBitmap;
    private int l_width, l_height, l_x, l_y;
    private int r_width, r_height, r_x, r_y;

    public void screenShotProcess(GL10 gl, int[] leftEyeRect, int[] rightEyeRect) {
        //l_left, (mICamera.cameraHeight - l_bottom), l_right, (mICamera.cameraHeight - l_top)
        //0为图中左眼
        //1为图中右眼

        //图中左眼
        l_width = leftEyeRect[3] - leftEyeRect[1];
        l_height = leftEyeRect[2] - leftEyeRect[0];
        l_x = leftEyeRect[1];
        l_y = leftEyeRect[0];
        //图中右眼
        r_width = rightEyeRect[3] - rightEyeRect[1];
        r_height = rightEyeRect[2] - rightEyeRect[0];
        r_x = rightEyeRect[1];
        r_y = rightEyeRect[0];

        l_mScreenShotBuffer = ByteBuffer.allocate(l_width * l_height * 4);
        l_mScreenShotBuffer.position(0);
        l_mScreenShotBuffer.rewind();
        l_mBitmap = Bitmap.createBitmap(l_width, l_height, Bitmap.Config.ARGB_8888);

        r_mScreenShotBuffer = ByteBuffer.allocate(r_width * r_height * 4);
        r_mScreenShotBuffer.position(0);
        r_mScreenShotBuffer.rewind();
        r_mBitmap = Bitmap.createBitmap(r_width, r_height, Bitmap.Config.ARGB_8888);

        try {
            //截取选定区域
            gl.glReadPixels(l_x, l_y, l_width, l_height, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, l_mScreenShotBuffer);
            l_mScreenShotBuffer.rewind();
            gl.glReadPixels(r_x, r_y, r_width, r_height, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, r_mScreenShotBuffer);
            r_mScreenShotBuffer.rewind();

            //比较耗时的操作
            //放入新的线程中运行
            new Thread(new Runnable() {
                @Override
                public void run() {
                    l_mBitmap.copyPixelsFromBuffer(l_mScreenShotBuffer);
                    r_mBitmap.copyPixelsFromBuffer(r_mScreenShotBuffer);

                    //调用k-means算法
                    Bitmap l_KMeansBitmap = new ImageCluster().kmeans(l_mBitmap, 3, 10);
                    Bitmap r_KMeansBitmap = new ImageCluster().kmeans(r_mBitmap, 3, 10);

                    //进行一系列ImageCV操作
                    new ImageCV().process(l_mBitmap, r_mBitmap);

                    long timeStamp = System.currentTimeMillis();

                    //保存原始的图片
                    saveImage(l_mBitmap, timeStamp + "L");
                    saveImage(r_mBitmap, timeStamp + "R");

                    //保存k-means后的图片
                    saveImage(l_KMeansBitmap, timeStamp + "KL");
                    saveImage(r_KMeansBitmap, timeStamp + "KR");

                }
            }).start();
        } catch (GLException e) {
            e.printStackTrace();
        }
    }

    private int saveImage(Bitmap bmp, String eye) {

        //翻转Bitmap
        bmp = reverseImage(bmp);

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

    private Bitmap reverseImage(Bitmap originBitmap) {
        int w = originBitmap.getWidth();
        int h = originBitmap.getHeight();
        android.graphics.Matrix m = new android.graphics.Matrix();
        //垂直翻转
        m.setScale(1, -1);
        Bitmap reverseBitmap = Bitmap.createBitmap(originBitmap, 0, 0, w, h, m, true);
        return reverseBitmap;
    }
}
