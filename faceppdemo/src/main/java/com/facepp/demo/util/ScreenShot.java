package com.facepp.demo.util;

import android.graphics.Bitmap;
import android.opengl.GLES20;
import android.opengl.GLException;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import javax.microedition.khronos.opengles.GL10;

public class ScreenShot {

    //Singleton
    private volatile static ScreenShot instance;

    private ScreenShot() {

    }

    public static ScreenShot getInstance() {
        if (instance == null) {
            synchronized (Screen.class) {
                if (instance == null) {
                    instance = new ScreenShot();
                }
            }
        }

        return instance;
    }

    private static final String TAG = "ScreenShot";

    private ByteBuffer l_mScreenShotBuffer, r_mScreenShotBuffer;
    private Bitmap l_mBitmap, r_mBitmap;
    private int l_width, l_height, l_x, l_y;
    private int r_width, r_height, r_x, r_y;

    public int[] screenShotProcess(GL10 gl, int[] leftEyeRect, int[] rightEyeRect) {

        int[] pupilCenter = new int[4];

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

            l_mBitmap.copyPixelsFromBuffer(l_mScreenShotBuffer);
            r_mBitmap.copyPixelsFromBuffer(r_mScreenShotBuffer);

            //进行一系列ImageCV操作
            //获得虹膜中心坐标
            pupilCenter = ImageCV.getInstance().process(l_mBitmap, r_mBitmap);

        } catch (GLException e) {
            e.printStackTrace();
        }
        return pupilCenter;
    }

}
