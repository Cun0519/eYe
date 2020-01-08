package com.facepp.demo.util;

import android.graphics.Bitmap;
import android.graphics.Color;


public class ImageCluster {
    //主要功能就是读取一副图像，再对图像进行分割

    //需要分类的簇数
    private int k;

    //迭代次数
    private int m;

    //数据集合
    private dataItem[][] source;

    //中心集合
    private dataItem[] center;

    //统计每个簇的各项数据的总和，用于计算新的点数
    private dataItem[] centerSum;

    //统计每一个簇的rgb值之和
    //用于对虹膜区域着黑色
    private double[] rgbSum;

    //用来处理获取的像素数据，提取我们需要的写入dataItem数组
    private dataItem[][] InitData(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        dataItem[][] dataitems = new dataItem[width][height];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                int pixel = bitmap.getPixel(i, j);
                dataItem di = new dataItem();
                di.r = Color.red(pixel);
                di.g = Color.green(pixel);
                di.b = Color.blue(pixel);
                di.group = 1;
                dataitems[i][j] = di;
            }
        }
        return dataitems;
    }

    //生成随机的初始中心
    private void initCenters(int k) {
        center = new dataItem[k];
        //用来统计每个聚类里面的RGB分别之和，方便计算均值
        centerSum = new dataItem[k];
        int width, height;
        for (int i = 0; i < k; i++) {
            //boolean flag=true;
            dataItem cent = new dataItem();
            dataItem cent2 = new dataItem();

            width = (int) (Math.random() * source.length);
            height = (int) (Math.random() * source[0].length);
            cent.group = i;
            cent.r = (double) source[width][height].r;
            cent.g = (double) source[width][height].g;
            cent.b = (double) source[width][height].b;
            center[i] = cent;


            cent2.r = cent.r;
            cent2.g = cent.g;
            cent2.b = cent.b;
            cent2.group = 0;
            centerSum[i] = cent2;

            width = 0;
            height = 0;
        }
    }

    //计算两个像素之间的欧式距离，用RGB作为三维坐标
    private double distance(dataItem first, dataItem second) {
        double distance = 0;
        distance = Math.sqrt(Math.pow((first.r - second.r), 2) + Math.pow((first.g - second.g), 2) +
                Math.pow((first.b - second.b), 2));
        return distance;
    }

    //返回一个数组中最小的坐标
    private int minDistance(double[] distance) {
        double minDistance = distance[0];
        int minLocation = 0;
        for (int i = 0; i < distance.length; i++) {
            if (distance[i] < minDistance) {
                minDistance = distance[i];
                minLocation = i;
            } else if (distance[i] == minDistance) {
                if ((Math.random() * 10) < 5) {
                    minLocation = i;
                }
            }
        }
        return minLocation;
    }

    //每个点进行分类
    private void clusterSet() {
        int group = -1;
        double distance[] = new double[k];
        for (int i = 0; i < source.length; i++) {
            for (int j = 0; j < source[0].length; j++) {
                //求出距离中心点最短的中心
                for (int q = 0; q < center.length; q++) {
                    distance[q] = distance(center[q], source[i][j]);
                }
                //寻找该点最近的中心
                group = minDistance(distance);
                //把该点进行分类
                source[i][j].group = group;
                //分类完求出该类的RGB和
                centerSum[group].r += source[i][j].r;
                centerSum[group].g += source[i][j].g;
                centerSum[group].b += source[i][j].b;
                //这个就是用来统计聚类里有几个点
                centerSum[group].group += 1;
                group = -1;
            }
        }
    }

    //设置新的中心
    public void setNewCenter() {
        for (int i = 0; i < centerSum.length; i++) {
            System.out.println(i + ":" + centerSum[i].group + ":" + centerSum[i].r + ":" + centerSum[i].g + ":" + centerSum[i].b);
            //取平均值为新的中心
            center[i].r = (int) (centerSum[i].r / centerSum[i].group);
            center[i].g = (int) (centerSum[i].g / centerSum[i].group);
            center[i].b = (int) (centerSum[i].b / centerSum[i].group);
            //重置之前的求和结果
            centerSum[i].r = center[i].r;
            centerSum[i].g = center[i].g;
            centerSum[i].b = center[i].b;
            centerSum[i].group = 0;
        }
    }

    //输出聚类好的数据
    private Bitmap imageDataOut(Bitmap bitmap) {

        Bitmap outputBitmap = bitmap.copy(bitmap.getConfig(), true);

        for (int i = 0; i < source.length; i++) {
            for (int j = 0; j < source[0].length; j++) {
                rgbSum[source[i][j].group] += source[i][j].r + source[i][j].g + source[i][j].b;
            }
        }
        //计算rgb值之和最小的group
        double num = rgbSum[0]; //0为第一个数组下标
        int flag = 0;
        for (int i = 0; i < rgbSum.length; i++) { //开始循环一维数组
            if (rgbSum[i] < num) {
                num = rgbSum[i];
                flag = i;
            }
        }
        //重新着色
        for (int i = 0; i < source.length; i++) {
            for (int j = 0; j < source[0].length; j++) {
                if (source[i][j].group == flag) {
                    outputBitmap.setPixel(i, j, Color.BLACK);
                } else {
                    outputBitmap.setPixel(i, j, Color.WHITE);
                }
            }
        }

        return outputBitmap;
    }

    //进行kmeans计算的核心函数
    public Bitmap kmeans(Bitmap bitmap, int k, int m) {
        source = InitData(bitmap);
        this.k = k;
        this.m = m;
        rgbSum = new double[k];
        //初始化聚类中心
        initCenters(k);

        //进行m次聚类
        for (int level = 0; level < m; level++) {
            clusterSet();
            setNewCenter();
        }
        clusterSet();
        return imageDataOut(bitmap);
    }

}

class dataItem {
    public double r;
    public double g;
    public double b;
    public int group;
}