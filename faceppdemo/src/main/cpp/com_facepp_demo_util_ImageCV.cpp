#include "com_facepp_demo_util_ImageCV.h"

#include <android/log.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#define LOG_TAG  "C_TAG"
#define LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

using namespace std;
using namespace cv;

void kmeans(Mat inputImg);
int removeConnectedComponents(Mat inputImg);
Point2f fillConvexHulltoGetCentroid(Mat inputImg, Point2f searchingArea[]);

void debugDrawSearchingAre(Mat inputImg, Point2f searchingArea[]);

JNIEXPORT jintArray JNICALL Java_com_facepp_demo_util_ImageCV_imageCVProcess(JNIEnv * env, jobject, jlong mat_Addr_L, jlong mat_Addr_R) {

    //----------------------------------------------------------------------------------------------

    ////初始化步骤

    int size = 4;
    //返回值
    //jintArray对象
    jintArray returnArray = env -> NewIntArray(size);
    //jint指针
    jint *intArray = env -> GetIntArrayElements(returnArray, JNI_FALSE);

    //从Java获取Mat
    Mat tempMat_L = *(Mat*)mat_Addr_L;
    Mat tempMat_R = *(Mat*)mat_Addr_R;

    Mat inputImg_L = Mat(tempMat_L.rows, tempMat_L.cols, CV_8UC3);
    Mat inputImg_R = Mat(tempMat_R.rows, tempMat_R.cols, CV_8UC3);

    //将4通道Mat转换为3通道Mat
    cvtColor(tempMat_L, inputImg_L, CV_RGBA2BGR);
    cvtColor(tempMat_R, inputImg_R, CV_RGBA2BGR);
    //上下翻转
    flip(inputImg_L, inputImg_L, 0);
    flip(inputImg_R, inputImg_R, 0);

    //imwrite("/sdcard/cunxie_Demo/origin_L.jpg", inputImg_L);
    //imwrite("/sdcard/cunxie_Demo/origin_R.jpg", inputImg_R);

    //----------------------------------------------------------------------------------------------

    ////处理左眼

    //k-means
    kmeans(inputImg_L);
    //imwrite("/sdcard/cunxie_Demo/kmeans_L.jpg", inputImg_L);

    //去除连通区域
    int connectedComponentsCount_L = removeConnectedComponents(inputImg_L);
    //LOGD("connectedComponentsCount_L: %d", connectedComponentsCount_L);
    //imwrite("/sdcard/cunxie_Demo/removeConnectedComponents_L.jpg", inputImg_L);

    //填充凸包获得质心
    Point2f searchingArea_L[2];
    Point2f centroid_L = fillConvexHulltoGetCentroid(inputImg_L, searchingArea_L);
    //imwrite("/sdcard/cunxie_Demo/fillConvexHulltoGetCentroid_L.jpg", inputImg_L);

    //绘制搜索区域
    debugDrawSearchingAre(inputImg_L, searchingArea_L);
    //imwrite("/sdcard/cunxie_Demo/debugDrawSearchingAre_L.jpg", inputImg_L);

    //四舍五入求得质心
    intArray[0] = round(centroid_L.x);
    intArray[1] = round(centroid_L.y);

    //----------------------------------------------------------------------------------------------

    ////处理右眼

    //k-means
    kmeans(inputImg_R);
    //imwrite("/sdcard/cunxie_Demo/kmeans_R.jpg", inputImg_R);

    //去除连通区域
    int connectedComponentsCount_R = removeConnectedComponents(inputImg_R);
    //LOGD("connectedComponentsCount_R: %d", connectedComponentsCount_R);
    //imwrite("/sdcard/cunxie_Demo/removeConnectedComponents_R.jpg", inputImg_R);

    //填充凸包获得质心
    Point2f searchingArea_R[2];
    Point2f centroid_R = fillConvexHulltoGetCentroid(inputImg_R, searchingArea_R);
    //imwrite("/sdcard/cunxie_Demo/fillConvexHulltoGetCentroid_R.jpg", inputImg_R);

    //绘制搜索区域
    debugDrawSearchingAre(inputImg_R, searchingArea_R);
    //imwrite("/sdcard/cunxie_Demo/debugDrawSearchingAre_R.jpg", inputImg_R);

    //四舍五入求得质心
    intArray[2] = round(centroid_R.x);
    intArray[3] = round(centroid_R.y);

    //----------------------------------------------------------------------------------------------

    ////返回数据

    //把jint指针中的元素设置到jintArray对象中
    env -> SetIntArrayRegion(returnArray, 0, size, intArray);

    return returnArray;
}

//k-means
void kmeans(Mat inputImg) {

    CV_Assert(!inputImg.empty());

    int index = 0;

    int width = inputImg.cols;
    int height = inputImg.rows;
    int sampleCount = width * height;
    int dims = inputImg.channels();

    //Data for clustering. An array of N-Dimensional points with float coordinates is needed.
    Mat data(sampleCount, dims, CV_32F, Scalar(10));

    //将原始的RGB数据转换到data
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            index = row * width + col;
            Vec3b bgr = inputImg.at<Vec3b>(row, col);
            data.at<float>(index, 0) = static_cast<int>(bgr[0]);
            data.at<float>(index, 1) = static_cast<int>(bgr[1]);
            data.at<float>(index, 2) = static_cast<int>(bgr[2]);
        }
    }

    //Number of clusters to split the set by.
    int k = 3;

    //Input/output integer array that stores the cluster indices for every sample.
    Mat bestLabels;

    //The algorithm termination criteria, that is, the maximum number of iterations and/or the desired accuracy. The accuracy is specified as criteria.epsilon. As soon as each of the cluster centers moves by less than criteria.epsilon on some iteration, the algorithm stops.
    TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 1.0);

    //Flag to specify the number of times the algorithm is executed using different initial labellings. The algorithm returns the labels that yield the best compactness (see the last function parameter).
    int attempts = 3;

    //Flag that can take values of cv::KmeansFlags
    int flags = KMEANS_RANDOM_CENTERS;

    //Finds centers of clusters and groups input samples around the clusters.
    kmeans(data, k, bestLabels, criteria, attempts, flags);

    //聚类后每簇的bgr值之和
    int rgbSum[k];
    for (int i = 0; i < k; i++) {
        rgbSum[i] = 0;
    }
    for (int i = 0; i < bestLabels.rows * bestLabels.cols; i++) {
        rgbSum[bestLabels.at<int>(i, 0)] += data.at<float>(i, 0) + data.at<float>(i, 1) + data.at<float>(i, 2);
    }
    //找出bgr值之和的最小值
    int num = rgbSum[0];
    int flag = 0;
    for (int i = 0; i < k; i++) {
        if (rgbSum[i] < num) {
            num = rgbSum[i];
            flag = i;
        }
    }

    //显示图像分割结果
    //把样本数据点转换回去
    Scalar blackWhite[] = {
            Scalar(255,255,255),
            Scalar(0,0,0)
    };
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            index = row * width + col;
            int label = bestLabels.at<int>(index, 0);
            if (label == flag) {
                inputImg.at<Vec3b>(row, col)[0] = blackWhite[0][0];
                inputImg.at<Vec3b>(row, col)[1] = blackWhite[0][1];
                inputImg.at<Vec3b>(row, col)[2] = blackWhite[0][2];
            } else {
                inputImg.at<Vec3b>(row, col)[0] = blackWhite[1][0];
                inputImg.at<Vec3b>(row, col)[1] = blackWhite[1][1];
                inputImg.at<Vec3b>(row, col)[2] = blackWhite[1][2];
            }
        }
    }
}

//去除连通区域
int removeConnectedComponents(Mat inputImg) {

    CV_Assert(!inputImg.empty());

    /*
     第一轮去除虹膜外部连通域
     */

    //灰度图
    Mat grayImg;

    //Converts an image from one color space to another.
    cvtColor(inputImg, grayImg, CV_BGR2GRAY);

    Mat image;
    image = grayImg;

    //destination labeled image
    Mat labels;
    //statistics output for each label, including the background label, see below for available statistics. Statistics are accessed via stats(label, COLUMN) where COLUMN is one of ConnectedComponentsTypes. The data type is CV_32S.
    Mat stats;

    //centroid output for each label, including the background label. Centroids are accessed via centroids(label, 0) for x and centroids(label, 1) for y. The data type CV_64F.
    Mat centroids;

    //computes the connected components labeled image of boolean image and also produces a statistics output for each label
    int nums_0 = connectedComponentsWithStats(image, labels, stats, centroids);

    //原图像大小
    int statArea_0[nums_0];
    for (int i = 0; i < nums_0; i++) {
        statArea_0[i] = stats.at<int>(i, cv::CC_STAT_AREA);
    }
    //对连通域面积进行排序
    sort(statArea_0, statArea_0 + nums_0);
    //背景区域
    int backGroundSize = statArea_0[nums_0 - 1];
    //虹膜区域
    int irisSize = statArea_0[nums_0 - 2];
    vector<Vec3b> colors_0(nums_0);
    for(int i = 0; i < nums_0; i++ ) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) == backGroundSize) {
            //保留背景
            colors_0[i] = Vec3b(0, 0, 0);
        } else if (stats.at<int>(i, cv::CC_STAT_AREA) == irisSize) {
            //保留核心区域
            colors_0[i] = Vec3b(255, 255, 255);
        } else {
            //第一轮去除外围连通域
            colors_0[i] = Vec3b(0, 0, 0);

        }
    }

    for( int y = 0; y < inputImg.rows; y++ ) {
        for( int x = 0; x < inputImg.cols; x++ ) {
            int label = labels.at<int>(y, x);
            CV_Assert(0 <= label && label <= nums_0);
            inputImg.at<Vec3b>(y, x) = colors_0[label];
        }
    }

    //cout << "第一轮连通域： " << nums_0 << endl;

    /*
    第二轮去除虹膜外部连通域
    */

    //Converts an image from one color space to another.
    cvtColor(inputImg, grayImg, CV_BGR2GRAY);

    threshold(grayImg, image, 0, 255, THRESH_BINARY_INV);

    int nums_1 = connectedComponentsWithStats(image, labels, stats, centroids);

    //原图像大小
    int statArea_1[nums_1];
    for (int i = 0; i < nums_1; i++) {
        statArea_1[i] = stats.at<int>(i, cv::CC_STAT_AREA);
    }
    //对连通域面积进行排序
    sort(statArea_1, statArea_1 + nums_1);
    //背景区域
    backGroundSize = statArea_1[nums_1 - 1];
    //虹膜区域
    irisSize = statArea_1[nums_1 - 2];
    vector<Vec3b> colors_1(nums_1);
    for(int i = 0; i < nums_1; i++ ) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) == backGroundSize) {
            //保留背景
            colors_1[i] = Vec3b(0, 0, 0);
        } else if (stats.at<int>(i, cv::CC_STAT_AREA) == irisSize) {
            //保留核心区域
            colors_1[i] = Vec3b(255, 255, 255);
        } else {
            //第二轮去除内部连通域
            colors_1[i] = Vec3b(255, 255, 255);

        }
    }

    for( int y = 0; y < inputImg.rows; y++ ) {
        for( int x = 0; x < inputImg.cols; x++ ) {
            int label = labels.at<int>(y, x);
            CV_Assert(0 <= label && label <= nums_1);
            inputImg.at<Vec3b>(y, x) = colors_1[label];
        }
    }

    return nums_0 + nums_1;
}

//填充凸包
Point2f fillConvexHulltoGetCentroid(Mat inputImg, Point2f searchingArea[]) {

    CV_Assert(!inputImg.empty());

    Mat grayImg;
    //Converts an image from one color space to another.
    cvtColor(inputImg, grayImg, CV_BGR2GRAY);

    //Detected contours. Each contour is stored as a vector of points (e.g. std::vector<std::vector<cv::Point> >).
    vector< vector <Point> > contours;

    //Contour retrieval mode
    int mode = RETR_CCOMP;

    //Contour approximation method
    int method = CHAIN_APPROX_NONE;

    //寻找轮廓
    findContours(grayImg, contours, mode, method);

    CV_Assert(contours.size() > 0);

    //Output convex hull. It is either an integer vector of indices or vector of points. In the first case, the hull elements are 0-based indices of the convex hull points in the original array (since the set of convex hull points is a subset of the original point set). In the second case, hull elements are the convex hull points themselves.
    vector <vector <Point> > hull(contours.size());

    //填充凸包
//    for (int i = 0; i < contours.size(); i++) {
//        //Finds the convex hull of a point set.
//        convexHull(Mat(contours[i]), hull[i]);
//        fillConvexPoly(inputImg, hull[i], Scalar(255, 255, 255), LINE_8);
//    }
    //Finds the convex hull of a point set.
    convexHull(Mat(contours[0]), hull[0]);
    fillConvexPoly(inputImg, hull[0], Scalar(255, 255, 255), LINE_8);

    //求质心
    int sumX = 0, sumY = 0;
    for (int i = 0; i < contours[0].size(); i++) {
        sumX += contours[0][i].x;
        sumY += contours[0][i].y;
    }
    Point2f centroid;
    centroid.x = sumX / contours[0].size();
    centroid.y = sumY / contours[0].size();

    //计算瞳孔中心搜索区域的搜索长度
    double searchingLength = sqrt(contourArea(contours[0])) / 4;
    searchingArea[0].x = centroid.x - searchingLength / 2.0;
    searchingArea[0].y = centroid.y - searchingLength / 2.0;
    searchingArea[1].x = centroid.x + searchingLength / 2.0;
    searchingArea[1].y = centroid.y + searchingLength / 2.0;

    return centroid;

}




void debugDrawSearchingAre(Mat inputImg, Point2f searchingArea[]) {

    for (int x = round(searchingArea[0].x); x < round(searchingArea[1].x); x++) {
        for (int y = round(searchingArea[0].y); y < round(searchingArea[1].y); y++) {
            inputImg.ptr<Vec3b>(y)[x][0] = 0;
            inputImg.ptr<Vec3b>(y)[x][1] = 255;
            inputImg.ptr<Vec3b>(y)[x][2] = 0;
        }
    }

}