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

//IrisCenterLocalization



/**
 * Debug
 */
class Debug {
public:
    static void debugDrawAre(Mat inputImg, Point2i area[]);
    static void debugDrawPoint(Mat inputImg, Point2i point);
    static void debugShow(Mat img);
    static void debugDrawCross(Mat inputImg, Point2i point);
};



/**
 * IrisCenterLocalizationPreProcess
 */
class IrisCenterLocalizationPreProcess {
public:
    static void kmeans(Mat inputImg);
    static int removeConnectedComponents(Mat inputImg);
    static Point2i getCentroid(Mat inputImg, Point2i searchingArea[]);
};



/**
 * IrisCenterLocator
 */
class IrisCenterLocator {
public:
    vector< vector<Mat> > ordinaryIrisTemplates;
    vector<float> ordinaryWeights;

    IrisCenterLocator* init();
    void setIrisRadiusRange(int irisRadiusRange[]);
    Point2i convolutionCore(Mat grayImage, vector<Mat> templates, Mat1b mask, float windowSizeRatio, float percentile, bool debug);
    Point2i localizeIrisCenter(Mat eyeImage, Point2i searchingArea[]);
    //void extractAccurateTemplateParametersFromMask(float returnValue[], Mat maskImage, Point2f irisCenter, float radius);
    //Mat DaugmanIrisCore(Mat eyeImage, vector<Point2f> eyeContour, float irisRadius, Point2f irisCenterPoint);
    //void localizeIrisCenterIn(Mat eyeImage, vector<Point2f> eyeContour, Point2f irisCenter, float irisRadius, Mat outputImage);
};



/**
 * IrisTemplateGeneration
 */
class IrisTemplateGeneration {
public:
    static vector<Mat> configurableTopBottomTemplates(int aIrisRadiusRange[], float topBar, float bottomBar, bool bold);
    static vector<float> getOrdinaryWeightsForTemplates();
    static vector< vector<Mat> > generateOrdinaryIrisTemplateSetWithIrisRadiusRange(int aIrisRadiusRange[]);
    static vector<Mat> generateIrisTemplatesStandard(int aIrisRadiusRange[]);
};



/**
 * CVE
 */
namespace cve {

    /**
     Identify the local maxima(s) in the given image.

     @param input A 1-channel gray image or float image.
     @param nLocMax The maximum number of local maxima to be identified.
     @param minDistBtwLocMax The minimum pixel distance between two local maximas.
     @param threshold The threshold value of local maxima. If negative(< -1), the threshold = minValue(input).
     @return A vector of pixel coordinate(s) (in Pixel2i type) of the identified local maxima(s).
     */
    std::vector<cv::Point2i> imageLocalMaxima(const cv::Mat & input, int nLocMax = 1, float minDistBtwLocMax = 1, float threshold = -1, cv::InputArray mask = cv::noArray());

    cv::Mat cropROIWithBoundaryDetection(const cv::Mat & sourceImage, const cv::Rect & roi);

    template<typename _Tp> cv::Rect_<_Tp> CvRectMakeWithCenterPointAndSize(const cv::Point_<_Tp> & centerPoint, _Tp width, _Tp height);

    /**
     return the nth value of sort result of the given matrix. The input mat is first reshaped to long vector, and the nth operation is carried out by std::nth_element function
     *-->  User should guarantee the sourceMat IS CONTINOUS (if not, clone it!) <--*

     @param sourceMat 1-channel matrix
     @param nth the bar
     @param flag CV_SORT_ASCENDING for ascending sort, CV_SORT_DESCENDING for descending sort.
     @return the n-th value
     */
    template<typename _Tp> _Tp nthValueOfMat(const cv::Mat_<_Tp> & sourceMat, int nth, int flag = CV_SORT_ASCENDING);

    /**
     top n-percentile value of the given mat. This method invokes nthValueOfMat internally.
     *-->  User should guarantee the sourceMat IS CONTINOUS (if not, clone it!) <--*

     @param sourceMat 1-channel matrix
     @param percentile percentile value in range [0.0, 1.0]
     @param flag CV_SORT_ASCENDING for ascending sort, CV_SORT_DESCENDING for descending sort
     @return the n-percentile value
     */
    template<typename _Tp> _Tp topNPercentileValueOfMat(const cv::Mat_<_Tp> & sourceMat, float percentile, int flag = CV_SORT_ASCENDING);

    /**
     remove the small blobs (or noise dots) and leaving only the largest blob
     @param bwImage binary image to be processed
     @return a binary image containing only the largest blob
     */
    cv::Mat removeSmallBlobsExceptLargest(const cv::Mat & bwImage);
}

template<typename _Tp> cv::Rect_<_Tp> cve::CvRectMakeWithCenterPointAndSize(const cv::Point_<_Tp> & centerPoint, _Tp width, _Tp height) {
    return cv::Rect_<_Tp>(centerPoint.x - width/2, centerPoint.y - height/2, width, height);
}

template<typename _Tp> _Tp cve::nthValueOfMat(const cv::Mat_<_Tp> & sourceMat, int nth, int flag) {
    cv::Mat_<_Tp> singleRow = sourceMat.reshape(0,1);
    std::vector<_Tp> vecFromMat;
    singleRow.copyTo(vecFromMat);

    if (flag == CV_SORT_DESCENDING) {
        std::nth_element(vecFromMat.begin(), vecFromMat.begin() + nth, vecFromMat.end(), std::greater<_Tp>());
    } else {
        std::nth_element(vecFromMat.begin(), vecFromMat.begin() + nth, vecFromMat.end());
    }
    return vecFromMat[nth];
}

template<typename _Tp> _Tp cve::topNPercentileValueOfMat(const cv::Mat_<_Tp> & sourceMat, float percentile, int flag) {
    int nth = 1.0f * sourceMat.rows * sourceMat.cols * percentile;
    return cve::nthValueOfMat(sourceMat, nth, flag);
}



/**
 * Main
 */
JNIEXPORT jintArray JNICALL Java_com_facepp_demo_util_ImageCV_imageCVProcess(JNIEnv * env, jobject, jlong mat_Addr_L, jlong mat_Addr_R) {

    //初始化

    // 基于当前系统的当前日期/时间
    time_t now = time(0);
    // 把 now 转换为字符串形式
    string time = ctime(&now);
    string inputJpg_L = "/sdcard/cunxie_Demo/" + time + "_input_L.jpg";
    string outputJpg_L = "/sdcard/cunxie_Demo/" + time + "_output_L.jpg";
    string inputJpg_R = "/sdcard/cunxie_Demo/" + time + "_input_R.jpg";
    string outputJpg_R = "/sdcard/cunxie_Demo/" + time + "_output_R.jpg";

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

    //保存原始眼部区域图像
    Mat eyeImg_L = inputImg_L.clone();
    Mat eyeImg_R = inputImg_R.clone();

    imwrite(inputJpg_L, inputImg_L);
    imwrite(inputJpg_R, inputImg_R);



    //处理左眼

    //k-means
    IrisCenterLocalizationPreProcess::kmeans(inputImg_L);
    //imwrite("/sdcard/cunxie_Demo/kmeans_L.jpg", inputImg_L);

    //去除连通区域
    IrisCenterLocalizationPreProcess::removeConnectedComponents(inputImg_L);
    //LOGD("connectedComponentsCount_L: %d", connectedComponentsCount_L);
    //imwrite("/sdcard/cunxie_Demo/removeConnectedComponents_L.jpg", inputImg_L);

    //填充凸包获得质心
    Point2i searchingArea_L[2];
    IrisCenterLocalizationPreProcess::getCentroid(inputImg_L, searchingArea_L);
    //imwrite("/sdcard/cunxie_Demo/getCentroid_R.jpg", inputImg_L);

    //通过卷积定位瞳孔中心
    IrisCenterLocator locator_L;
    Point2i irisCenter_L = locator_L.localizeIrisCenter(eyeImg_L, searchingArea_L);
    Debug::debugDrawCross(eyeImg_L, irisCenter_L);
    imwrite(outputJpg_L, eyeImg_L);


    //处理右眼

    //k-means
    IrisCenterLocalizationPreProcess::kmeans(inputImg_R);
    //imwrite("/sdcard/cunxie_Demo/kmeans_R.jpg", inputImg_R);

    //去除连通区域
    IrisCenterLocalizationPreProcess::removeConnectedComponents(inputImg_R);
    //LOGD("connectedComponentsCount_R: %d", connectedComponentsCount_R);
    //imwrite("/sdcard/cunxie_Demo/removeConnectedComponents_R.jpg", inputImg_R);

    //填充凸包获得质心
    Point2i searchingArea_R[2];
    IrisCenterLocalizationPreProcess::getCentroid(inputImg_R, searchingArea_R);
    //imwrite("/sdcard/cunxie_Demo/getCentroid_R.jpg", inputImg_R);

    //通过卷积定位瞳孔中心
    IrisCenterLocator locator_R;
    Point2i irisCenter_R = locator_R.localizeIrisCenter(eyeImg_R, searchingArea_R);
    Debug::debugDrawCross(eyeImg_R, irisCenter_R);
    imwrite(outputJpg_R, eyeImg_R);


    //返回数据
    intArray[0] = irisCenter_L.x;
    intArray[1] = irisCenter_L.y;
    intArray[2] = irisCenter_R.x;
    intArray[3] = irisCenter_R.y;

    //把jint指针中的元素设置到jintArray对象中
    env -> SetIntArrayRegion(returnArray, 0, size, intArray);

    return returnArray;
}




/**
 * Debug
 */
void Debug::debugDrawPoint(Mat inputImg, Point2i point) {
    inputImg.ptr<Vec3b>(point.y)[point.x][0] = 0;
    inputImg.ptr<Vec3b>(point.y)[point.x][1] = 255;
    inputImg.ptr<Vec3b>(point.y)[point.x][2] = 0;
}
void Debug::debugDrawAre(Mat inputImg, Point2i area[]) {

    for (int x = area[0].x; x < area[1].x; x++) {
        for (int y = area[0].y; y < area[1].y; y++) {
            inputImg.ptr<Vec3b>(y)[x][0] = 0;
            inputImg.ptr<Vec3b>(y)[x][1] = 255;
            inputImg.ptr<Vec3b>(y)[x][2] = 0;
        }
    }

}
void Debug::debugDrawCross(Mat inputImg, Point2i point) {
    int crossLength = 10;
    for (int x = point.x - crossLength; x < point.x + crossLength; x++) {
        inputImg.ptr<Vec3b>(point.y)[x][0] = 0;
        inputImg.ptr<Vec3b>(point.y)[x][1] = 255;
        inputImg.ptr<Vec3b>(point.y)[x][2] = 0;
    }
    for (int y = point.y - crossLength; y < point.y + crossLength; y++) {
        inputImg.ptr<Vec3b>(y)[point.x][0] = 0;
        inputImg.ptr<Vec3b>(y)[point.x][1] = 255;
        inputImg.ptr<Vec3b>(y)[point.x][2] = 0;
    }
}



/**
 * PreProcess
 */
//k-means
void IrisCenterLocalizationPreProcess::kmeans(Mat inputImg) {

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
    ::kmeans(data, k, bestLabels, criteria, attempts, flags);

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
int IrisCenterLocalizationPreProcess::removeConnectedComponents(Mat inputImg) {

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
        statArea_0[i] = stats.at<int>(i, CC_STAT_AREA);
    }
    //对连通域面积进行排序
    sort(statArea_0, statArea_0 + nums_0);
    //背景区域
    int backGroundSize = statArea_0[nums_0 - 1];
    //虹膜区域
    int irisSize = statArea_0[nums_0 - 2];
    vector<Vec3b> colors_0(nums_0);
    for(int i = 0; i < nums_0; i++ ) {
        if (stats.at<int>(i, CC_STAT_AREA) == backGroundSize) {
            //保留背景
            colors_0[i] = Vec3b(0, 0, 0);
        } else if (stats.at<int>(i, CC_STAT_AREA) == irisSize) {
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
        statArea_1[i] = stats.at<int>(i, CC_STAT_AREA);
    }
    //对连通域面积进行排序
    sort(statArea_1, statArea_1 + nums_1);
    //背景区域
    backGroundSize = statArea_1[nums_1 - 1];
    //虹膜区域
    irisSize = statArea_1[nums_1 - 2];
    vector<Vec3b> colors_1(nums_1);
    for(int i = 0; i < nums_1; i++ ) {
        if (stats.at<int>(i, CC_STAT_AREA) == backGroundSize) {
            //保留背景
            colors_1[i] = Vec3b(0, 0, 0);
        } else if (stats.at<int>(i, CC_STAT_AREA) == irisSize) {
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

//获取质心区域
Point2i IrisCenterLocalizationPreProcess::getCentroid(Mat inputImg, Point2i searchingArea[]) {

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
    //vector <vector <Point> > hull(contours.size());

    //填充凸包
    //Finds the convex hull of a point set.
    //convexHull(Mat(contours[0]), hull[0]);
    //fillConvexPoly(inputImg, hull[0], Scalar(255, 255, 255), LINE_8);

    //求质心
    int sumX = 0, sumY = 0;
    for (int i = 0; i < contours[0].size(); i++) {
        sumX += contours[0][i].x;
        sumY += contours[0][i].y;
    }
    Point2i centroid;
    centroid.x = round(sumX / contours[0].size());
    centroid.y = round(sumY / contours[0].size());

    //计算瞳孔中心搜索区域的搜索长度
    double searchingLength = sqrt(contourArea(contours[0])) / 4;
    searchingArea[0].x = round(centroid.x - searchingLength / 2.0);
    searchingArea[0].y = round(centroid.y - searchingLength / 2.0);
    searchingArea[1].x = round(centroid.x + searchingLength / 2.0);
    searchingArea[1].y = round(centroid.y + searchingLength / 2.0);

    return centroid;

}



/**
 * IrisCenterLocator
 */
IrisCenterLocator* IrisCenterLocator::init() {
    int _irisRadiusRange[2];
    _irisRadiusRange[0] = 30;
    _irisRadiusRange[1] = 40;
    this -> ordinaryIrisTemplates = IrisTemplateGeneration::generateOrdinaryIrisTemplateSetWithIrisRadiusRange(_irisRadiusRange);
    this -> ordinaryWeights = IrisTemplateGeneration::getOrdinaryWeightsForTemplates();

    return this;
}

void IrisCenterLocator::setIrisRadiusRange(int irisRadiusRange[]) {
    int _irisRadiusRange[2];
    _irisRadiusRange[0] = irisRadiusRange[0];
    _irisRadiusRange[1] = irisRadiusRange[1];
    this -> ordinaryIrisTemplates = IrisTemplateGeneration::generateOrdinaryIrisTemplateSetWithIrisRadiusRange(_irisRadiusRange);
    this -> ordinaryWeights = IrisTemplateGeneration::getOrdinaryWeightsForTemplates();
}

Point2i IrisCenterLocator::convolutionCore(Mat grayImage, vector<Mat> templates, Mat1b mask, float windowSizeRatio, float percentile, bool debug) {
    vector<cv::Mat> convResults(templates.size());
    vector<cv::Mat> convDiff(templates.size() - 1);

    //用不同的卷积核对原图像进行卷积
    for (int i = 0; i < templates.size(); i++) {
        Mat convolutionResult;
        filter2D(grayImage, convolutionResult, CV_32F, templates[i]);
        //矩阵归一化
        cv::normalize(convolutionResult, convolutionResult, 0.0f, 1.0f, NORM_MINMAX,CV_32FC1);
        convResults[i] = convolutionResult.clone();
        if (i > 0) {
            convDiff[i-1] = cv::Mat(convResults[i] - convResults[i - 1]);
        }
    }

    // find the best center in each layer
    int squareLength = 1.0f * grayImage.cols * windowSizeRatio;
    std::vector<cv::Point2i> bestCenterInEachLayer(convDiff.size());
    std::vector<float> bestCenterInEachLayerValue(convDiff.size());
    std::vector<cv::Mat1f> bestCenterInEachLayerSurrounding(convDiff.size());

    int maxValue = 0;
    int flag = 0;

    for (int it = 0; it < convDiff.size(); it++) {
        Mat1f sourceImage = convDiff[it];
        vector<Point2i> localMaximas = cve::imageLocalMaxima(sourceImage, 1, 1, -1, mask);
        cv::Mat1f croppedGray = cve::cropROIWithBoundaryDetection(sourceImage, cve::CvRectMakeWithCenterPointAndSize(localMaximas[0], squareLength, squareLength));
        bestCenterInEachLayerValue[it] = sourceImage(localMaximas[0].y, localMaximas[0].x);
        bestCenterInEachLayer[it] = localMaximas[0];
        bestCenterInEachLayerSurrounding[it] = croppedGray;

        if (bestCenterInEachLayerValue[it] > maxValue) {
            maxValue = bestCenterInEachLayerValue[it];
            flag = it;
        }
    }

    // find the best layer
    /*
    int bestIndex = 0;
    float bestScore = -1.f;
    for (int i = 0; i < bestCenterInEachLayer.size(); i++) {
        cv::Mat1f tile = bestCenterInEachLayerSurrounding[i];
        float topValue = cve::topNPercentileValueOfMat(tile, percentile, CV_SORT_DESCENDING);
        cv::Mat bwImage;
        cv::threshold(tile, bwImage, topValue, 1.0f, CV_THRESH_BINARY);
        bwImage.convertTo(bwImage, CV_8UC1);
        bwImage = cve::removeSmallBlobsExceptLargest(bwImage);
        cv::Mat nonZeroCoordinates;
        findNonZero(bwImage, nonZeroCoordinates);
        cv::Point2f enclosingCenter;
        float enclosingRadius;
        cv::minEnclosingCircle(nonZeroCoordinates, enclosingCenter, enclosingRadius);
        float concavity  = 1.0f * nonZeroCoordinates.total() / (CV_PI * enclosingRadius * enclosingRadius);
        float score = topValue * concavity;
        if (bestScore < score) {
            bestScore = score;
            bestIndex = i;
        }
    }
     */

    return bestCenterInEachLayer[flag];
}
/*
void IrisCenterLocator::extractAccurateTemplateParametersFromMask(float returnValue[], Mat maskImage, Point2f irisCenter, float radius) {
    vector<vector<cv::Point>> contours;
    vector<Vec4i> hierarchy;

    findContours( maskImage, contours, hierarchy,
                 CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );

    if (contours.size() == 0) {
        return;
    }

    cv::Point2f leftTop = cv::Point2f(contours[0][0]), rightTop = cv::Point2f(contours[0][0]), leftBottom = cv::Point2f(contours[0][0]), rightBottom = cv::Point2f(contours[0][0]);
    for (int i = 0 ; i < contours[0].size(); i ++) {
        cv::Point2f p = cv::Point2f(contours[0][i]);
        if (cv::abs(cv::norm(irisCenter - p) - radius)<2) { // on circle edge
            if (p.x < irisCenter.x && p.y < irisCenter.y && p.y < leftTop.y) {
                leftTop = p;
            }
            if (p.x > irisCenter.x && p.y < irisCenter.y && p.y < rightTop.y) {
                rightTop = p;
            }
            if (p.x < irisCenter.x && p.y > irisCenter.y && p.y > leftBottom.y) {
                leftBottom = p;
            }
            if (p.x > irisCenter.x && p.y > irisCenter.y && p.y > rightBottom.y) {
                rightBottom = p;
            }
        }
    }

    float topBar = 0.5 - (irisCenter.y - (leftTop.y + rightTop.y) * 0.5)  / (2 * radius);
    float bottomBar = ((leftBottom.y + rightBottom.y) * 0.5 - irisCenter.y + radius) / (2 * radius);
    bottomBar -= 0.1; // the bottom bar is often lower than actual.
    if (bottomBar < 0) {
        bottomBar = 0.9;
    }

    //
    returnValue[0] = topBar;
    returnValue[1] = bottomBar;
}

Mat IrisCenterLocator::DaugmanIrisCore(Mat eyeImage, vector<Point2f> eyeContour, float irisRadius, Point2f irisCenterPoint) {
}
void IrisCenterLocator::localizeIrisCenterIn(Mat eyeImage, vector<Point2f> eyeContour, Point2f irisCenter, float irisRadius, Mat outputImage) {

}
*/

Point2i IrisCenterLocator::localizeIrisCenter(Mat eyeImage, Point2i searchingArea[]) {
    IrisCenterLocator locator;

    locator.init();

    Mat grayImg;
    //Converts an image from one color space to another.
    cvtColor(eyeImage, grayImg, CV_BGR2GRAY);

    //通过mask确定搜索区域
    Mat1b mask;
    Rect rect(searchingArea[0].x, searchingArea[0].y, searchingArea[1].x - searchingArea[0].x, searchingArea[1].y - searchingArea[0].y);
    mask = Mat::zeros(grayImg.size(), CV_8UC1);
    mask(rect).setTo(255);

    //0.2f， 0.02f
    Point2i point = locator.convolutionCore(grayImg, locator.ordinaryIrisTemplates[0], mask, 0.33f, 0.04f, false);
    //Debug::debugDrawPoint(grayImg, point);
    //Debug::debugShow(grayImg);

    return point;
}



/**
 * IrisTemplateGeneration
 */
vector<Mat> IrisTemplateGeneration::configurableTopBottomTemplates(int aIrisRadiusRange[], float topBar, float bottomBar, bool bold) {
    static Mat diskStrelKernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    vector<Mat> irisTemplates;
    // makes the template and adds them to the template vector.
    for (int radius = aIrisRadiusRange[0]; radius < aIrisRadiusRange[1]; radius += 3) {
        // draw the circle.
        Mat ring = Mat::zeros(radius * 2 + 1, radius * 2 + 1, CV_32FC1);
        circle(ring, Point2f(radius, radius), radius, Scalar::all(1));
        if (bold) {
            dilate(ring, ring, diskStrelKernel);
        }

        // cut the top and bottom part.
        int upperShelterRow = (int) (1.0f * ring.rows * topBar);
        int bottomShelterRow = (int) (1.0f * ring.rows * bottomBar);
        ring(Range(0, upperShelterRow), Range::all()) = Scalar(0.0f);
        ring(Range(bottomShelterRow, ring.rows), Range::all()) = Scalar(0.0f);

        irisTemplates.push_back(ring);
    }
    return irisTemplates;
}

vector<float> IrisTemplateGeneration::getOrdinaryWeightsForTemplates() {
    vector<float> weights;
    weights.push_back(0.4);
//    weights.push_back(0.1);
//    weights.push_back(0.1);
//    weights.push_back(0.1);
//    weights.push_back(0.3);

    return weights;
}

vector< vector<Mat> > IrisTemplateGeneration::generateOrdinaryIrisTemplateSetWithIrisRadiusRange(int aIrisRadiusRange[]) {
    vector< vector<Mat> > allIrisTemplates;
    allIrisTemplates.push_back(IrisTemplateGeneration::generateIrisTemplatesStandard(aIrisRadiusRange));

    return allIrisTemplates;
}

//标准模板
vector<Mat> IrisTemplateGeneration::generateIrisTemplatesStandard(int aIrisRadiusRange[]) {
    static Mat diskStrelKernel = getStructuringElement(MORPH_RECT, Size(3,3));

    return configurableTopBottomTemplates(aIrisRadiusRange, 0.2, 0.8, false);
}



/**
 * CVE
 */
namespace cve {

    std::vector<cv::Point2i> imageLocalMaxima(const cv::Mat& input, int nLocMax, float minDistBtwLocMax, float threshold, cv::InputArray mask) {
        cv::Mat1f playground;
        input.convertTo(playground, CV_32FC1);
        std::vector<cv::Point2i> peakPoints;
        std::vector<double> maxValues;

        if (threshold == -1) {
            double temp = 0;
            cv::minMaxLoc(playground, &temp, NULL, NULL, NULL, mask);
            threshold = temp;
        }
        for (int i = 0 ; i < nLocMax; i++) {
            cv::Point2i location;
            double maxVal;
            cv::minMaxLoc(playground, NULL, &maxVal, NULL, &location, mask);
            if (maxVal >= threshold) {
                peakPoints.push_back(location);
                maxValues.push_back(maxVal);
                cv::circle(playground, location, minDistBtwLocMax, cv::Scalar::all(threshold), -1);
            } else
                break;
        }

        return peakPoints;
    }

    cv::Mat cropROIWithBoundaryDetection(const cv::Mat & sourceImage, const cv::Rect & roi) {
        cv::Rect intRoi = roi;
        cv::Rect imageRect = cv::Rect(0,0,sourceImage.cols, sourceImage.rows);
        cv::Rect intersection = imageRect & intRoi;
        return sourceImage(intersection).clone();
    }

    cv::Mat removeSmallBlobsExceptLargest(const cv::Mat & bwImage) {
        std::vector<std::vector<cv::Point2i> > contours, onlyContours(1);
        std::vector<cv::Vec4i> hierarchy;

        findContours( bwImage.clone(), contours, hierarchy,
                      CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
        std::vector<double> areas(contours.size());
        if (contours.size() >0 ){
            for(size_t i= 0 ; i < contours.size() ; i ++) {
                areas[i] = contourArea(contours[i]);
            }

            long biggestIndex = distance(areas.begin(), max_element(areas.begin(),areas.end()));
            onlyContours[0] =contours[biggestIndex];
            cv::Mat mask(bwImage.size(),CV_8UC1,cv::Scalar::all(0));
            cv::drawContours(mask, onlyContours, -1, cv::Scalar(255), CV_FILLED);
            return mask;
        }
        return bwImage;
    }
}