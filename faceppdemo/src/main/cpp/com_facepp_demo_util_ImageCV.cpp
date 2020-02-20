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
};

/**
 * PreProcess
 */
class IrisCenterLocalizationPreProcess {
public:
    static void kmeans(Mat inputImg);
    static int removeConnectedComponents(Mat inputImg);
    static Point2i fillConvexHulltoGetCentroid(Mat inputImg, Point2i searchingArea[]);
};

/**
 * Process
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
class IrisTemplateGeneration {
public:
    static vector<Mat> configurableTopBottomTemplates(int aIrisRadiusRange[], float topBar, float bottomBar, bool bold);
    static vector<float> getOrdinaryWeightsForTemplates();
    static vector< vector<Mat> > generateOrdinaryIrisTemplateSetWithIrisRadiusRange(int aIrisRadiusRange[]);
    static vector<Mat> generateIrisTemplatesStandard(int aIrisRadiusRange[]);
};
namespace cve {



    /**
     * CVE_Image_hpp
     */

    /**
     Identify the local maxima(s) in the given image.

     @param input A 1-channel gray image or float image.
     @param nLocMax The maximum number of local maxima to be identified.
     @param minDistBtwLocMax The minimum pixel distance between two local maximas.
     @param threshold The threshold value of local maxima. If negative(< -1), the threshold = minValue(input).
     @return A vector of pixel coordinate(s) (in Pixel2i type) of the identified local maxima(s).
     */
    std::vector<cv::Point2i> imageLocalMaxima(const cv::Mat & input, int nLocMax = 1, float minDistBtwLocMax = 1, float threshold = -1, cv::InputArray mask = cv::noArray());



    /**
     * CVE_Core_hpp
     */

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
     * CVE_Analysis_hpp
     */

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

JNIEXPORT jintArray JNICALL Java_com_facepp_demo_util_ImageCV_imageCVProcess(JNIEnv * env, jobject, jlong mat_Addr_L, jlong mat_Addr_R) {

    //初始化

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

    //imwrite("/sdcard/cunxie_Demo/origin_L.jpg", inputImg_L);
    //imwrite("/sdcard/cunxie_Demo/origin_R.jpg", inputImg_R);



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
    IrisCenterLocalizationPreProcess::fillConvexHulltoGetCentroid(inputImg_L, searchingArea_L);
    //imwrite("/sdcard/cunxie_Demo/fillConvexHulltoGetCentroid_L.jpg", inputImg_L);

    //通过卷积定位瞳孔中心
    IrisCenterLocator locator_L;
    Point2i irisCenter_L = locator_L.localizeIrisCenter(eyeImg_L, searchingArea_L);
    //Debug::debugDrawPoint(eyeImg_L, irisCenter_L);



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
    IrisCenterLocalizationPreProcess::fillConvexHulltoGetCentroid(inputImg_R, searchingArea_R);
    //imwrite("/sdcard/cunxie_Demo/fillConvexHulltoGetCentroid_R.jpg", inputImg_R);

    //通过卷积定位瞳孔中心
    IrisCenterLocator locator_R;
    Point2i irisCenter_R = locator_R.localizeIrisCenter(eyeImg_R, searchingArea_R);
    //Debug::debugDrawPoint(eyeImg_R, irisCenter_R);



    //返回数据

    //把jint指针中的元素设置到jintArray对象中
    env -> SetIntArrayRegion(returnArray, 0, size, intArray);

    return returnArray;
}
