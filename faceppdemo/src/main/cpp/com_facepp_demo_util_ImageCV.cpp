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
#include <thread>

#define LOG_TAG  "C_TAG"
#define LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

using namespace std;
using namespace cv;

//IrisCenterLocalization

void mainProcess(Mat inputImg, string name, int returnValue[]);

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
    static Point2i preProcess(Mat inputImg, Point2i searchingArea[]);
    static void qualityOptimization(Mat inputImg);
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

    /**
     Smart color image segmentation using iterative k-means clustering. This method will perform multiple binary k-means clustering on the "darker" part, so that the darkest part can be clearly seperated (e.g iris area).

     @param image the image to be segmented
     @param kmeansIterations level of segmentation. Default is 4.
     @param coordinateWeight the weight factor for pixel adjacency. 0 will perform clustering purely be color, and higher coordinateWeight will weight more on pixel adjacency. Default is 0.4f
     @param kmeansRepeats times of k-means repeating for each level. Default is 4.
     @return a ranked image. higher value mean darker.
     */
    cv::Mat1b imageSegmentationViaIterativeKmeans(const cv::Mat& image, int kmeansIterations = 4, float coordinateWeight = 0.4f, int kmeansRepeats = 4);

    /**
     Fill the convex hulls in binary image.

     @return concavity-filled binary image
     */
    cv::Mat fillConvexHulls(const cv::Mat & bwImage);

    /**
     return the mass center coordinates for the blobs in the given binary image.
     */
    std::vector<cv::Point2f> blobMassCenter(const cv::Mat & bwImage);
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
    string nameL = "/sdcard/cunxie_Demo/" + time + "_L_";
    string nameR = "/sdcard/cunxie_Demo/" + time + "_R_";

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

    //多线程计算双眼
    int returnValue_L[2];
    int returnValue_R[2];
    thread t1(mainProcess, inputImg_L, nameL, returnValue_L);
    thread t2(mainProcess, inputImg_R, nameR, returnValue_R);
    t1.join();
    t2.join();

    //返回数据
    intArray[0] = returnValue_L[0];
    intArray[1] = returnValue_L[1];
    intArray[2] = returnValue_R[0];
    intArray[3] = returnValue_R[1];

    //把jint指针中的元素设置到jintArray对象中
    env -> SetIntArrayRegion(returnArray, 0, size, intArray);

    return returnArray;
}

/**
 * MainProcess
 */
void mainProcess(Mat inputImg, string name, int returnValue[]) {

    Mat eyeImg = inputImg.clone();

    //获取质心区域
    Point2i searchingArea[2];
    Point2i massCenter = IrisCenterLocalizationPreProcess::preProcess(inputImg, searchingArea);

    //通过卷积定位瞳孔中心
    IrisCenterLocator locator;
    Point2i irisCenter = locator.localizeIrisCenter(eyeImg, searchingArea);
    //Debug::debugDrawCross(eyeImg, irisCenter);

    returnValue[0] = irisCenter.x;
    returnValue[1] = irisCenter.y;
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
//画质优化
void IrisCenterLocalizationPreProcess::qualityOptimization(Mat inputImg) {

    CV_Assert(!inputImg.empty());

}

//preProcess
Point2i IrisCenterLocalizationPreProcess::preProcess(Mat inputImg, Point2i searchingArea[]) {

    CV_Assert(!inputImg.empty());

    const int iterationLevel = 5;
    cv::Mat1b ratedImage = cve::imageSegmentationViaIterativeKmeans(inputImg, iterationLevel);

    double maxAreaRatio = 0.025 * ratedImage.cols * ratedImage.cols * CV_PI / ratedImage.size().area();
    double minAreaRatio = maxAreaRatio * 0.4;
    cv::Mat irisArea;
    double currentMinArea = 1e10;
    for (int i = 2; i < iterationLevel; i++) {
        cv::Mat currentLayer = ratedImage >= i;
        double areaValue = cv::sum(currentLayer)[0] / 255.0 / ratedImage.size().area();
        if (areaValue > minAreaRatio && areaValue < maxAreaRatio && areaValue < currentMinArea) {
            irisArea = currentLayer;
            currentMinArea = areaValue;
        }
    }
    if (irisArea.empty()) {
        irisArea = ratedImage == 3;
    }

    irisArea = cve::removeSmallBlobsExceptLargest(irisArea);
    irisArea = cve::fillConvexHulls(irisArea);
    cv::Point2f massCenter = cve::blobMassCenter(irisArea)[0];

    //计算瞳孔中心搜索区域的搜索长度
    double searchingLength = sqrt(inputImg.cols);
    searchingArea[0].x = round(massCenter.x - searchingLength);
    searchingArea[0].y = round(massCenter.y - searchingLength);
    searchingArea[1].x = round(massCenter.x + searchingLength);
    searchingArea[1].y = round(massCenter.y + searchingLength);

    //Debug::debugDrawAre(inputImg, searchingArea);
    //Debug::debugShow(inputImg);

    return massCenter;
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

    for (int it = 0; it < convDiff.size(); it++) {
        Mat1f sourceImage = convDiff[it];
        vector<Point2i> localMaximas = cve::imageLocalMaxima(sourceImage, 1, 1, -1, mask);
        cv::Mat1f croppedGray = cve::cropROIWithBoundaryDetection(sourceImage, cve::CvRectMakeWithCenterPointAndSize(localMaximas[0], squareLength, squareLength));
        bestCenterInEachLayerValue[it] = sourceImage(localMaximas[0].y, localMaximas[0].x);
        bestCenterInEachLayer[it] = localMaximas[0];
        bestCenterInEachLayerSurrounding[it] = croppedGray;
    }

    // find the best layer
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

    return bestCenterInEachLayer[bestIndex];
}

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

    cv::Mat1b imageSegmentationViaIterativeKmeans(const cv::Mat& image, int kmeansIterations, float coordinateWeight, int kmeansRepeats)  {

        if (coordinateWeight <= 0) {
            coordinateWeight = 0.001;
        }

        cv::Mat1f kmeansPoints((int)image.total(), 5, 0.0f);
        image.reshape(1,(int)image.total()).convertTo(kmeansPoints.colRange(0,3), CV_32F);
        std::vector<cv::Point2f> coords(image.total());
        for(int i = 0 ; i < image.rows; i ++)
            for(int j = 0 ; j < image.cols; j ++)
                coords[i * image.cols + j] = cv::Point2f(i,j) * coordinateWeight;
        cv::Mat(coords).reshape(1, (int)coords.size()).copyTo(kmeansPoints.colRange(3,5));

        cv::Mat1b ratedImage(image.size(), 0);
        cv::Mat bestLabels, centers, colorsum;
        std::vector<cv::Point2i> darkerPointIndex;

        for(int it = 1 ; it < kmeansIterations ; it++) {
            if (kmeansPoints.rows < 2) {
                break;
            }
            cv::kmeans(kmeansPoints, 2, bestLabels, cv::TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, kmeansRepeats, 0.001), kmeansRepeats, cv::KMEANS_PP_CENTERS, centers);
            reduce(centers.colRange(0, 3), colorsum, 1, CV_REDUCE_SUM);

            darkerPointIndex.clear();
            if (colorsum.at<float>(0) < colorsum.at<float>(1)) {
                cv::findNonZero(bestLabels==0, darkerPointIndex);
            }
            else {
                cv::findNonZero(bestLabels==1, darkerPointIndex);
            }

            for (int i = 0 ; i < darkerPointIndex.size() ; i ++) {
                int indexInInteration = darkerPointIndex[i].y;
                int r = (int) (kmeansPoints(indexInInteration, 3) / coordinateWeight);
                int c = (int) (kmeansPoints(indexInInteration, 4) / coordinateWeight);
                ratedImage(r,c) += 1;
            }

            cv::Mat1f temp;
            for (int  i = 0; i <darkerPointIndex.size() ; i ++) {
                temp.push_back(kmeansPoints.row(darkerPointIndex[i].y));
            }
            temp.copyTo(kmeansPoints);
        }

        return ratedImage;
    }

    cv::Mat fillConvexHulls(const cv::Mat & bwImage) {
        std::vector<std::vector<cv::Point2i> > contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::findContours(bwImage.clone(), contours, hierarchy,
                         CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE );

        // Find the convex hull object for each contour
        std::vector< std::vector<cv::Point2i> > hull(contours.size());
        for( int i = 0; i < contours.size(); i++ )
        {
            cv::convexHull( cv::Mat(contours[i]), hull[i], false );
        }

        cv::Mat1b resultCanvas = cv::Mat1b::zeros(bwImage.size());
        cv::drawContours( resultCanvas, hull, -1, cv::Scalar(255), CV_FILLED);

        return resultCanvas;
    }

    std::vector<cv::Point2f> blobMassCenter(const cv::Mat & bwImage) {
        std::vector<cv::Point2f> centers;

        std::vector<std::vector<cv::Point2i> > contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::findContours(bwImage.clone() , contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );

        /// Get the moments
        std::vector<cv::Moments> mu(contours.size());
        for( int i = 0; i < contours.size(); i++ ) {
            mu[i] = cv::moments( contours[i], false );
        }

        ///  Get the mass centers:
        for( int i = 0; i < contours.size(); i++ ) {
            centers.push_back(cv::Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00));
        }

        return centers;

    }
}