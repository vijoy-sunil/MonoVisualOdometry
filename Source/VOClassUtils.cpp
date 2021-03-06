#include "../Include/VOClass.h"
#include "../Include/Utils.h"
#include "../Include/Logger.h"
#include "../Include/Constants.h"

void VOClass::constructProjectionMatrix(std::string line, cv::Mat& dest){
    /* split line to words
    */
    std::vector<std::string> sub = tokenize(line);
    /* skip first word
    */
    int k = 1;
    for(int r = 0; r < 3; r++){
        for(int c = 0; c < 4; c++){
            dest.at<double>(r, c) = std::stod(sub[k++]);
        }
    }
}

void VOClass::constructExtrinsicMatrix(std::string line, cv::Mat& dest){
    /* split line to words
    */
    std::vector<std::string> sub = tokenize(line);
    int k = 0;
    /* Each line contains 12 values, and the number 12 comes from flattening
     * a 3x4 transformation matrix of the left camera with respect to the global 
     * coordinate frame. 
     * 
     * A 3x4 transfomration matrix contains a 3x3 rotation matrix horizontally 
     * stacked with a 3x1 translation vector in the form R|t
     * 
     * [Xworld, Yworld, Zworld] = [R|t] * [Xcamera, Ycamera, Zcamera]
     * The camera's coordinate system is where the Z axis points forward, the 
     * Y axis points downwards, and the X axis is horizontal
    */
    for(int r = 0; r < 3; r++){
        for(int c = 0; c < 4; c++){
            dest.at<double>(r, c) = std::stod(sub[k++]);
        }
    }
    /* add last row since that is omitted in the text file
     * last row will be [0, 0, 0, 1]
    */
    dest.at<double>(3, 3) = 1;
}

void VOClass::extractRT(cv::Mat& R, cv::Mat& T, cv::Mat src){
    /* extract 3x3 R
    */
    for(int r = 0; r < 3; r++){
        for(int c = 0; c < 3; c++){
            R.at<double>(r, c) = src.at<double>(r, c);
        }
    }
    /* extract 3x1 R
    */
    for(int r = 0; r < 3; r++)
        T.at<double>(r, 0) = src.at<double>(r, 3);
}

bool VOClass::isOutOfBounds(cv::Point2f featurePoint){
    if(featurePoint.x < 0 || featurePoint.x > frameW)
        return true;
    
    if(featurePoint.y < 0 || featurePoint.y > frameH)
        return true;

    return false;
}

void VOClass::markInvalidFeaturesBounds(std::vector<cv::Point2f> featurePoints, 
                                    std::vector<unsigned char>& status){
    int numFeatures = featurePoints.size();
    for(int i = 0; i < numFeatures; i++){
        if(isOutOfBounds(featurePoints[i]))
            status[i] = 0;
    }
}

int VOClass::countValidMatches(std::vector<unsigned char> status){
    int n = status.size();
    int numOnes = 0;
    for(int i = 0; i < n; i++){
        /* A feature point is only matched correctly if the status is set
        */
        if(status[i] == 1)
            numOnes++;
    }
    return numOnes;
}

void VOClass::removeInvalidFeatures(std::vector<cv::Point2f>& featurePointsPrev, 
                                    std::vector<cv::Point2f>& featurePointsCurrent, 
                                    std::vector<unsigned char> status){
    /* create an empty feature vector, push valid ones into this and
     * finally copy this to the original
    */
    std::vector<cv::Point2f> validPointsPrev, validPointsCurrent;
    int numFeatures = featurePointsPrev.size();
    for(int i = 0; i < numFeatures; i++){
        if(status[i] == 1){
            validPointsPrev.push_back(featurePointsPrev[i]);
            validPointsCurrent.push_back(featurePointsCurrent[i]);
        }
    }
    featurePointsPrev = validPointsPrev;
    featurePointsCurrent = validPointsCurrent;
}

double VOClass::getScaleFactor(int frameNumber){
    float scale = 0;
    /* scale = (xnext, ynext, znext) - (x, y, z)
    */
    double xNext = groundTruth[frameNumber+1].at<double>(0, 0);
    double yNext = groundTruth[frameNumber+1].at<double>(1, 0);
    double zNext = groundTruth[frameNumber+1].at<double>(2, 0);

    double xCurr = groundTruth[frameNumber].at<double>(0, 0);
    double yCurr = groundTruth[frameNumber].at<double>(1, 0);
    double zCurr = groundTruth[frameNumber].at<double>(2, 0);
#if 0
    Logger.addLog(Logger.levels[DEBUG], "Scale factor variables", xNext, yNext, zNext, 
                                                                  xCurr, yCurr, zCurr);
#endif
    scale = sqrt(pow(xNext - xCurr, 2) + 
                 pow(yNext - yCurr, 2) + 
                 pow(zNext - zCurr, 2));
    Logger.addLog(Logger.levels[INFO], "Computed scale factor", scale, frameNumber);
    return scale;
}