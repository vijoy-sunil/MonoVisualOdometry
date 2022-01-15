#ifndef VOCLASS_H
#define VOCLASS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <fstream>

class VOClass{
    private:
        /* input frame dimensions
        */
        int frameW;
        int frameH;
        /* intrinsic params
        */
        double fx, fy, cx, cy;
        /* 3x4 projection matrix of camera1 (left)
        */
        cv::Mat projectionCL = cv::Mat::zeros(3, 4, CV_64F);
        /* read from calib file and store into matrix; this fn is called by 
         * getProjectionMatrix()
        */
        void constructProjectionMatrix(std::string line, cv::Mat& dest);
        /* extrinsic matrix used to find camera pose (ground truth)
        */
        cv::Mat extrinsicMat = cv::Mat::zeros(4, 4, CV_64F);
        /* vector to hold ground truth poses
        */
        std::vector<cv::Mat> groundTruth;
        /* read from poses.txt and store it into matrix
        */
        void constructExtrinsicMatrix(std::string line, cv::Mat& dest);
        /* extract R and T from extrinsic matrix
        */
        void extractRT(cv::Mat& R, cv::Mat& T, cv::Mat src);
        /* check if a feature is out of bounds
        */
        bool isOutOfBounds(cv::Point2f featurePoint);
        /* update status vector for out of bounds points
        */
        void markInvalidFeaturesBounds(std::vector<cv::Point2f> featurePoints, 
        std::vector<unsigned char>& status);
        /* get number of valid matches (count of one's) in status vector
        */
        int countValidMatches(std::vector<unsigned char> status);
        /* remove invalid features based on status vector
        */
        void removeInvalidFeatures(std::vector<cv::Point2f>& featurePointsPrev, 
                                   std::vector<cv::Point2f>& featurePointsCurrent, 
                                   std::vector<unsigned char> status);
        /* get scale factor from ground truth
        */
        double getScaleFactor(int frameNumber);
        /* file handler for output pose
        */
        std::ofstream estimatedPoseFileHandler;
    public:
        /* we need to hold 2 images at a time; 1x at time t and 1x at time (t+1)
        */
        cv::Mat imgLT1, imgLT2;

        VOClass(void);
        ~VOClass(void);
        /* read images from directory, manipulate file name based on frame number
        */
        bool readImagesT1T2(int frameNumber);
        /* construct projection matrix for left camera from the calibration file
        */
        bool getProjectionMatrix(const std::string calibrationFile);
        /* get ground truth output poses, so that we can compare our estimate with 
         * it at the end; number of frames is computed from this as well
        */
        bool getGroundTruthPath(const std::string groundTruthFile, int& numFrames);
        /* feature detection
        */
        std::vector<cv::Point2f> getFeaturesFAST(cv::Mat img);
        /* feature matching
        */
        std::vector<cv::Point2f> matchFeatureKLT(std::vector<cv::Point2f> &featurePointsLT1);
        /* integrated pose of the camera RPose, tPose;
        */
        cv::Mat RPose = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat tPose = cv::Mat::zeros(3, 1, CV_64F);
        /* estimate motion
        */
        cv::Mat estimateMotion(std::vector<cv::Point2f> featurePointsT1, 
                               std::vector<cv::Point2f> featurePointsT2,
                               int frameNumber);
        /* compute error
        */
        float computeErrorInPoseEstimation(std::vector<cv::Mat> estimatedTrajectory);

        /* test fns
        */
        void testShowImagePair(cv::Mat imgLeft, cv::Mat imgLeftPlusOne, int frameNumber);
        void testShowGroundTruthTrajectory(void);
        void testShowDetectedFeatures(cv::Mat img, std::vector<cv::Point2f> featurePoints);
        void testShowFeatureMatchingOpticalFlow(cv::Mat img, 
                                                std::vector<cv::Point2f> featurePointsCurrent, 
                                                std::vector<cv::Point2f> featurePointsNext, 
                                                std::vector<unsigned char> status);
        void testShowConnectMatchedFeatures(cv::Mat img1, 
                                            std::vector<cv::Point2f> fLT1, 
                                            cv::Mat img2, 
                                            std::vector<cv::Point2f> fLT2);
        void testShowTrajectoryPair(std::vector<cv::Mat> estimatedTrajectory);
        void testShowTrajectoryPairFromFile(const std::string filePath);
        /* live trajectory window
        */    
        int liveWindowR, liveWindowC;
        cv::Mat liveWindow;
        void testShowLiveTrajectory(int frameNumber, cv::Mat currentPose, int numFrames);
};
#endif /* VOCLASS_H
*/