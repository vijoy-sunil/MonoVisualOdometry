#include "../Include/VOClass.h"
#include "../Include/Logger.h"
#include "../Include/Utils.h"

/* this displays the image pair t1 and t2
*/
void VOClass::testShowImagePair(cv::Mat imgLeft, cv::Mat imgLeftPlusOne, int frameNumber){
    Logger.addLog(Logger.levels[TEST], "Show images pair", frameNumber);

    cv::Mat imgPair;
    /* Mat, text, point, font, font scale, color, thickness
    */
    std::string text = "FRAME: " + std::to_string(frameNumber);
    cv::putText(imgLeft, text, cv::Point(20, 40), cv::FONT_HERSHEY_DUPLEX, 
                1, cv::Scalar(255, 255, 255), 1);

    text = "FRAME: " + std::to_string(frameNumber + 1);
    cv::putText(imgLeftPlusOne, text, cv::Point(20, 40), cv::FONT_HERSHEY_DUPLEX, 
                1, cv::Scalar(255, 255, 255), 1);

    cv::vconcat(imgLeft, imgLeftPlusOne, imgPair);
    imshow("Image Pair", imgPair);
    cv::waitKey(0);
}

/* plot ground truth trajectory
*/
void VOClass::testShowGroundTruthTrajectory(void){
    int numPoints = groundTruth.size();
    Logger.addLog(Logger.levels[TEST], "Show ground truth trajectory");
    /* create an empty image
    */
    const int windowR = 800;
    const int windowC = 800;
    cv::Mat window = cv::Mat::zeros(windowR, windowC, CV_8UC3);
    for(int i = 0; i < numPoints; i++){
        /* shift origin of the path so that the entire path is visible, 
         * default origin is at top left of the screen
         *
         * NOTE:
         * Some trajectories may go out of bounds and become invisible,
         * the origin shift has to be updated accordingly
         * o---------------------> x or cols
         * |
         * |
         * |
         * | y or rows
         * v
        */

        /* the camera on the car is facing the z axis, so to get a
         * top down view, we plot x-z axis
        */
        int p1 = groundTruth[i].at<double>(0, 0) + windowC/2;
        int p2 = groundTruth[i].at<double>(2, 0) + windowR/4;
        /* img, center, radius, color, thickness
         */
        /* different color for the starting point and ending point
         */
        if(i == 0)
            cv::circle(window, cv::Point(p1, p2), 5, CV_RGB(0, 255, 0), 2);
        else if(i == numPoints - 1)
            cv::circle(window, cv::Point(p1, p2), 5, CV_RGB(255, 0, 0), 2);
        else
            cv::circle(window, cv::Point(p1, p2), 1, CV_RGB(255, 255, 0), 2);   
    }

    imshow("Ground Truth", window);
    cv::waitKey(0); 
}

/* display detected features
*/
void VOClass::testShowDetectedFeatures(cv::Mat img, std::vector<cv::Point2f> featurePoints){
    Logger.addLog(Logger.levels[TEST], "Show features detected");
    /* img is a single channel image, we will convert it to 3
     * to color our feature points
    */
    cv::Mat imgChannelChanged;
    cv::cvtColor(img, imgChannelChanged, cv::COLOR_GRAY2RGB);
    /* mark all feature  points on img
    */
    int numFeatures = featurePoints.size();
    for(int i = 0; i < numFeatures; i++){
        cv::circle(imgChannelChanged, featurePoints[i], 1, CV_RGB(0, 0, 255), 2);
    }
    imshow("Feature Points", imgChannelChanged);
    cv::waitKey(0);
}

/* display result of feature matching optical flow
*/
void VOClass::testShowFeatureMatchingOpticalFlow(cv::Mat img, 
                                   std::vector<cv::Point2f> featurePointsCurrent, 
                                   std::vector<cv::Point2f> featurePointsNext, 
                                   std::vector<unsigned char> status){
    /* even if they are of the same size, not all of them would be
     * matched, we need to look at the status vector to confirm match
    */
    assert(featurePointsCurrent.size() == featurePointsNext.size());
    Logger.addLog(Logger.levels[TEST], "Show feature matching optical flow result");
    /* img is a single channel image, we will convert it to 3
     * to color our feature points and lines
    */
    cv::Mat imgChannelChanged;
    cv::cvtColor(img, imgChannelChanged, cv::COLOR_GRAY2RGB); 
    /* line connecting current feature to prev feature point to show
     * optical flow
    */    
    for(int i = 0; i < featurePointsCurrent.size(); i++){
        if(status[i] == 1){
            cv::line(imgChannelChanged, featurePointsCurrent[i], featurePointsNext[i], 
            cv::Scalar(0, 255, 0), 2);
        }
    }
    imshow("Optical flow", imgChannelChanged);
    cv::waitKey(0);
}

/* display connection between matched feature
*/
void VOClass::testShowConnectMatchedFeatures(cv::Mat img1, 
                                            std::vector<cv::Point2f> fLT1, 
                                            cv::Mat img2, 
                                            std::vector<cv::Point2f> fLT2){
    int n = fLT1.size();
    Logger.addLog(Logger.levels[TEST], "Show connection between matched features");
    /* choose random featuer point to display
    */
    int idx = getRandomAmount(0, n);

    assert(idx < n);
    /* change images to 3 channels
    */
    cv::cvtColor(img1, img1, cv::COLOR_GRAY2RGB); 
    cv::cvtColor(img2, img2, cv::COLOR_GRAY2RGB);
    /* annotate frame
    */
    std::string text[2] = {"imgLT1", "imgLT2"};
    cv::putText(img1, text[0], cv::Point(20, 40), cv::FONT_HERSHEY_DUPLEX, 
                1, cv::Scalar(255, 255, 255), 1);
    cv::putText(img2, text[1], cv::Point(20, 40), cv::FONT_HERSHEY_DUPLEX, 
                1, cv::Scalar(255, 255, 255), 1);

    /* mark feature in frame before connecting lines
    */
    cv::circle(img1, fLT1[idx], 5, CV_RGB(0, 0, 255), 2);
    cv::circle(img2, fLT2[idx], 5, CV_RGB(0, 0, 255), 2);

    cv::Mat imgPair;
    cv::vconcat(img1, img2, imgPair);
    /* connect 
    */
    cv::line(imgPair, fLT1[idx], 
                      cv::Point(fLT2[idx].x, frameH + fLT2[idx].y), 
                      cv::Scalar(0, 255, 0), 2); 

    imshow("Feature Matching", imgPair);
    cv::waitKey(0);
}

/* plot ground truth and estimated trajectory
*/
void VOClass::testShowTrajectoryPair(std::vector<cv::Mat> estimatedTrajectory){
    int numPoints = estimatedTrajectory.size();
    Logger.addLog(Logger.levels[TEST], "Show ground truth and estimated trajectory", numPoints); 
    /* create an empty image
    */
    const int windowR = 800;
    const int windowC = 800;
    cv::Mat window = cv::Mat::zeros(windowR, windowC, CV_8UC3);   
    for(int i = 0; i < numPoints; i++){
        /* shift origins
        */
        /* the camera on the car is facing the z axis, so to get a
         * top down view, we plot x-z axis
        */
        int p1G = groundTruth[i].at<double>(0, 0) + windowC/2;
        int p2G = groundTruth[i].at<double>(2, 0) + windowR/4; 

        int p1E = estimatedTrajectory[i].at<double>(0, 0) + windowC/2;
        int p2E = estimatedTrajectory[i].at<double>(2, 0) + windowR/4;
        /* different color for the starting point and ending point
         */
        if(i == 0){
            cv::circle(window, cv::Point(p1G, p2G), 5, CV_RGB(0, 255, 0), 2);
            cv::circle(window, cv::Point(p1E, p2E), 5, CV_RGB(0, 255, 0), 2);
        }
        else if(i == numPoints - 1){
            cv::circle(window, cv::Point(p1G, p2G), 5, CV_RGB(255, 0, 0), 2);
            cv::circle(window, cv::Point(p1E, p2E), 5, CV_RGB(255, 0, 0), 2);
        }
        else{
            cv::circle(window, cv::Point(p1G, p2G), 1, CV_RGB(0, 0, 255), 2); 
            cv::circle(window, cv::Point(p1E, p2E), 1, CV_RGB(255, 255, 0), 2);
        }
    }
    imshow("Ground Truth & Estimated Trajectory", window);
    cv::waitKey(0); 
}

void VOClass::testShowTrajectoryPairFromFile(const std::string filePath){
    /* convert file contents to vector of mats
    */
    std::vector<cv::Mat> estimatedTrajectory;
    /* ifstream to read from file
    */
    std::ifstream file(filePath);
    if(file.is_open()){
        std::string line;
        /* read all lines
        */
        while(std::getline(file, line)){
            cv::Mat currentPose = cv::Mat::zeros(3, 1, CV_64F);
            /* split line into words
            */
            std::vector<std::string> sub = tokenize(line);
            for(int r = 0; r < 3; r++){
                currentPose.at<double>(r, 0) = std::stod(sub[r]);
            }
            estimatedTrajectory.push_back(currentPose);
        }
        testShowTrajectoryPair(estimatedTrajectory);
    }
    else{
        Logger.addLog(Logger.levels[WARNING], "Unable to open estimated pose file");
    }
}

/* live plot ground truth ,estimated pose and image
*/
void VOClass::testShowLiveTrajectory(int frameNumber, cv::Mat currentPose, int numFrames){
    /* min frame number passed in has to be 1
    */
    assert(frameNumber != 0);
    cv::Mat imgPair;
    /* if this function is called with frameNumber = 1, then it means that
     * currentPose is the pose estimated from frame 0 to frame 1; so we need
     * to plot frame 0 (imgLT1) and initial pose first
    */
    if(frameNumber - 1 == 0){
        int p1GOrigin = groundTruth[0].at<double>(0, 0) + liveWindowC/2;
        int p2GOrigin = groundTruth[0].at<double>(2, 0) + liveWindowR/4; 

        int p1EOrigin = 0 + liveWindowC/2;
        int p2EOrigin = 0 + liveWindowR/4;   
        /* plot points; origin point in green
        */
        cv::circle(liveWindow, cv::Point(p1GOrigin, p2GOrigin), 5, CV_RGB(0, 255, 0), 2);
        cv::circle(liveWindow, cv::Point(p1EOrigin, p2EOrigin), 5, CV_RGB(0, 255, 0), 2);
        /* use imgLT1 for first frame
        */  
        cv::Mat imgChannelChangedOrigin;
        cv::cvtColor(imgLT1, imgChannelChangedOrigin, cv::COLOR_GRAY2RGB);
        /* annotate frame
        */
        std::string text = "FRAME: " + std::to_string(frameNumber - 1);
        cv::putText(imgChannelChangedOrigin, text, cv::Point(20, 40), 
                                                   cv::FONT_HERSHEY_DUPLEX, 
                                                   1, cv::Scalar(255, 255, 255), 1);
        /* concat
        */   
        cv::vconcat(imgChannelChangedOrigin, liveWindow, imgPair);
        imshow("Live Trajectory", imgPair);
        cv::waitKey(1);        
    }
    /* frame number 1, 2, ...., numFrames - 1
    */
    /* current pose which is the last added element in estimated trajectory
    */    
    int p1G = groundTruth[frameNumber].at<double>(0, 0) + liveWindowC/2;
    int p2G = groundTruth[frameNumber].at<double>(2, 0) + liveWindowR/4; 

    int p1E = currentPose.at<double>(0, 0) + liveWindowC/2;
    int p2E = currentPose.at<double>(2, 0) + liveWindowR/4;
    /* plot points; last point in red
    */
    if(frameNumber == numFrames - 1){
        cv::circle(liveWindow, cv::Point(p1G, p2G), 5, CV_RGB(255, 0, 0), 2);
        cv::circle(liveWindow, cv::Point(p1E, p2E), 5, CV_RGB(255, 0, 0), 2);
    }
    else{
        cv::circle(liveWindow, cv::Point(p1G, p2G), 1, CV_RGB(0, 0, 255), 2);
        cv::circle(liveWindow, cv::Point(p1E, p2E), 1, CV_RGB(255, 255, 0), 2);
    }
    /* img is a single channel image, we will convert it to 3
     * to color our feature points
    */
    cv::Mat imgChannelChanged;
    cv::cvtColor(imgLT2, imgChannelChanged, cv::COLOR_GRAY2RGB);
    /* annotate frame
    */
    std::string text = "FRAME: " + std::to_string(frameNumber);
    cv::putText(imgChannelChanged, text, cv::Point(20, 40), 
                                         cv::FONT_HERSHEY_DUPLEX, 
                                         1, cv::Scalar(255, 255, 255), 1);
    /* concat
    */
    cv::vconcat(imgChannelChanged, liveWindow, imgPair);
    imshow("Live Trajectory", imgPair);
    /* wait idefinitely if last frame
    */
    if(frameNumber == numFrames - 1)
        cv::waitKey(0);
    else
        cv::waitKey(1);       
}