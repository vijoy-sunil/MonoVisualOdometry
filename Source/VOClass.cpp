#include "../Include/VOClass.h"
#include "../Include/Logger.h"
#include "../Include/Utils.h"
#include "../Include/Constants.h"
#include <cmath>
#include <fstream>

VOClass::VOClass(void){
    /* width = number of cols (x)
     * height = number of rows (y)
     * KITTI dataset specification
    */
    frameW = 1241; 
    frameH = 376;
    /* live trajectory plot window
    */
    liveWindowR = 800;
    liveWindowC = frameW;
    liveWindow = cv::Mat::zeros(liveWindowR, liveWindowC, CV_8UC3);

#if WRITE_ESTIMATED_POSE_FILE
    /* output poses are written to this file; write the starting pose
    */
    estimatedPoseFileHandler.open(estiamtedPoseFilePath);
    Logger.addLog(Logger.levels[INFO], "Write initial tPose to file");
    if(estimatedPoseFileHandler.is_open())
        estimatedPoseFileHandler<<0<<" "<<0<<" "<<0<<std::endl;
    else
        Logger.addLog(Logger.levels[WARNING], "Unable to open estiamtedPoseFile");
    /* close file after writing initial position
    */
    estimatedPoseFileHandler.close();
#endif
}

VOClass::~VOClass(void){
}

/* this function will be called in a loop to read through all sets of images 
 * within  a sequence directory; frameNumber will go from 0 to sizeof(imageDir)-1
*/
bool VOClass::readImagesT1T2(int frameNumber){
    /* construct image file name from frameNumber, and img format (.png in our case)
    */
    const int nameWidth = 6;
    /* read image at t = 1
    */
    std::string imgName = formatStringWidth(frameNumber, nameWidth) + ".png";
    imgLT1 = cv::imread(leftImagesPath + imgName, cv::ImreadModes::IMREAD_GRAYSCALE);
    Logger.addLog(Logger.levels[INFO], "Read image", leftImagesPath + imgName, imgLT1.rows, 
                                                                               imgLT1.cols);
    if(imgLT1.empty()){
        Logger.addLog(Logger.levels[ERROR], "Unable to open imgLT1", "leftImagesPath + imgName");
        assert(false);
    }

    /* read image at t+1
    */
    imgName = formatStringWidth(frameNumber+1, nameWidth) + ".png";
    imgLT2 = cv::imread(leftImagesPath + imgName, cv::ImreadModes::IMREAD_GRAYSCALE);
    Logger.addLog(Logger.levels[INFO], "Read image", leftImagesPath + imgName, imgLT2.rows, 
                                                                               imgLT2.cols);
    if(imgLT2.empty()){
        Logger.addLog(Logger.levels[ERROR], "Unable to open imgLT2", "leftImagesPath + imgName");
        assert(false);
    }

#if SHOW_IMAGE_PAIR
    testShowImagePair(imgLT1, imgLT2, frameNumber);
#endif
    return true;
}

/* calib.txt: Calibration data for the cameras
 * P0/P1 are the  3x4 projection matrices after rectification. Here P0 denotes the 
 * left and P1 denotes the right camera. P2/P3 are left color camera and right color 
 * camera, which we won't be using here
 * 
 * These matrices contain intrinsic information about the camera's focal length and 
 * optical center. Further, they also contain tranformation information which relates 
 * each camera's coordinate frame to the global coordinate frame (in this case that 
 * of the left grayscale camera). - rectified projection matrix
 * 
 * The global frame is the established coordinate frame of the camera's first position
 *  _                        _
 *  | fu    0   cx  -fu * bx |
 *  | 0     fv  cy  0        |
 *  | 0     0   0   1        |
 *  -                        -
*/
bool VOClass::getProjectionMatrix(const std::string calibrationFile){
    /* ifstream to read from file
    */
    std::ifstream file(calibrationFile);
    if(file.is_open()){
        std::string line;
        /* read first line
        */
        std::getline(file, line);
        constructProjectionMatrix(line, projectionCL);
        Logger.addLog(Logger.levels[INFO], "Constructed projectionCL");
        for(int r = 0; r < 3; r++){
            Logger.addLog(Logger.levels[DEBUG], projectionCL.at<double>(r, 0), 
                                                projectionCL.at<double>(r, 1), 
                                                projectionCL.at<double>(r, 2), 
                                                projectionCL.at<double>(r, 3)
            );
        }
        /* extract instrinsic params from projectionCL
        */
        fx = projectionCL.at<double>(0, 0);
        fy = projectionCL.at<double>(1, 1);
        cx = projectionCL.at<double>(0, 2);
        cy = projectionCL.at<double>(1, 2);
        Logger.addLog(Logger.levels[INFO], "Extracted intrinsic params", fx, fy, cx, cy);
        return true;  
    }
    else{
        Logger.addLog(Logger.levels[ERROR], "Unable to open calibration file");
        assert(false);
    }
}

/* poses/XX.txt contains the 4x4 homogeneous matrix flattened out to 12 elements; 
 * r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
 *  _                _
 *  | r11 r12 r13 tx |
 *  | r21 r22 r21 ty |
 *  | r31 r32 r33 tz |
 *  | 0   0   0   1  |
 *  -                - 
 * 
 * The number 12 comes from flattening a 3x4 transformation matrix of the left 
 * camera with respect to the global coordinate frame (first frame of left camera)
*/
bool VOClass::getGroundTruthPath(const std::string groundTruthFile, int& numFrames){
    /* ifstream to read from file
    */
    std::ifstream file(groundTruthFile);
    if(file.is_open()){
        std::string line;
        /* read all lines
        */
        while(std::getline(file, line)){
            constructExtrinsicMatrix(line, extrinsicMat);
            /* extract R, T from extrinsicMat
            */
            cv::Mat R = cv::Mat::zeros(3, 3, CV_64F);
            cv::Mat T = cv::Mat::zeros(3, 1, CV_64F);
            extractRT(R, T, extrinsicMat);
            /* construct ground x, y, z
             * If you are interested in getting where camera0 is located in the 
             * world coordinate frame, you can transform the origin (0,0,0) of the 
             * camera0's local coordinate frame to the world coordinate
             * 
             * ? = R * {0, 0, 0} + T; or the same calculation in homogeneous coords,
             * [R|t] * [0, 0, 0, 1]
             * 
             * This tells where the camera is located in the world coordinate.
             * The resulting (x, y, z) will be in meteres
            */
            groundTruth.push_back(T);
#if 0
            /* display one instance of the extrinsic matrix
            */
            Logger.addLog(Logger.levels[DEBUG], "Constructed extrinsicMat");
            for(int r = 0; r < 4; r++){
                Logger.addLog(Logger.levels[DEBUG], extrinsicMat.at<double>(r, 0), 
                                                    extrinsicMat.at<double>(r, 1), 
                                                    extrinsicMat.at<double>(r, 2), 
                                                    extrinsicMat.at<double>(r, 3)
                );
            }
            /* display one instance of R
            */
            Logger.addLog(Logger.levels[DEBUG], "Extracted R from extrinsicMat");
            for(int r = 0; r < 3; r++){
                Logger.addLog(Logger.levels[DEBUG], R.at<double>(r, 0), 
                                                    R.at<double>(r, 1), 
                                                    R.at<double>(r, 2)
                );
            }
            /* display one instance of T
            */
            Logger.addLog(Logger.levels[DEBUG], "Extracted T from extrinsicMat");
            for(int r = 0; r < 3; r++){
                Logger.addLog(Logger.levels[DEBUG], T.at<double>(r, 0)
                );
            }
            /* display ground x, y, z
            */
            Logger.addLog(Logger.levels[DEBUG], "Computed groundTruth");
            Logger.addLog(Logger.levels[DEBUG], T.at<double>(0, 0), 
                                                T.at<double>(1, 0), 
                                                T.at<double>(2, 0));
#endif
        }
#if SHOW_GROUND_TRUTH_TRAJECTORY
        testShowGroundTruthTrajectory();
#endif
        numFrames = groundTruth.size();
        Logger.addLog(Logger.levels[INFO], "Constructed ground truth trajectory", numFrames);
        return true;
    }
    else{
        Logger.addLog(Logger.levels[ERROR], "Unable to open ground truth file");
        assert(false);
    }
}

/* estimate motion using matched feature points between LT1 and LT2
*/
cv::Mat VOClass::estimateMotion(std::vector<cv::Point2f> featurePointsT1, 
                                std::vector<cv::Point2f> featurePointsT2,
                                int frameNumber){
    /* Compute essential matrix E, using the formula X'(transpose) * E * X = 0
     * The equation basically relates the point 𝑋 and the same point, but described 
     * in the other camera frame ( 𝑋′)
    */
    /* 
     * NOTE: focal length and principal point (cx, cy) are used to construct the camera 
     * intrinsic matrix inside the function. Note that this function assumes that points1 
     * and points2 are feature points from cameras with the same camera intrinsic matrix. 
     * And the function uses K to transform image coordinates to normalized image coords 
     * (by multiplying with the inverse of K)
     * 
     * RANSAC
     * If all of our point correspondences were perfect, then we would have need only 5 
     * feature correspondences between two successive frames to estimate motion accurately. 
     * However, the feature tracking algorithms are not perfect, and therefore we have 
     * several erroneous correspondence. A standard technique of handling outliers when 
     * doing model estimation is RANSAC. It is an iterative algorithm. At every iteration, 
     * it randomly samples five points from out set of correspondences, estimates the 
     * Essential Matrix, and then checks if the other points are inliers when using this 
     * essential matrix. The algorithm terminates after a fixed number of iterations, and 
     * the Essential matrix with which the maximum number of points agree, is used.
     */
    /* NOTE: first parameter is features from imgLT2 and second parameter is features
     * from imgLT1
    */
    cv::Mat E = cv::findEssentialMat(featurePointsT2, featurePointsT1, 
                                     fx, cv::Point2d(cx, cy), cv::RANSAC);
    Logger.addLog(Logger.levels[INFO], "Computed essential matrix", E.rows, E.cols);
#if 0
    for(int r = 0; r < 3; r++){
        Logger.addLog(Logger.levels[DEBUG], E.at<double>(r, 0), 
                                            E.at<double>(r, 1), 
                                            E.at<double>(r, 2));
    }
#endif
    /* outputs R and t
    */
    cv::Mat Rmat;
    cv::Mat tvec;
    /* We know that X'(transpose) * E * X = 0; and E = [t]R; the next step extracts R
     * and t from the essential matrix E, by computing SVD of E
     * NOTE: first parameter is features from imgLT2 and second parameter is features
     * from imgLT1
    */
    cv::recoverPose(E, featurePointsT2, featurePointsT1, Rmat, tvec, fx, cv::Point2d(cx, cy));
    Logger.addLog(Logger.levels[INFO], "Computed SVD of E");
#if 0
    Logger.addLog(Logger.levels[DEBUG], "Rmat", Rmat.rows, Rmat.cols);
    for(int r = 0; r < 3; r++)
        Logger.addLog(Logger.levels[DEBUG], Rmat.at<double>(r, 0), 
                                            Rmat.at<double>(r, 1),
                                            Rmat.at<double>(r, 2));
    Logger.addLog(Logger.levels[DEBUG], "tvec", tvec.rows, tvec.cols);
    for(int r = 0; r < 3; r++)
        Logger.addLog(Logger.levels[DEBUG], tvec.at<double>(r, 0));
#endif

    /* NOTE: Essential matrix is only defined up to scale.  This makes sense, as there 
     * is an ambiguity of size and shape if your only information is point correspondences.
     * The translation vector you get from decomposing E is a unit vector. You cannot 
     * get the actual distance between the cameras in world units without more information, 
     * such as a reference object of known size in the scene or data from another sensor.
     * 
     * For example, if you have two images of a ball and many point correspondences, taken 
     * with calibrated cameras with unknown positions and poses: then is it a normal football 
     * or a mountain-sized ball sculpture? You can infer the camera's relative rotation; 
     * one is in front of the ball (denote this camera as [I | 0] ) and the other is on 
     * the side of the ball. 
     * 
     * You also know in which direction the camera traveled (unit vector of actual translation), 
     * but not how far. To get the real value, you need to multiply with scale which we 
     * will extract from our ground truth data. Normally, you would get the scale factor 
     * from other sensors like an IMU
    */
    double scale = getScaleFactor(frameNumber);
    /* check scale value; has to be > 0.1 to be considered valid, and check if the dominant
     * motion is forward. The entire visual odometry algorithm makes the assumption that 
     * most of the points in its environment are rigid. However, if we are in a scenario 
     * where the vehicle is at a stand still, and a buss passes by (on a road intersection, 
     * for example), it would lead the algorithm to believe that the car has moved sideways, 
     * which is physically impossible. As a result, if we ever find the translation is 
     * dominant in a direction other than forward, we simply ignore that motion.
    */
    if(scale > 0.1 && tvec.at<double>(2, 0) > tvec.at<double>(0, 0) &&
                      tvec.at<double>(2, 0) > tvec.at<double>(1, 0)){
        /* integration/summing up
        */
        tPose = tPose + scale * (RPose * tvec);
        RPose = Rmat * RPose;
        Logger.addLog(Logger.levels[INFO], "Computed RPose and tPose");
#if 0
        Logger.addLog(Logger.levels[DEBUG], "Integrated RPose");
        for(int r = 0; r < 3; r++)
            Logger.addLog(Logger.levels[DEBUG], RPose.at<double>(r, 0), 
                                                RPose.at<double>(r, 1),
                                                RPose.at<double>(r, 2));
        Logger.addLog(Logger.levels[DEBUG], "Integrated tPose");
        for(int r = 0; r < 3; r++)
            Logger.addLog(Logger.levels[DEBUG], tPose.at<double>(r, 0));
#endif
    }
    /* ignore motion estimation at this time step; use previous values
    */
    else
        Logger.addLog(Logger.levels[WARNING], "Scale value too low, OR dominant motion error");

    /* output translation
    */
    cv::Mat tCurr = cv::Mat::zeros(3, 1, CV_64F);
    for(int r = 0; r < 3; r++)
        tCurr.at<double>(r, 0) = tPose.at<double>(r, 0);

#if WRITE_ESTIMATED_POSE_FILE
    estimatedPoseFileHandler.open(estiamtedPoseFilePath, std::fstream::out | 
                                                         std::fstream:: app | 
                                                         std::fstream::ate);
    Logger.addLog(Logger.levels[INFO], "Write tPose to file");
    if(estimatedPoseFileHandler.is_open())
        estimatedPoseFileHandler<<tCurr.at<double>(0, 0)<<" "
                                <<tCurr.at<double>(1, 0)<<" "
                                <<tCurr.at<double>(2, 0)<<std::endl;
    else
        Logger.addLog(Logger.levels[WARNING], "Unable to open estiamtedPoseFile");
    /* close file after writing initial position
    */
    estimatedPoseFileHandler.close();
#endif

    return tCurr;
}

/* compute rmse between groud truth and estimated trajectory
*/
float VOClass::computeErrorInPoseEstimation(std::vector<cv::Mat> estimatedTrajectory){
    Logger.addLog(Logger.levels[INFO], "Estimated trajectory vector size", estimatedTrajectory.size());
    Logger.addLog(Logger.levels[INFO], "Ground truth vector size", groundTruth.size());

    float error = 0;
    for(int i = 0; i < estimatedTrajectory.size(); i++){
#if 1
        Logger.addLog(Logger.levels[DEBUG], "Calculated: ", estimatedTrajectory[i].at<double>(0, 0),
                                                            estimatedTrajectory[i].at<double>(1, 0),
                                                            estimatedTrajectory[i].at<double>(2, 0), 
                                            " Truth: ",     groundTruth[i].at<double>(0, 0), 
                                                            groundTruth[i].at<double>(1, 0),
                                                            groundTruth[i].at<double>(2, 0));
#endif
        error += pow(groundTruth[i].at<double>(0, 0) - estimatedTrajectory[i].at<double>(0, 0), 2) +
                 pow(groundTruth[i].at<double>(1, 0) - estimatedTrajectory[i].at<double>(1, 0), 2) +
                 pow(groundTruth[i].at<double>(2, 0) - estimatedTrajectory[i].at<double>(2, 0), 2);
    }
    /* mean
    */
    if(estimatedTrajectory.size() != 0)
        error = sqrt(error/estimatedTrajectory.size());
    else
        error = -1;
    return error;
}