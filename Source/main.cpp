#include "../Include/VOClass.h"
#include "../Include/Constants.h"
#include "../Include/Logger.h"
#include "../Include/Utils.h"
#include <iostream>

int main(void){
    int numFrames = 0;
    VOClass VO;
    /* read ground truth poses
    */
    VO.getGroundTruthPath(groundTruthFilePath, numFrames);
    /* plot previously saved estimated trajectory
    */
#if READ_ESTIMATED_POSE_FILE
    VO.testShowTrajectoryPairFromFile(estiamtedPoseFilePath);
#else
    /* read from input files
    */
    VO.getProjectionMatrix(calibrationFilePath);
    /* Instead of running through the entire range of frames, run the
     * application for only a limited number of frames
    */
#if LIMITED_FRAMES_TEST_MODE
    numFrames = limitedFramesCount;
#endif
    assert(numFrames != 1);
    /* output trajectory
    */
    std::vector<cv::Mat> estimatedTrajectory;
    /* first element in trajectory has to be (0, 0, 0)
    */
    estimatedTrajectory.push_back(cv::Mat::zeros(3, 1, CV_64F));
    /* main loop
    */
    for(int i = 0; i < numFrames-1; i++){
        /* read image pair at t and t+1
        */
        VO.readImagesT1T2(i);
        /* detect features in imgLT1
        */
        std::vector<cv::Point2f> featurePointsT1 = VO.getFeaturesFAST(VO.imgLT1);
        /* match feature points imgLT1 -> imgLT2
        */
        std::vector<cv::Point2f> featurePointsT2 = VO.matchFeatureKLT(featurePointsT1);
        /* estimate motion
        */
        estimatedTrajectory.push_back(VO.estimateMotion(featurePointsT1, 
                                                        featurePointsT2, 
                                                        i));
        /* show progress bar
        */
        float progress = (float)(i+1)/(float)(numFrames-1);
        showProgressBar(progress);
    }
    /* compute error between estimated trajectory and ground truth
    */
    float error = VO.computeErrorInPoseEstimation(estimatedTrajectory);
    Logger.addLog(Logger.levels[INFO], "Measured error", error);
    std::cout<<"Measured error: "<<error<<std::endl;

#if SHOW_GROUND_TRUTH_AND_ESTIMATED_TRAJECTORY
    /* plot trajectory
    */
    VO.testShowTrajectoryPair(estimatedTrajectory);
#endif
#endif
    return 0;
}