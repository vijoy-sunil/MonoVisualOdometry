#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>

/* When set, the application reads pose output from previous run and
 * plots it agains ground truth
*/
#define READ_ESTIMATED_POSE_FILE                            0
/* Debug macros
*/
#define LIMITED_FRAMES_TEST_MODE                            1
#define SHOW_IMAGE_PAIR                                     0
#define SHOW_GROUND_TRUTH_TRAJECTORY                        0
#define SHOW_ALL_FAST_FEATURES                              0
#define SHOW_FEATURE_MATCHING_OPTICAL_FLOW                  0
#define SHOW_FEATURE_MATCHING_OPTICAL_FLOW_BOUNDS_FILTER    0
#define SHOW_FEATURE_MATCHING_CONNECT                       1
#define SHOW_ALL_FAST_FEATURES_STABLE                       0
#define SHOW_GROUND_TRUTH_AND_ESTIMATED_TRAJECTORY          0
/* This is done so that it doesn't clear contents of saved output pose
 * file
*/
#if READ_ESTIMATED_POSE_FILE
    #define WRITE_ESTIMATED_POSE_FILE                       0
#else
    #define WRITE_ESTIMATED_POSE_FILE                       1   
#endif

/* choose the set of images to use in the KITTI dataset; 00 to 10
 * sets of data
*/
const std::string sequenceID = "00";
/* limited frame mode
*/
const int limitedFramesCount = 2;
/* file names
*/
const std::string calibrationFile= "calib.txt";
const std::string dumpFile = "log.txt";
const std::string estiamtedPoseFile = "outputPoses.txt";
/* paths
*/
const std::string datasetPath = "../../Data/sequences/";
const std::string posePath = "../../Data/poses/";
const std::string leftImagesDir = "image_0/";
const std::string dataPath = "../Log/";

const std::string calibrationFilePath = datasetPath +  sequenceID + "/" + calibrationFile;
const std::string groundTruthFilePath = posePath + sequenceID + ".txt";
const std::string leftImagesPath = datasetPath + sequenceID + "/" + leftImagesDir;
const std::string dumpFilePath = dataPath + dumpFile;
const std::string estiamtedPoseFilePath = dataPath + estiamtedPoseFile;
#endif /* CONSTANTS_H
*/
