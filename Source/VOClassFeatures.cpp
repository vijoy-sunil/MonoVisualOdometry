#include "../Include/VOClass.h"
#include "../Include/Logger.h"
#include "../Include/Constants.h"

/* FAST feature detection for detecting corners
*/
std::vector<cv::Point2f> VOClass::getFeaturesFAST(cv::Mat img){
    /* The keypoint is characterized by the 2D position, scale (proportional 
     * to the diameter of the neighborhood that needs to be taken into account), 
     * orientation and some other parameters. 
     * 
     * The keypoint neighborhood is then analyzed by another algorithm that builds 
     * a descriptor (usually represented as a feature vector)
    */
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::Point2f> featurePoints;
    /* theshold: threshold on difference between intensity of the central pixel 
     * and pixels of a circle around this pixel. 
     * pixel p is a corner if there exists a set of n contiguous pixels in the 
     * circle (of 16 pixels) which are all brighter than Ip+t, or all darker than 
     * Ip−t.
     * 
     * nonmaxSuppression: algorithm faces issues when there are adjacent keypoints,
     * so a score matrix is computed and the one with the lower value is discarded
     * https://docs.opencv.org/4.x/df/d0c/tutorial_py_fast.html
    */
    int threshold = 20;
    bool nonmaxSuppression = true;
    cv::FAST(img, keypoints, threshold, nonmaxSuppression);  
    /* This method converts vector of keypoints to vector of points
    */
    cv::KeyPoint::convert(keypoints, featurePoints);
    Logger.addLog(Logger.levels[INFO], "Computed feature vector", featurePoints.size());

#if SHOW_ALL_FAST_FEATURES
    testShowDetectedFeatures(img, featurePoints);
#endif
    return featurePoints;
}

/* KLT feature matcher based on Sparse Optical Flow
 * NOTE: There are 2 types of optical flow. Dense and sparse. Dense finds 
 * flow for all the pixels while sparse finds flow for the selected points.
*/
std::vector<cv::Point2f> VOClass::matchFeatureKLT(std::vector<cv::Point2f> &featurePointsLT1){
    /* create termination criteria for optical flow calculation
     * The first argument of this function tells the algorithm that we want 
     * to terminate either after some number of iterations or when the 
     * convergence metric reaches some small value (respectively). The next 
     * two arguments set the values at which one, the other, or both of these 
     * criteria should terminate the algorithm.
     * 
     * The reason we have both options is so we can  stop when either limit is 
     * reached.
     * 
     * Here, the criteria specifies the termination criteria of the iterative 
     * search algorithm (after the specified maximum number of iterations maxCount 
     * or when the search window moves by less than epsilon) in the pyramid.
    */
    const int maxCount = 50;
    const float epsilon = 0.03;
    cv::TermCriteria termCrit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 
                                                 maxCount, epsilon);
    
    /* matched feattures in LT2
    */
    std::vector<cv::Point2f> featurePointsLT2;
    /* output status vector (of unsigned chars); each element of the vector is set
     * to 1 if the flow for the corresponding features has been found, otherwise, 
     * it is set to 0. We uses these to remove unmatched features from the vector
    */
    std::vector<unsigned char> status;
    /* err:  output vector of errors; each element of the vector is set to an error 
     * for the corresponding feature. Optical flow basically works by matching a patch, 
     * around each input point, from the input image to the second image. The parameter 
     * err allows you to retrieve the matching error (e.g. you may think of that as the 
     * correlation error) for each input point.
     * 
     * winSize: size of the search window at each pyramid level.
     * 
     * pyramidLevels: if set to 0, pyramids are not used (single level), if set to 1, 
     * two levels are used, and so on.
    */
    std::vector<float> err;                    
    cv::Size winSize = cv::Size(15,15); 
    const int pyramidLevels = 3;

    cv::calcOpticalFlowPyrLK(imgLT1, imgLT2, featurePointsLT1, featurePointsLT2, 
                             status, err, winSize, pyramidLevels, termCrit);

    Logger.addLog(Logger.levels[INFO], "Feature matching complete");
    Logger.addLog(Logger.levels[INFO], "Feature vector sizes", 
    featurePointsLT1.size(), featurePointsLT2.size());

    Logger.addLog(Logger.levels[INFO], "Status vector sizes", status.size());
    Logger.addLog(Logger.levels[INFO], "Status vector valid points", countValidMatches(status));   

#if SHOW_FEATURE_MATCHING_OPTICAL_FLOW
    testShowFeatureMatchingOpticalFlow(imgLT1, featurePointsLT1, featurePointsLT2, status);
#endif
    /* update status vector for invalid feature points; calculated point 
     * (x,y) would be out of bounds
    */
    markInvalidFeaturesBounds(featurePointsLT2, status);

    Logger.addLog(Logger.levels[INFO], "Status vector valid points", "Bounds filter", 
    countValidMatches(status));
 
#if SHOW_FEATURE_MATCHING_OPTICAL_FLOW_BOUNDS_FILTER
    testShowFeatureMatchingOpticalFlow(imgLT1, featurePointsLT1, featurePointsLT2, status);
#endif 
    /* extract common features
    */
    std::vector<cv::Point2f> fLT1, fLT2;
    for(int i = 0; i < status.size(); i++){
        if(status[i] == 1){
            fLT1.push_back(featurePointsLT1[i]);
            fLT2.push_back(featurePointsLT2[i]);
        }
    }
    int numCommonFeatures = fLT1.size();
    Logger.addLog(Logger.levels[INFO], "Extracted common features", numCommonFeatures);

#if SHOW_FEATURE_MATCHING_CONNECT
    testShowConnectMatchedFeatures(imgLT1, fLT1, imgLT2, fLT2);
#endif

#if 0
    for(int i = 0; i < fLT1.size(); i++)
        Logger.addLog(Logger.levels[DEBUG], fLT1[i].x, fLT1[i].y, flT2[i].x, flT2[i].y);
#endif

#if SHOW_ALL_FAST_FEATURES_STABLE
    testShowDetectedFeatures(imgLT1, fLT1);
    testShowDetectedFeatures(imgLT2, fLT2);
#endif
    /* return the final output after removing invalid features
     * the matching features between LT1 and LT2 are fLT1 and fLT2
    */
    featurePointsLT1 = fLT1;
    return fLT2;
}
