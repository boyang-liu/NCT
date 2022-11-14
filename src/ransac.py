import numpy as np
import cv2 as cv

def ransac_fundamental_opencv(kp1, kp2, matches, max_epipolar_line_distance = 3, confidence = 0.99, max_iterations = 2000):
    """Uses the OpenCV implementation of Fundamental Matrix computation for outlier rejection"""
    # directly return if number of matches is too small
    if len(matches) < 8:
        return [1 for match in matches]

    # split position data from matches
    positions1 = np.array([kp1[mat.queryIdx].pt for mat in matches])
    positions2 = np.array([kp2[mat.trainIdx].pt for mat in matches])

    # calculate Fundamental Matrix and inlier mask from OpenCV implemenation
    F, mask_array  = cv.findFundamentalMat(positions1, positions2, cv.FM_RANSAC, ransacReprojThreshold=max_epipolar_line_distance, confidence=confidence, maxIters=max_iterations)

    # return matches mask
    return [element[0] for element in mask_array]


def ransac_homography_opencv(kp1, kp2, matches, max_reprojection_error = 3, confidence = 0.995, max_iterations = 2000):
    """Uses the OpenCV implementation of Homography computation for outlier rejection"""
    # directly return if number of matches is too small
    if len(matches) < 4:
        return [1 for match in matches]
    
    # split position data from matches
    positions1 = np.array([kp1[mat.queryIdx].pt for mat in matches])
    positions2 = np.array([kp2[mat.trainIdx].pt for mat in matches])

    # calculate Fundamental Matrix and inlier mask from OpenCV implemenation
    H, mask_array  = cv.findHomography(positions1, positions2, cv.FM_RANSAC, ransacReprojThreshold=max_reprojection_error, confidence=confidence, maxIters=max_iterations)

    # return matches mask
    return [element[0] for element in mask_array]