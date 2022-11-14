from enum import Enum, auto

import numpy as np
import cv2 as cv

class Detectors(Enum):
    AKAZE   = auto() # Hamming
    BRISK   = auto() # Hamming
    KAZE    = auto() # L2
    ORB     = auto() # Hamming
    SIFT    = auto() # L2
    SURF    = auto() # L2

class Matchers(Enum):
    BF      = auto()
    KNN2    = auto()

ORB_PARAMETERS = {
    'nfeatures' : 500, 
    'scaleFactor' : 1.2, 
    'nlevels' : 8, 
    'edgeThreshold' : 31, 
    'firstLevel' : 0,
    'WTA_K' : 2,
    'scoreType' : cv.ORB_HARRIS_SCORE,
    'patchSize' : 31,
    'fastThreshold' : 20
}

SIFT_PARAMETERS = {
    'nfeatures' : 0, 
    'nOctaveLayers' : 3,
    'contrastThreshold' : 0.04,
    'edgeThreshold' : 10,
    'sigma' : 1.6
}

SURF_PARAMETERS = {
    'hessianThreshold' : 100,
    'nOctaves' : 4,
    'nOctaveLayers' : 3,
    'extended' : False,
    'upright' : False
}

KAZE_PARAMETERS = {
    'threshold' : 0.001,
    'nOctaves' : 4,
    'nOctaveLayers' : 4,
    'diffusivity' : cv.KAZE_DIFF_PM_G2,
    'extended' : False,
    'upright' : False
}

AKAZE_PARAMETERS = {
    'threshold' : 0.001,
    'nOctaves' : 4,
    'nOctaveLayers' : 4,
    'diffusivity' : cv.KAZE_DIFF_PM_G2,
    'descriptor_type' : cv.AKAZE_DESCRIPTOR_MLDB,
    'descriptor_size' : 0,
    'descriptor_channels' : 3
}

BRISK_PARAMETERS = {
    'thresh' : 30,
    'octaves' : 3,
    'patternScale' : 1.0
}

def keypoints_and_descriptors(image : np.array, detector_type : Detectors, verbose : bool = False):
    """Computes keypoints and descriptors in a given image with a given method."""
    if detector_type == Detectors.AKAZE:
        detector = cv.AKAZE_create(**AKAZE_PARAMETERS)
    elif detector_type == Detectors.BRISK:
        detector = cv.BRISK_create(**BRISK_PARAMETERS)
    elif detector_type == Detectors.KAZE:
        detector = cv.KAZE_create(**KAZE_PARAMETERS)
    elif detector_type == Detectors.ORB:
        detector = cv.ORB_create(**ORB_PARAMETERS)
    elif detector_type == Detectors.SIFT:
        detector = cv.SIFT_create(**SIFT_PARAMETERS)
    elif detector_type == Detectors.SURF:
        detector = cv.xfeatures2d.SURF_create()

    else:
        raise ValueError(f"Inappropriate value of detector type : detector_type = {detector_type}")

    kp, des = detector.detectAndCompute(image, None)

    if verbose:
        print(f"Number of keypoints: {len(kp)}")

    return kp, des


def sorted_matches(descriptors1, descriptors2, detector_type : Detectors, matcher_type : Matchers, distance_ratio = 0.7, verbose : bool = False):
    """Computes and sorts matches between descriptors chosing the right matching norm according to the given detector type."""
    if detector_type in [Detectors.AKAZE, Detectors.BRISK, Detectors.ORB]:
        norm = cv.NORM_HAMMING
    elif detector_type in [Detectors.KAZE, Detectors.SIFT, Detectors.SURF]:
        norm = cv.NORM_L2
    else:
        raise ValueError(f"Inappropriate value of detector type : detector_type = {detector_type}")

    matcher = cv.BFMatcher(norm)

    if matcher_type == Matchers.BF:
        matches = sorted(matcher.match(descriptors1, descriptors2), key=lambda x: x.distance)
    elif matcher_type == Matchers.KNN2:
        knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
        matches = []
        for match1, match2 in knn_matches:
            if match1.distance < distance_ratio * match2.distance:
                matches.append(match1)
        matches = sorted(matches, key=lambda x: x.distance)
    else:
        raise ValueError(f"Inappropriate value of matcher type : matcher_type = {matcher_type}")

    if verbose:
        print(f"Matching norm: {norm}")
        print(f"Number of matches: {len(matches)}")

    return matches


def keypoints_descriptors_matches(image1 : np.array, image2 : np.array, detector_type : Detectors, matcher_type : Matchers, distance_ratio = 0.7, verbose : bool = False):
    """Returns keypoints, descriptors and sorted matches between two images using a given detector."""
    kp1, des1   = keypoints_and_descriptors(image1, detector_type, verbose=verbose)
    kp2, des2   = keypoints_and_descriptors(image2, detector_type, verbose=verbose)
    matches     = sorted_matches(des1, des2, detector_type, matcher_type, distance_ratio=distance_ratio, verbose=verbose)

    return kp1, des1, kp2, des2, matches