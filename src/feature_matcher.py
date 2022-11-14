import os
from re import X
import numpy as np
import cv2 as cv
import cv2
import imageio
from matplotlib import pyplot as plt
import math
import imageio
imageio.plugins.freeimage.download()

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

import detectors
import ransac
import OF
import raft_files.opticalflow as raft_of
import Superpoint.SuperGluePretrainedNetwork.match_pairs as superpoint

def main():
    # setup global variables
    global orb, bf, PATH_TO_SAMPLE, i, raft_rejector, superpoint_matcher

    PATH_TO_SEQUENCES = r"..\evaluation_sequences"
    SEQUENCE_NAME = "scene_5"

    PATH_TO_SAMPLE = PATH_TO_SEQUENCES + "\\" + SEQUENCE_NAME

    raft_rejector = raft_of.RAFT_Rejector(raft_of.RAFT_Models.THINGS, PATH_TO_SAMPLE + "/translation")
    superpoint_matcher = superpoint.match_pairs()

    # iterate through all images of the scene, calculate ORB features, BF matches, ...
    for i in range(50, 99):
        # open current and next image
        filename1 = PATH_TO_SAMPLE + r"\translation\translation" + str(i).zfill(4) + ".png"
        filename2 = PATH_TO_SAMPLE + r"\translation\translation" + str(i + 1).zfill(4) + ".png"

        img1 = cv.imread(filename1)
        img2 = cv.imread(filename2)

        # calculate features and matches (rejection included)
        kp1, des1, kp2, des2, matches, mask = match(img1, img2)
        
        match_repeatability, match_localization_error = evaluate_matches(kp1, kp2, des1, des2, matches, i)
        
        
        # create output image and show
        output = generate_output_image(img1, kp1, img2, kp2, matches, mask)
        cv.imshow("output", output)
        cv.moveWindow("output", 0, 0)

        # evaluate matches by their distance
        rej_precision, rej_recall, rej_localization_error = evaluate_rejection(kp1, kp2, matches, mask, i)

        # wait for key and quit if q was pressed
        if cv.waitKey() == ord('q'):
            break

    # remove plot png file and close all windows
    if os.path.exists("plot.png"):
        os.remove("plot.png")
    cv.destroyAllWindows()


def match(img1, img2):
    # keypoints, descriptors and matches computation
    # detector_type   = detectors.Detectors.SIFT
    # matcher_type    = detectors.Matchers.BF
    # kp1, des1, kp2, des2, matches = detectors.keypoints_descriptors_matches(img1, img2, detector_type, matcher_type, verbose=True)
    kp1, des1, kp2, des2, matches = superpoint_matcher.match(cv.cvtColor(img1, cv.COLOR_BGR2GRAY), cv.cvtColor(img2, cv.COLOR_BGR2GRAY))
    #print(len(matches))
    # call rejection function current matches
    mask = reject_false_matches(img1, img2,kp1, des1, kp2, des2, matches)

    return kp1, des1, kp2, des2, matches, mask

def return_mask_without_rejection(matches):
    mask = [ 1 for i in range(len(matches))]
    return mask

def reject_false_matches(img1, img2,kp1, des1, kp2, des2, matches):
    mask = ransac.ransac_homography_opencv(kp1, kp2, matches)
    #mask2 = OF.OF_with_Lucas_Kanade(img1, img2,kp1, des1, kp2, des2, matches)
    # mask = return_mask_without_rejection(matches)
    return mask


def generate_output_image(img1, kp1, img2, kp2, matches, mask):
    # split matches into inliers and outliers according to mask
    inliers     = [match for i, match in enumerate(matches) if mask[i]]
    outliers    = [match for i, match in enumerate(matches) if not mask[i]]

    # draw inliers
    output = cv2.drawMatches(img1, kp1, img2, kp2, inliers, None, matchColor = (0, 255,0),
        flags = cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # draw outliers on top
    output = cv2.drawMatches(img1, kp1, img2, kp2, outliers, output, matchColor = (0, 0, 255),
        flags = cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS | cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)

    return output

def evaluate_rejection(kp1, kp2, matches, mask, i, threshold=3):
    # extract pixel positions of matches
    # see:
    # https://stackoverflow.com/questions/30716610/how-to-get-pixel-coordinates-from-feature-matching-in-opencv-python
    inlier_positions    = [(kp1[mat.queryIdx].pt, kp2[mat.trainIdx].pt) for i, mat in enumerate(matches) if mask[i]]
    outlier_positions   = [(kp1[mat.queryIdx].pt, kp2[mat.trainIdx].pt) for i, mat in enumerate(matches) if not mask[i]]
                     
    # read actual coordinates from match positions
    coords_filename1 = PATH_TO_SAMPLE + r"\3Dcoordinates\coords" + str(i).zfill(4) + ".exr"
    coords1 = cv2.imread(coords_filename1, cv2.IMREAD_UNCHANGED)
    
    # dstack ones to the coordinates to make them homogenous
    H, W = coords1.shape[0], coords1.shape[1]
    ones = np.ones((H, W, 1), dtype="float32")
    coords1 = torch.tensor(np.dstack((coords1, ones)))

    # reshape coordinates
    coords1_reshaped = coords1.view(H, W, 4, 1).transpose(1,0)

    # read 2nd projection matrix
    proj_filename2 = PATH_TO_SAMPLE + r"\projection_matrices\proj" + str(i + 1).zfill(4) + ".exr"
    proj2 = torch.Tensor(cv2.imread(proj_filename2, cv2.IMREAD_UNCHANGED))

    # calculate corresponding positions of image2 to image1
    xyz = torch.matmul(proj2, coords1_reshaped).squeeze(dim=-1)
    xy, z = torch.split(xyz, [2,1], dim=-1)
    xy = (xy / z).round()
    xy_numpy = xy.numpy()

    # calculate distances between matched points of img1 and img2
    inlier_distances = [np.linalg.norm(pos2 - xy_numpy[int(pos1[0])][int(pos1[1])]) for pos1, pos2 in inlier_positions]
    outlier_distances = [np.linalg.norm(pos2 - xy_numpy[int(pos1[0])][int(pos1[1])]) for pos1, pos2 in outlier_positions]
    
                     
    # compute evaluation measures
    num_fp = 0
    num_fn = 0
    for dis in inlier_distances:        
        if dis > threshold:
            num_fp = num_fp + 1
    for dis in outlier_distances:       
        if dis <= threshold:
            num_fn = num_fn + 1
    precision   = (len(inlier_distances) - num_fp) / len(inlier_distances)
    recall      = (len(inlier_distances) - num_fp) / (len(inlier_distances) - num_fp + num_fn)
         
    precision   = round(precision,2)
    recall      = round(recall,2)
    
    # find max distance
    max_inlier_distance = np.max(inlier_distances) if len(inlier_distances) > 0 else 0
    max_outlier_distance = np.max(outlier_distances) if len(outlier_distances) > 0 else 0

    # localization error is same as average inlier distance
    localization_error = np.average(inlier_distances)

    # print(f"EVALUATION {i}:")
    # print(f"max_inlier_distance = {max_inlier_distance}")
    # print(f"max_outlier_distance = {max_outlier_distance}")
    
    # show histogram plot of matched points distances
    # work around since pyplot is not good in positioning windows, letting code run after show() and interaction
    # save plot as png from pyplot and open and show via opencv
    fig = plt.figure(0)
    plt.clf()

    # plot inliers
    plt.hist(inlier_distances, color="green", bins=max(5, 1 + int(max_inlier_distance)), range=[0, max(5, 1 + int(max_inlier_distance))])
    # plot outliers
    plt.hist(outlier_distances, color="red", bins=max(5, 1 + int(max_outlier_distance)), range=[0, max(5, 1 + int(max_outlier_distance))])

    print(f"Rejection threshold = {threshold}")
    print(f"Rejection precision = {precision}")
    print(f"Rejection recall = {recall}")
    print(f"Rejection Localization Error = {localization_error}")

    fig.savefig("plot.png")
    fig_img = cv.imread("plot.png")
    cv.imshow("Distance Histogram", fig_img)
    cv.moveWindow("Distance Histogram", 0, 300)
    
    return precision, recall, localization_error
                 
def evaluate_matches(kp1, kp2, des1, des2, matches, i, threshold=3):
                      
    coords_filename1 = PATH_TO_SAMPLE + r"\3Dcoordinates\coords" + str(i).zfill(4) + ".exr"
    coords1 = cv2.imread(coords_filename1, cv2.IMREAD_UNCHANGED)
    
    # dstack ones to the coordinates to make them homogenous
    H, W = coords1.shape[0], coords1.shape[1]
    ones = np.ones((H, W, 1), dtype="float32")
    coords1 = torch.tensor(np.dstack((coords1, ones)))

    # reshape coordinates
    coords1_reshaped = coords1.view(H, W, 4, 1).transpose(1,0)

    # read 2nd projection matrix
    proj_filename2 = PATH_TO_SAMPLE + r"\projection_matrices\proj" + str(i + 1).zfill(4) + ".exr"
    proj2 = torch.Tensor(cv2.imread(proj_filename2, cv2.IMREAD_UNCHANGED))

    # calculate corresponding positions of image2 to image1
    xyz = torch.matmul(proj2, coords1_reshaped).squeeze(dim=-1)
    xy, z = torch.split(xyz, [2,1], dim=-1)
    xy = (xy / z).round()
    xy_numpy = xy.numpy()

    # get distances from ground truth position in second image
    distances   = [np.linalg.norm(kp2[mat.trainIdx].pt - xy_numpy[int(kp1[mat.queryIdx].pt[0])][int(kp1[mat.queryIdx].pt[1])]) for mat in matches]

    # localization error is average distance
    localization_error = np.average(distances)

    # calculate inliers
    inlier_count = sum(distance <= threshold for distance in distances)
    # repeatability is fraction of correctly associated matches
    repeatability = inlier_count / len(distances)
    
    print(f"Matching Threshold = {threshold}")
    print(f"Matching Localization_Error = {localization_error}")
    print(f"Matching Repeatability = {repeatability}")
    return repeatability, localization_error
    
                     
if __name__ == '__main__':
    main()
