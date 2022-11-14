import os
import numpy as np
import cv2 as cv
import cv2
import imageio
from matplotlib import pyplot as plt
import math
import imageio
imageio.plugins.freeimage.download()
import detectors





def OF_with_Lucas_Kanade(img1, img2,kp1, des1, kp2, des2, matches):
    # TODO: rejection implementation
    
    #bf = cv2.BFMatcher()
    #matches = bf.knnMatch(des1, des2, k=2)
    
    #print(len(matches))
    
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.02))

 
    ## Ratio test

    #matchesMask = [[0, 0] for i in range(len(matches))]

    tracks_pt1 = []
    tracks_pt2 = []
    for i, m1 in enumerate(matches):
    
        #matchesMask[i] = [1]
        
        pt1 = kp1[m1.queryIdx].pt  # trainIdx    is the index of the corresponding key point after matching, the inde of the matching key point of the first loaded image
        pt2 = kp2[m1.trainIdx].pt  # queryIdx  is the index of the corresponding key point after matching, and index of the matching key point of the second loaded image
        
        #print("the num of match:",i)
        #print("pos in left:",pt1,"pos in rgiht:" ,pt2)
        
        tracks_pt1.append([(pt1[0], pt1[1])])
        tracks_pt2.append([(pt2[0], pt2[1])])

    p0 = np.float32([tr[-1] for tr in tracks_pt1]).reshape(-1,1,2)
    p2 = np.float32([tr[-1] for tr in tracks_pt2]).reshape(-1,1,2)
    p1, st, err = cv.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)
    d = abs(p2-p1).reshape(-1,2).max(-1)
    good = d < 1
    matchesMask1 = [ 0 for i in range(len(matches))]
    for i, (tr, (x, y), flag) in enumerate(zip(tracks_pt1, p1.reshape(-1, 2), good)):
        
       
        if not flag:
                continue
        matchesMask1[i] = 1
        
        #print("p0:",i,p0[i])
        #print("p1:",i,p1[i])
        #print("p2:",i,p2[i])
        
    
    
    return matchesMask1

