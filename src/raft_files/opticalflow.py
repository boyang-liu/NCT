import sys

from torch._C import StringType

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from enum import Enum

main_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, main_dir + "/core")

from raft import RAFT
from utils.utils import InputPadder

class RAFT_Models(Enum):
    CHAIRS  = 'models/raft-chairs.pth'
    KITTI   = 'models/raft-kitti.pth'
    SINTEL  = 'models/raft-sintel.pth'
    SMALL   = 'models/raft-small.pth'
    THINGS  = 'models/raft-things.pth'

DEVICE = 'cuda'

class RAFT_Rejector:

    def __init__(self, model : RAFT_Models, path : StringType, small = False, alternate_corr = False, mixed_precision = False):
        # prepare args
        args_dict = {}
        args_dict['model']              = main_dir + "/" + model.value
        args_dict['path']               = path
        args_dict['small']              = small
        args_dict['alternate_corr']     = alternate_corr
        args_dict['mixed_precision']    = mixed_precision

        args = argparse.Namespace(**args_dict)

        # load model
        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(args.model))

        self.model = self.model.module
        self.model.to(DEVICE)
        self.model.eval()

        # load image paths
        self.images = glob.glob(os.path.join(args.path, '*.png')) + \
                    glob.glob(os.path.join(args.path, '*.jpg'))

    def load_image(self, imfile):
        img = np.array(Image.open(imfile)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(DEVICE)
    
    def compute_flow(self, i):
        with torch.no_grad():
            image1 = self.load_image(self.images[i])
            image2 = self.load_image(self.images[i + 1])

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True) # what is flow_low?
            return flow_up.cpu().numpy()[0].transpose(1, 2, 0)
    
    def reject(self, kp1, kp2, matches, index, threshold = 5):
        flow = self.compute_flow(index)
        mask = [0 for match in matches]

        for i, match in enumerate(matches):
            pos1 = kp1[match.queryIdx].pt
            pos2 = kp2[match.trainIdx].pt

            match_flow = np.array([pos2[0] - pos1[0], pos2[1] - pos1[1]])
            computed_flow = flow[round(pos1[1]), round(pos1[0])]

            delta = match_flow - computed_flow
            if np.linalg.norm(delta) < threshold:
                mask[i] = 1

        return mask
