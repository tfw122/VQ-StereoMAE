import torch
from torch.utils.data.dataset import Dataset
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
import numpy as np 
import random
import math
import logging
import os
import re
from datasets.base_dataset import BaseDataset
from imports.registry import registry
from omegaconf import OmegaConf
from pathlib import Path
from glob import glob
import copy 
import os.path as osp

import modules.raft.frame_utils
from modules.raft.augmentor import FlowAugmentor, SparseFlowAugmentor
from datasets.dataset_zoo.downstream.base_stereo_dataset import *  

@registry.register_datamodule("eth3d_downstream")
class ETH3D(StereoDataset):
    def __init__(self, config, aug_params=None, root='../data/eth3d', split='training'):
        super(ETH3D, self).__init__(config, aug_params, sparse=True)
        if split=='train':
            split_path = "training"
        else:
            split_path = split

        image1_list = sorted( glob(osp.join(root, f'two_view_{split_path}/*/im0.png')) )
        image2_list = sorted( glob(osp.join(root, f'two_view_{split_path}/*/im1.png')) )
        disp_list = sorted( glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm')) ) if split_path == 'training' else [osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')]*len(image1_list)

        if split=='train':
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
        
        else:
            self.image_list = []
            self.disparity_list= []

        logging.info(f"Added {len(self.disparity_list)} from eth3d")