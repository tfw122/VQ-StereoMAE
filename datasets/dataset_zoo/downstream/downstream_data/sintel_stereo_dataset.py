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



@registry.register_datamodule("sintel_stereo_downstream")
class SintelStereo(StereoDataset):
    def __init__(self, config, aug_params=None, root='../data/sintel_stereo', split='train'):
        super().__init__(config, aug_params, sparse=True, reader=frame_utils.readDispSintelStereo)
        self.config = config
        self.dataset_config = self.config.dataset_config

        image1_list = sorted( glob(osp.join(root, 'training/*_left/*/frame_*.png')) )
        image2_list = sorted( glob(osp.join(root, 'training/*_right/*/frame_*.png')) )
        disp_list = sorted( glob(osp.join(root, 'training/disparities/*/frame_*.png')) ) * 2

        if split=='train':
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                assert img1.split('/')[-2:] == disp.split('/')[-2:]
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
        else:
            self.image_list = []
            self.disparity_list = []
        
        logging.info(f"Added {len(self.disparity_list)} from sintel stereo")