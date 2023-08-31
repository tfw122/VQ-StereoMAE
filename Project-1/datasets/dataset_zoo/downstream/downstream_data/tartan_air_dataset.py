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

@registry.register_datamodule("tartan_air_easy_downstream")
class TartanAirEasy(StereoDataset):
    def __init__(self, config, aug_params=None, root='../data/tartan_air_extracted', split='train', keywords=[]):
        super().__init__(config, aug_params, reader=frame_utils.readDispTartanAir)
        assert os.path.exists(root)
        self.config = config
        self.dataset_config = self.config.dataset_config

        self.split_ratio = self.dataset_config.db_split_ratio
        # <ROOT DIR>/tartan_air_extracted/abandonedfactory/abandonedfactory/Easy/P000/image_left/*.png
        image1_list= sorted(glob('{}/**/**/Easy/**/image_left/*.png'.format(root)))
        image2_list= sorted(glob('{}/**/**/Easy/**/image_right/*.png'.format(root)))
        disp_list = sorted(glob('{}/**/**/Easy/**/depth_left/*.npy'.format(root)))
        
        #with open(os.path.join(root, 'tartanair_filenames.txt'), 'r') as f:
        #    filenames = sorted(list(filter(lambda s: 'seasonsforest_winter/Easy' not in s, f.read().splitlines())))
        #    for kw in keywords:
        #        filenames = sorted(list(filter(lambda s: kw in s.lower(), filenames)))

        #image1_list = [osp.join(root, e) for e in filenames]
        #image2_list = [osp.join(root, e.replace('_left', '_right')) for e in filenames]
        #disp_list = [osp.join(root, e.replace('image_left', 'depth_left').replace('left.png', 'left_depth.npy')) for e in filenames]

        if split=='train':
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
        else:
            self.image_list = []
            self.disparity_list = []
        
        logging.info(f"Added {len(self.disparity_list)} from tartan air Easy")


@registry.register_datamodule("tartan_air_hard_downstream")
class TartanAirHard(StereoDataset):
    def __init__(self, config, aug_params=None, root='../data/stereo_data/tartan_air_extracted', split='train', keywords=[]):
        super().__init__(config, aug_params, reader=frame_utils.readDispTartanAir)
        assert os.path.exists(root)
        self.config = config
        self.dataset_config = self.config.dataset_config

        self.split_ratio = self.dataset_config.db_split_ratio
        # <ROOT DIR>/tartan_air_extracted/abandonedfactory/abandonedfactory/Easy/P000/image_left/*.png
        image1_list= sorted(glob('{}/**/**/Hard/**/image_left/*.png'.format(root)))
        image2_list= sorted(glob('{}/**/**/Hard/**/image_right/*.png'.format(root)))
        disp_list = sorted(glob('{}/**/**/Hard/**/depth_left/*.npy'.format(root)))
        
        #with open(os.path.join(root, 'tartanair_filenames.txt'), 'r') as f:
        #    filenames = sorted(list(filter(lambda s: 'seasonsforest_winter/Easy' not in s, f.read().splitlines())))
        #    for kw in keywords:
        #        filenames = sorted(list(filter(lambda s: kw in s.lower(), filenames)))

        #image1_list = [osp.join(root, e) for e in filenames]
        #image2_list = [osp.join(root, e.replace('_left', '_right')) for e in filenames]
        #disp_list = [osp.join(root, e.replace('image_left', 'depth_left').replace('left.png', 'left_depth.npy')) for e in filenames]

        if split=='train':
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
        else:
            self.image_list = []
            self.disparity_list = []
        
        logging.info(f"Added {len(self.disparity_list)} from tartan air Hard")