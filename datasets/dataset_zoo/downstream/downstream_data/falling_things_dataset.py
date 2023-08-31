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

@registry.register_datamodule("falling_things_single_downstream")
class FallingThingsSingle(StereoDataset):
    def __init__(self, config, aug_params=None, root='../data/falling_things', split='train'):
        super().__init__(config, aug_params, reader=frame_utils.readDispFallingThings)
        assert os.path.exists(root)
        self.config = config
        self.dataset_config = config.dataset_config
        
        self.root_dir = '{}/fat/single'.format(root)
        self.split_ratio = self.dataset_config.db_split_ratio

        self.image1_list= sorted(glob('{}/**/**/*.left.jpg'.format(self.root_dir)))
        self.image2_list= sorted(glob('{}/**/**/*.right.jpg'.format(self.root_dir)))
        self.disp_list = sorted(glob('{}/**/**/*.left.depth.png'.format(self.root_dir)))

        #if split =='train':
        #    self.image1_list= self.image1_list[0: int(self.split_ratio*len(self.image1_list))]
        #    self.image2_list= self.image2_list[0: int(self.split_ratio*len(self.image2_list))]
        #    self.disp_list = self.disp_list[0: int(self.split_ratio*len(self.disp_list))]

        #elif split=='val' or split=='test':
        #    self.image1_list= self.image1_list[int(self.split_ratio*len(self.image1_list)): len(self.image1_list)]
        #    self.image2_list= self.image2_list[int(self.split_ratio*len(self.image2_list)): len(self.image2_list)]
        #    self.disp_list= self.disp_list[int(self.split_ratio*len(self.disp_list)): len(self.disp_list)]


        #image1_list = [osp.join(root, e) for e in filenames]
        #image2_list = [osp.join(root, e.replace('left.jpg', 'right.jpg')) for e in filenames]
        #disp_list = [osp.join(root, e.replace('left.jpg', 'left.depth.png')) for e in filenames]

        if split=='train':
            for img1, img2, disp in zip(self.image1_list, self.image2_list, self.disp_list):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
        
        else:
            self.image_list = []
            self.disparity_list= []

        logging.info(f"Added {len(self.disparity_list)} from falling things single")

@registry.register_datamodule("falling_things_mixed_downstream")
class FallingThingsMixed(StereoDataset):
    def __init__(self, config, aug_params=None, root='../data/falling_things', split='train'):
        super().__init__(config, aug_params, reader=frame_utils.readDispFallingThings)
        assert os.path.exists(root)
        self.config = config
        self.dataset_config = self.config.dataset_config
        
        self.root_dir = '{}/fat/mixed'.format(root)
        self.split_ratio = self.dataset_config.db_split_ratio

        self.image1_list= sorted(glob('{}/**/*.left.jpg'.format(self.root_dir)))
        self.image2_list= sorted(glob('{}/**/*.right.jpg'.format(self.root_dir)))
        self.disp_list = sorted(glob('{}/**/*.left.depth.png'.format(self.root_dir)))

        #if split =='train':
        #    self.image1_list= self.image1_list[0: int(self.split_ratio*len(self.image1_list))]
        #    self.image2_list= self.image2_list[0: int(self.split_ratio*len(self.image2_list))]
        #    self.disp_list = self.disp_list[0: int(self.split_ratio*len(self.disp_list))]

        #elif split=='val' or split=='test':
        #    self.image1_list= self.image1_list[int(self.split_ratio*len(self.image1_list)): len(self.image1_list)]
        #    self.image2_list= self.image2_list[int(self.split_ratio*len(self.image2_list)): len(self.image2_list)]
        #    self.disp_list= self.disp_list[int(self.split_ratio*len(self.disp_list)): len(self.disp_list)]


        #image1_list = [osp.join(root, e) for e in filenames]
        #image2_list = [osp.join(root, e.replace('left.jpg', 'right.jpg')) for e in filenames]
        #disp_list = [osp.join(root, e.replace('left.jpg', 'left.depth.png')) for e in filenames]

        if split=='train':
            for img1, img2, disp in zip(self.image1_list, self.image2_list, self.disp_list):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
        
        else:
            self.image_list = []
            self.disparity_list= []

        logging.info(f"Added {len(self.disparity_list)} from falling things mixed")