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


@registry.register_datamodule("kitti_downstream")
class KITTI(StereoDataset):
    def __init__(self, config, aug_params=None, root='../data/kitti_data', split='train', image_set='training'):
        super(KITTI, self).__init__(config, aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)
        self.config = config
        self.dataset_config = self.config.dataset_config
        self.root_dir = '{}'.format(root)

        image1_list = sorted(glob(os.path.join(root, image_set, 'image_2/*_10.png')))
        image2_list = sorted(glob(os.path.join(root, image_set, 'image_3/*_10.png')))
        disp_list = sorted(glob(os.path.join(root, 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else [osp.join(root, 'training/disp_occ_0/000085_10.png')]*len(image1_list)

        
        if split=='train':
            for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
        else:
            self.image_list = [ ]
            self.disparity_list = [ ]

        logging.info(f"Added {len(self.disparity_list)} from kitti stereo 2015")