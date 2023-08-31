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


@registry.register_datamodule("sceneflow_clean_downstream")
class SceneFlowDatasetsClean(StereoDataset):
    """
    flying things 3d no. of samples: 22390
    monkaa no. of samples: 8664
    driving no. of samples: 4400
    """
    def __init__(self, config, aug_params=None, root='../data/scene_flow', split='train', dstype='frames_cleanpass', things_test=False):
        super(SceneFlowDatasetsClean, self).__init__(config, aug_params)
        self.config = config
        self.dataset_config = self.config.dataset_config
        self.root = root
        self.dstype = dstype
        self.split= split.upper()
        if self.split=="VAL":
            self.split_path = "TEST"
        else:
            self.split_path = self.split

        # ONLY FLYINGTHINGS 3D TRAINING SUBSET IS USED FOR VALIDATION AND TESTING
        if things_test or split=="VAL":
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa()
            self._add_driving()

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'flyingthings3d')
        left_images = sorted( glob(osp.join(root, self.dstype, self.split_path, '*/*/left/*.png')) )
        right_images = [ im.replace('left', 'right') for im in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        # Choose a random subset of 400 images for validation only; for testing; you select all testing images from split_path
        state = np.random.get_state()
        np.random.seed(1000)
        val_idxs = set(np.random.permutation(len(left_images))[:400])
        np.random.set_state(state)

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            if (self.split== 'VAL' and idx in val_idxs) or split == 'TRAIN':
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'monkaa')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]
        
        if self.split=='TRAIN':
            for img1, img2, disp in zip(left_images, right_images, disparity_images):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
            #logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")
        else:
            self.image_list = []
            self.disparity_list=[]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")

    def _add_driving(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'driving')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/*/*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        if self.split=='TRAIN':
            for img1, img2, disp in zip(left_images, right_images, disparity_images):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
            #logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")
        else:
            self.image_list = []
            self.disparity_list=[]
        
        logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")


@registry.register_datamodule("sceneflow_final_downstream")
class SceneFlowDatasetsFinal(StereoDataset):
    """
    flying things 3d no. of samples: 22390
    monkaa no. of samples: 8664
    driving no. of samples: 4400
    """
    def __init__(self, config, aug_params=None, root='../data/scene_flow', split='train', dstype='frames_finalpass', things_test=False):
        super(SceneFlowDatasetsFinal, self).__init__(config, aug_params)
        self.config = config
        self.dataset_config = self.config.dataset_config
        self.root = root
        self.dstype = dstype
        self.split= split.upper()
        if self.split=="VAL":
            self.split_path = "TEST"
        else:
            self.split_path = self.split

        # ONLY FLYINGTHINGS 3D TRAINING SUBSET IS USED FOR VALIDATION AND TESTING
        if things_test or split=="VAL":
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa()
            self._add_driving()

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'flyingthings3d')
        left_images = sorted( glob(osp.join(root, self.dstype, self.split_path, '*/*/left/*.png')) )
        right_images = [ im.replace('left', 'right') for im in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        # Choose a random subset of 400 images for validation only; for testing; you select all testing images from split_path
        state = np.random.get_state()
        np.random.seed(1000)
        val_idxs = set(np.random.permutation(len(left_images))[:400])
        np.random.set_state(state)

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            if (self.split== 'VAL' and idx in val_idxs) or split == 'TRAIN':
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'monkaa')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]
        
        if self.split=='TRAIN':
            for img1, img2, disp in zip(left_images, right_images, disparity_images):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
            #logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")
        else:
            self.image_list = []
            self.disparity_list=[]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")

    def _add_driving(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'driving')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/*/*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        if self.split=='TRAIN':
            for img1, img2, disp in zip(left_images, right_images, disparity_images):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
            #logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")
        else:
            self.image_list = []
            self.disparity_list=[]
        
        logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")