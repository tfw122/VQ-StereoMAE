import torch
from torch.utils.data.dataset import Dataset
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
import pandas
import numpy as np 
import random
from datasets.base_dataset import BaseDataset
from imports.registry import registry
from omegaconf import OmegaConf
from glob import glob
import itertools 
import logging 

@registry.register_datamodule("sceneflow_cleanpass_mim")
class SceneFlowLoader(Dataset):
    """
    flying things 3d no. of samples: 22390
    monkaa no. of samples: 8664
    driving no. of samples: 300
    """
    def __init__(self, config, split, transform=None):
        self.dataset_config = config.dataset_config
        self.root_dir = '{}/scene_flow'.format(self.dataset_config.root_dir)
        self.split_ratio = self.dataset_config.db_split_ratio
        dstype = "frames_cleanpass"

        self.monkaa_left= sorted(glob('{}/monkaa/{}/**/left/*.png'.format(self.root_dir, dstype)))
        self.driving_left= sorted(glob('{}/driving/{}/**/**/**/left/*.png'.format(self.root_dir, dstype)))
        
        self.monkaa_right= sorted(glob('{}/monkaa/{}/**/right/*.png'.format(self.root_dir, dstype)))
        self.driving_right= sorted(glob('{}/driving/{}/**/**/**/right/*.png'.format(self.root_dir, dstype)))
        
        if split =='train' or split =='val':
            # All training left and right frames; (later split by the split_ratio into train and val)
            self.flying_things_left = sorted(glob('{}/flyingthings3d/{}/TRAIN/**/**/left/*.png'.format(self.root_dir, dstype)))
            # right frames;
            self.flying_things_right = sorted(glob('{}/flyingthings3d/{}/TRAIN/**/**/right/*.png'.format(self.root_dir, dstype)))
            print("sceneflow_data_process")
            
        else:
            # TEST
            # left frames;
            self.flying_things_left = sorted(glob('{}/flyingthings3d/{}/TEST/**/**/left/*.png'.format(self.root_dir, dstype)))
            # right frames;
            self.flying_things_right = sorted(glob('{}/flyingthings3d/{}/TEST/**/**/right/*.png'.format(self.root_dir, dstype)))
            
        self.left_paths = self.flying_things_left + self.monkaa_left + self.driving_left
        self.right_paths = self.flying_things_right + self.monkaa_right + self.driving_right

        # split training files into train and val
        if split == 'train':
            # split a sub sample of train data for val;
            self.left_paths= self.left_paths[0: int(self.split_ratio*len(self.left_paths))]
            self.right_paths= self.right_paths[0: int(self.split_ratio*len(self.right_paths))]
        
        elif split =='val':
            # split a sub sample of train data for val;
            self.left_paths= self.left_paths[int(self.split_ratio*len(self.left_paths)): len(self.left_paths)]
            self.right_paths= self.right_paths[int(self.split_ratio*len(self.right_paths)): len(self.right_paths)]
            
        else:
            print('training data has been split into train and val via the split_ratio: {} and {} for train and val respectively'.format(self.split_ratio, 1-self.split_ratio))

        
        self.transform = transform
        logging.info(f"Added {len(self.flying_things_left)} from Flying Things 3D clean pass")
        logging.info(f"Added {len(self.monkaa_left)} from Monkaa clean pass")
        logging.info(f"Added {len(self.driving_left)} from Driving clean pass")

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx]).convert('RGB')
        right_image = Image.open(self.right_paths[idx]).convert('RGB')
            
        sample = {'left_image': left_image, 'right_image': right_image}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.left_paths)
    

@registry.register_datamodule("sceneflow_finalpass_mim")
class SceneFlowLoader(Dataset):
    """
    flying things 3d no. of samples: 22390
    monkaa no. of samples: 8664
    driving no. of samples: 300
    """
    def __init__(self, config, split, transform=None):
        self.dataset_config = config.dataset_config
        self.root_dir = '{}/sceneflow'.format(self.dataset_config.root_dir)
        self.split_ratio = self.dataset_config.db_split_ratio
        dstype = "frames_finalpass"

        self.monkaa_left= sorted(glob('{}/monkaa/{}/**/left/*.png'.format(self.root_dir, dstype)))
        self.driving_left= sorted(glob('{}/driving/{}/**/**/**/left/*.png'.format(self.root_dir, dstype)))
        
        self.monkaa_right= sorted(glob('{}/monkaa/{}/**/right/*.png'.format(self.root_dir, dstype)))
        self.driving_right= sorted(glob('{}/driving/{}/**/**/**/right/*.png'.format(self.root_dir, dstype)))
        
        if split =='train' or split =='val':
            # All training left and right frames; (later split by the split_ratio into train and val)
            self.flying_things_left = sorted(glob('{}/flyingthings3d/{}/TRAIN/**/**/left/*.png'.format(self.root_dir, dstype)))
            # right frames;
            self.flying_things_right = sorted(glob('{}/flyingthings3d/{}/TRAIN/**/**/right/*.png'.format(self.root_dir, dstype)))
            
        else:
            # TEST
            # left frames;
            self.flying_things_left = sorted(glob('{}/flyingthings3d/{}/TEST/**/**/left/*.png'.format(self.root_dir, dstype)))
            # right frames;
            self.flying_things_right = sorted(glob('{}/flyingthings3d/{}/TEST/**/**/right/*.png'.format(self.root_dir, dstype)))
            
        self.left_paths = self.flying_things_left + self.monkaa_left + self.driving_left
        self.right_paths = self.flying_things_right + self.monkaa_right + self.driving_right

        # split training files into train and val
        if split == 'train':
            # split a sub sample of train data for val;
            self.left_paths= self.left_paths[0: int(self.split_ratio*len(self.left_paths))]
            self.right_paths= self.right_paths[0: int(self.split_ratio*len(self.right_paths))]
        
        elif split =='val':
            # split a sub sample of train data for val;
            self.left_paths= self.left_paths[int(self.split_ratio*len(self.left_paths)): len(self.left_paths)]
            self.right_paths= self.right_paths[int(self.split_ratio*len(self.right_paths)): len(self.right_paths)]
            
        else:
            print('training data has been split into train and val via the split_ratio: {} and {} for train and val respectively'.format(self.split_ratio, 1-self.split_ratio))

        logging.info(f"Added {len(self.flying_things_left)} from Flying Things 3D final pass")
        logging.info(f"Added {len(self.monkaa_left)} from Monkaa final pass")
        logging.info(f"Added {len(self.driving_left)} from Driving final pass")
        
        self.transform = transform

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx]).convert('RGB')
        right_image = Image.open(self.right_paths[idx]).convert('RGB')
            
        sample = {'left_image': left_image, 'right_image': right_image}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.left_paths)