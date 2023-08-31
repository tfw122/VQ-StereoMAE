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
import logging 

@registry.register_datamodule("kitti_eigen_mim")
class KittiEigenStereoLoader(Dataset):
    def __init__(self, config, split, transform=None):
        self.dataset_config = config.dataset_config
        self.root_dir = '{}/kitti_data'.format(self.dataset_config.root_dir)

        if split =='train':
            self.all_paths= np.load('src/datasets/db_utils/dataset_splits/kitti_eigen_splits/eigen_train_paths.npy')
            print("kitti_data_process")
            
        elif split=='val' or split=='test':
            self.all_paths= np.load('src/datasets/db_utils/dataset_splits/kitti_eigen_splits/eigen_val_paths.npy')
        #test is actually meant to be the eigen paths used for testing! i.e. same as in the eval.py file.
        self.transform = transform

    def __getitem__(self, idx):
        split_path= self.all_paths[idx].split(' ') # so 0 = left path, 1 = right path
        left_image = Image.open('{}/kitti/{}'.format(self.root_dir, split_path[0])).convert('RGB')
        right_image = Image.open('{}/kitti/{}'.format(self.root_dir, split_path[1])).convert('RGB')
            
        sample = {'left_image': left_image, 'right_image': right_image}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.all_paths)

@registry.register_datamodule("kitti_custom_mim")
class KittiLoaderCustom(Dataset):
    def __init__(self, config, split, transform=None):
        self.dataset_config = config.dataset_config
        self.root_dir = self.dataset_config.root_dir
        self.split_ratio = self.dataset_config.db_split_ratio

        self.left_paths= sorted(glob('{}/kitti/**/**/image_02/data/*.png'.format(self.root_dir)))
        self.right_paths= sorted(glob('{}/kitti/**/**/image_03/data/*.png'.format(self.root_dir)))

        if split =='train':
            self.left_paths= self.left_paths[0: int(self.split_ratio*len(self.left_paths))]
            self.right_paths= self.right_paths[0: int(self.split_ratio*len(self.right_paths))]

        elif split=='val' or split=='test':
            self.left_paths= self.left_paths[int(self.split_ratio*len(self.left_paths)): len(self.left_paths)]
            self.right_paths= self.right_paths[int(self.split_ratio*len(self.right_paths)): len(self.right_paths)]
        
        self.transform = transform
        logging.info(f"Added {len(self.left_paths)} from Kitti")

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx]).convert('RGB')
        right_image = Image.open(self.right_paths[idx]).convert('RGB')
            
        sample = {'left_image': left_image, 'right_image': right_image}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.left_paths)
