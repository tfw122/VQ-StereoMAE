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

@registry.register_datamodule("sintel_stereo_mim")
class SintelStereoLoader(Dataset):
    def __init__(self, config, split, transform=None):
        self.dataset_config = config.dataset_config
        self.root_dir = "{}/sintel_stereo/".format(self.dataset_config.root_dir)
        self.split_ratio = self.dataset_config.db_split_ratio
        
       
        # All training left and right frames; (later split by the split_ratio into train and val)
        self.left_paths = sorted(glob('{}/training/clean_left/**/*.png'.format(self.root_dir)))
        self.right_paths= sorted(glob('{}/training/clean_right/**/*.png'.format(self.root_dir)))
        
        # split training files into train and val
        if split == 'train':
            # split a sub sample of train data for val;
            self.left_paths= self.left_paths[0: int(self.split_ratio*len(self.left_paths))]
            self.right_paths= self.right_paths[0: int(self.split_ratio*len(self.right_paths))]
            print("sintel_data_process")
            
        elif split=='val' or split=='test':
            # split a sub sample of train data for val;
            self.left_paths= self.left_paths[int(self.split_ratio*len(self.left_paths)): len(self.left_paths)]
            self.right_paths= self.right_paths[int(self.split_ratio*len(self.right_paths)): len(self.right_paths)]
            
        else:
            print('training data has been split into train and val via the split_ratio: {} and {} for train and val respectively'.format(self.split_ratio, 1-self.split_ratio))

        self.transform = transform
        logging.info(f"Added {len(self.left_paths)} from Sintel Stereo")

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx]).convert('RGB')
        right_image = Image.open(self.right_paths[idx]).convert('RGB')
            
        sample = {'left_image': left_image, 'right_image': right_image}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def __len__(self):
        return len(self.left_paths)