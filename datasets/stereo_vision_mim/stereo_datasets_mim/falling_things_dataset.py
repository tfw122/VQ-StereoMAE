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

@registry.register_datamodule("falling_things_single_mim")
class FallingThingsSingleLoader(Dataset):
    def __init__(self, config, split, transform=None):
        self.dataset_config = config.dataset_config
        self.root_dir = '{}/falling_things/fat/single'.format(self.dataset_config.root_dir)

        self.split_ratio = self.dataset_config.db_split_ratio

        self.left_paths= sorted(glob('{}/**/**/*.left.jpg'.format(self.root_dir)))
        self.right_paths= sorted(glob('{}/**/**/*.right.jpg'.format(self.root_dir)))

        if split =='train':
            self.left_paths= self.left_paths[0: int(self.split_ratio*len(self.left_paths))]
            self.right_paths= self.right_paths[0: int(self.split_ratio*len(self.right_paths))]
            print("fallingthings_data_process")

        elif split=='val' or split=='test':
            self.left_paths= self.left_paths[int(self.split_ratio*len(self.left_paths)): len(self.left_paths)]
            self.right_paths= self.right_paths[int(self.split_ratio*len(self.right_paths)): len(self.right_paths)]
            
        self.transform = transform
        logging.info(f"Added {len(self.left_paths)} from Falling Things Single")

    def __getitem__(self, idx):
        #print(self.left_paths[idx])
        left_image = Image.open(self.left_paths[idx]).convert('RGB')
        right_image = Image.open(self.right_paths[idx]).convert('RGB')
       
        sample = {'left_image': left_image, 'right_image': right_image}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.left_paths)
    


@registry.register_datamodule("falling_things_mixed_mim")
class FallingThingsMixedLoader(Dataset):
    def __init__(self, config, split, transform=None):
        self.dataset_config = config.dataset_config
        self.root_dir = '{}/fallingthings/fat/mixed'.format(self.dataset_config.root_dir)

        self.split_ratio = self.dataset_config.db_split_ratio

        self.left_paths= sorted(glob('{}/**/*.left.jpg'.format(self.root_dir)))
        self.right_paths= sorted(glob('{}/**/*.right.jpg'.format(self.root_dir)))

        if split =='train':
            self.left_paths= self.left_paths[0: int(self.split_ratio*len(self.left_paths))]
            self.right_paths= self.right_paths[0: int(self.split_ratio*len(self.right_paths))]

        elif split=='val' or split=='test':
            self.left_paths= self.left_paths[int(self.split_ratio*len(self.left_paths)): len(self.left_paths)]
            self.right_paths= self.right_paths[int(self.split_ratio*len(self.right_paths)): len(self.right_paths)]
            
        self.transform = transform
        logging.info(f"Added {len(self.left_paths)} from Falling Things Mixed")

    def __getitem__(self, idx):
        #print(self.left_paths[idx])
        left_image = Image.open(self.left_paths[idx]).convert('RGB')
        right_image = Image.open(self.right_paths[idx]).convert('RGB')
       
        sample = {'left_image': left_image, 'right_image': right_image}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.left_paths)