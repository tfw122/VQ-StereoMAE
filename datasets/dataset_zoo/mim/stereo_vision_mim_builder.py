import pandas as pd 
import numpy as np 
from pandas import DataFrame
from datasets.base_dataset_builder import BaseDatasetBuilder
from builder import build_dataset
import pytorch_lightning as pl
from imports.registry import registry
import torchvision.datasets as datasets
from torch.utils.data import random_split, Sampler
from omegaconf import OmegaConf
import torch

@registry.register_builder("stereo_vision_mim")
class StereoVisionMaskedImageModellingDatasetModule(BaseDatasetBuilder):
    """
    args:
        config: the config file from /configs/*
    output:
        LightningDatasetModule: Datasetloaders for the trainer
    """
    def __init__(self, config):
        self.config= config
        self.dataset_config = self.config.dataset_config
        # get the names;
        self.dataset_name = self.config.dataset_config.dataset_name
        self.transforms_name = self.config.dataset_config.preprocess.name
        print("transform name", self.transforms_name)
        self.image_type = self.dataset_config.dataset_mode.image_type
        self.individual_name = self.dataset_config.dataset_mode.individual_name


    def preprocess(self, split):
        data_transform_cls = registry.get_preprocessor_class(self.transforms_name)
        data_transforms_obj = data_transform_cls(self.config, split)
        return data_transforms_obj

    def data_setup(self, split):
        transform= self.preprocess(split)
        all_datasets = build_dataset(self.config)
        datasets = []

        if split=='train':
            # initiate datasets
            for _, dataset_class in all_datasets.items():
                datasets.append(dataset_class(self.config, split="train", transform = transform))

        elif split=='val':
            # initiate datasets
            for _, dataset_class in all_datasets.items():
                datasets.append(dataset_class(self.config, split="val", transform = transform))
        
        # Assign test dataset for use in dataloader(s)
        else:
            # initiate datasets
            for _, dataset_class in all_datasets.items():
                datasets.append(dataset_class(self.config, split="test", transform = transform))
        
        # concatenate datasets
        if len(datasets) > 1:
            # https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649/25?page=2
            dataset = torch.utils.data.ConcatDataset(datasets)
        else:
            dataset = datasets[0]
        
        return dataset

