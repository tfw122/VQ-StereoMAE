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
from datasets.dataset_zoo.downstream.base_stereo_dataset import * 
from datasets.dataset_zoo.downstream.downstream_data.sceneflow_dataset import *

@registry.register_builder("stereo_vision_downstream")
class StereoVisionDownstreamDatasetModule(BaseDatasetBuilder):
    """
    args:
        config: the config file from /configs/*
    output:
        LightningDatasetModule: Datasetloaders for the trainer
    """
    def __init__(self, config):
        self.config= config
        self.dataset_config = self.config.dataset_config
        self.multiply_bool = self.dataset_config.use_multiplier.multiply
        # get the names;
        self.preprocess_config = self.dataset_config.preprocess.vision_transforms.params
        self.dataset_name = self.config.dataset_config.dataset_name
        self.transforms_name = self.config.dataset_config.preprocess.name

    def augmentation_parameters(self):
        aug_params = {'crop_size': self.preprocess_config.Resize.size, 'min_scale': self.preprocess_config.spatial_scale[0], 'max_scale': self.preprocess_config.spatial_scale[1], 'do_flip': self.preprocess_config.do_flip, 'yjitter': not self.preprocess_config.noyjitter}
        if hasattr(self.preprocess_config, "saturation_range") and self.preprocess_config.saturation_range is not None:
            aug_params["saturation_range"] = tuple(self.preprocess_config.saturation_range)
        if hasattr(self.preprocess_config, "img_gamma") and self.preprocess_config.img_gamma is not None:
            aug_params["gamma"] = self.preprocess_config.img_gamma
        if hasattr(self.preprocess_config, "do_flip") and self.preprocess_config.do_flip is not None:
            aug_params["do_flip"] = self.preprocess_config.do_flip
        return aug_params

    def data_setup(self, split):
        aug_params = self.augmentation_parameters()
        all_datasets = build_dataset(self.config)
        # aug_params, root, split
        datasets = []

        if split=='train':
            # initiate datasets
            for key, dataset_class in all_datasets.items():
                db = dataset_class(config=self.config, aug_params=aug_params, split='train')
                if self.multiply_bool:
                    db = self.multiplier(db, key)
                datasets.append(db)

        elif split=='val':
            # initiate datasets
            datasets.append(SceneFlowDatasetsFinal(config=self.config, aug_params=aug_params, split='val', dstype='frames_finalpass', things_test=True))
        
        # Assign test dataset for use in dataloader(s)
        else:
            # ONLY NEED ONE DATASET = Sceneflow
            datasets.append(SceneFlowDatasetsClean(config=self.config, aug_params=aug_params, split='val', things_test=True))
            datasets.append(SceneFlowDatasetsFinal(config=self.config, aug_params=aug_params, split='val', things_test=True))
        
        # concatenate datasets
        if len(datasets) > 1:
            # https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649/25?page=2
            dataset = torch.utils.data.ConcatDataset(datasets)
        else:
            dataset = datasets[0]
        
        return dataset
    
    def multiplier(self, dataset_class, key):
        # for some reason, RAFT multiplies some of the dataset size;
        # https://github.com/princeton-vl/RAFT-Stereo/blob/main/core/stereo_datasets.py#L301
        multiplier_dict = self.dataset_config.use_multiplier.multiplier_dict
        multiplier_names = self.dataset_config.use_multiplier.db_to_multiply
        if key in multiplier_names:
            dataset_class = dataset_class * multiplier_dict[key]
        return dataset_class
