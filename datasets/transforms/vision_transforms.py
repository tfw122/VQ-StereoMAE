import collections
import torch
from torchaudio import transforms
import torchvision.transforms as transforms
import numpy as np 
import random
from PIL import ImageFilter, ImageOps
#import albumentations as A

import torch
from imports.constants import INCEPTION_IMAGE_NORMALIZE
from imports.registry import registry
from datasets.base_transforms import BaseTransforms
from omegaconf import OmegaConf
from datasets.transforms.vision_transforms_utils import *


@registry.register_transforms("vision")
class VisionTransforms(BaseTransforms):
    def __init__(self, config, split, *args, **kwargs):
        self.dataset_config = config.dataset_config
        
        self.transforms_config = config.dataset_config.preprocess.vision_transforms
        if split=='train':
            transforms_name_list = self.transforms_config.transforms_train
        else:
            transforms_name_list = self.transforms_config.transforms_test

        transform_params = self.transforms_config.params
        
        #if transform_params!=None:
        #    transform_params = [transform_params]

        transforms_list = []

        for name in transforms_name_list:
            # get the transform class;
            transforms_cls = registry.get_preprocessor_class(name)
            # get the params;
            params = transform_params.get(name)
            # initiate the transforms with the right arg;
            if params!=None:
                transforms_obj = transforms_cls(**params)
            else:
                transforms_obj = transforms_cls()
            
            transforms_list.append(transforms_obj)

        self.transform = transforms.Compose(transforms_list)

    def __call__(self, x):
        return self.transform(x)


