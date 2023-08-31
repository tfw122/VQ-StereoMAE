from pandas import DataFrame
from datasets.base_dataset_builder import BaseDatasetBuilder
import pytorch_lightning as pl
from imports.registry import registry
import torchvision.datasets as datasets
from torch.utils.data import random_split
import os 

@registry.register_builder("imagenet")
class ImagenetDatasetModule(BaseDatasetBuilder):
    def __init__(self, config):
        self.config= config
        self.toy_dataset = self.config.dataset_config.dataset_name
        self.download_bool = self.config.dataset_config.download
        self.save_dir = self.config.dataset_config.save_dir
        
        print("using the dataset: {}, stored in the following directory: {}. No Download necessary".format(self.toy_dataset, self.save_dir)) 
        self.transforms_name = self.config.dataset_config.preprocess.name

    def preprocess(self, split):
        data_transform_cls = registry.get_preprocessor_class(self.transforms_name)
        data_transforms_obj = data_transform_cls(self.config, split)
        return data_transforms_obj

    def data_setup(self, split):
        transform= self.preprocess(split)
        
        if split=='train':
            train_dataset = datasets.ImageNet(self.save_dir, split='train', transform=transform)
            return train_dataset
        
        if split=='val':
            print('<===== Warning! =====>')
            print('do not use val dataset for validation! in Imagenet for val dataset is used for testing')
            print('instead, this code will take a snippet of the training set and use it as val')
            print('<===== Warning! =====>')

            data_full = datasets.ImageNet(self.save_dir, split='train', transform=transform)
            total_files = len(data_full)
            val_samples = self.config.dataset_config.val_samples
            _, val_dataset = random_split(data_full, [total_files-val_samples, val_samples])
            return val_dataset
        
        # Assign test dataset for use in dataloader(s)
        if split == "test":
            print('the val dataset is used in Imagenet for testing')
            test_dataset = datasets.ImageNet(self.save_dir, split='val', transform=transform)
            #datasets.ImageFolder('{}/test/'.format(self.save_dir), transform=transform)
            return test_dataset