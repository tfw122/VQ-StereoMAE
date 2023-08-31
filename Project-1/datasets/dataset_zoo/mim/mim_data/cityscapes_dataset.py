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

@registry.register_datamodule("cityscapes_mim")
class CityScapesLoader(Dataset):
    def __init__(self, config, split, transform=None, n_val_samples=1000):
        self.dataset_config = config.dataset_config
        self.root_dir = self.dataset_config.root_dir
        # Crop out the bottom 25% of the image to remove the car logo
        self.crop_height=0.75 # keep 75% of the height from top.
        if split=='train':
            self.left_paths= sorted(glob('{}/cityscapes/leftImg8bit_sequence/train/**/*.png'.format(self.root_dir)))
            self.right_paths= sorted(glob('{}/cityscapes/rightImg8bit_sequence/train/**/*.png'.format(self.root_dir)))
            print("cityscapes_data_process")

        elif split=='val':
            self.left_paths= sorted(glob('{}/cityscapes/leftImg8bit_sequence/val/**/*.png'.format(self.root_dir)))[0:1000]
            self.right_paths= sorted(glob('{}/cityscapes/rightImg8bit_sequence/val/**/*.png'.format(self.root_dir)))[0:1000]

        elif split=='test':
            self.left_paths= sorted(glob('{}/cityscapes/leftImg8bit_sequence/test/**/*.png'.format(self.root_dir)))
            self.right_paths= sorted(glob('{}/cityscapes/rightImg8bit_sequence/test/**/*.png'.format(self.root_dir)))
            
        
        self.transform = transform

        logging.info(f"Added {len(self.left_paths)} from Cityscapes")

    def __getitem__(self, idx):
        #print(self.left_paths[idx])
        left_image = Image.open(self.left_paths[idx]).convert('RGB')
        right_image = Image.open(self.right_paths[idx]).convert('RGB')
        # crop the image to have the height up to 75%. To remove the car logo (Mercedes)
        h,_,_= np.shape(left_image)
        left_image= np.array(left_image)[:int(h*self.crop_height),:,:]
        right_image= np.array(right_image)[:int(h*self.crop_height),:,:]
        # convert image back to PIL:
        left_image= Image.fromarray(np.uint8(left_image))
        right_image= Image.fromarray(np.uint8(right_image))
       
        #img = imresize(img, (self.img_height, self.img_width))[:int(self.img_height*0.75)]
        sample = {'left_image': left_image, 'right_image': right_image}

        if self.transform:
            sample = self.transform(sample)
    
        return sample

    def __len__(self):
        return len(self.left_paths)


if __name__=='__main__':
    import torch
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    h,w= 256, 512
    size=(h,w)

    transformations= transforms.Compose([transforms.Resize(size),
                                        transforms.ToTensor()])

    dataset= CityScapesLoader('rand_path', 'train', transformations)


    def ToArray(img_t):
        img = img_t.detach().to("cpu").numpy()
        img = np.transpose(img, (1, 2, 0))

        return img

    for i in range(5):
        sample= dataset[i]
        left_img= sample['left_image']
        right_img= sample['right_image']
        print(left_img.size(), right_img.size())

        left_img= ToArray(left_img)
        right_img= ToArray(right_img)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('CityScapes Stereo Images')
        ax1.imshow(left_img)
        ax2.imshow(right_img)
        plt.show()