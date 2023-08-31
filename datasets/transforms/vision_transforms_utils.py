
import torch
import torchvision.transforms as transforms
import numpy as np 
import random
from PIL import ImageFilter, ImageOps
#import albumentations as A
from imports.constants import * #INCEPTION_IMAGE_NORMALIZE, IMAGE_COLOR_MEAN, IMAGE_COLOR_STD
from imports.registry import registry
from datasets.base_transforms import BaseTransforms

# dict input to all = {'left_image': left_image, 'right_image': right_image}

@registry.register_transforms("Normalise")
class Normalise(BaseTransforms):
    def __init__(self, norm_type='imagenet'):
        if norm_type=='imagenet':
            self.transform= transforms.Normalize(
                              mean=IMAGE_COLOR_MEAN,
                              std=IMAGE_COLOR_STD)
        elif norm_type=='inception':
            self.transform = transforms.Normalize(
                              mean=INCEPTION_IMAGE_NORMALIZE, 
                              std= INCEPTION_IMAGE_NORMALIZE)
        else:
            raise Exception("the following type of normalisation mean and std. : {} cannot be found in src/common/constants.py\
                             please add mean and std. accordingly".format(norm_type))

    def __call__(self, sample):
        if type(sample)==dict:
            images = sample['images']
        else:
            images= sample
        
        new_images = self.transform(images)
        
        if type(sample)==dict:
            sample['images']= new_images
        else:
            sample = new_images
        return sample

@registry.register_transforms("UnNormalise")
class UnNormalise(BaseTransforms):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

@registry.register_transforms("Resize")
class ResizeImages(BaseTransforms):
    def __init__(self, size):
        self.transform = transforms.Resize(size)

    def __call__(self, sample):
        left_image = sample['left_image']
        right_image = sample['right_image']

        new_right_image = self.transform(right_image)
        new_left_image = self.transform(left_image)

        sample = {'left_image': new_left_image, 'right_image': new_right_image}

        return sample

@registry.register_transforms("RandomResizedCrop")
class RandomResizedCrop(BaseTransforms):
    def __init__(self, size, scale=(0.2, 1.0), interpolation=3):
        # interpolation=3 is bicubic
        self.transform = transforms.RandomResizedCrop(size, scale=scale, interpolation=interpolation)

    def __call__(self, sample):
        left_image = sample['left_image']
        right_image = sample['right_image']

        new_right_image = self.transform(right_image)
        new_left_image = self.transform(left_image)

        sample = {'left_image': new_left_image, 'right_image': new_right_image}

        return sample

@registry.register_transforms("ToTensor")
class ToTensor(BaseTransforms):
    def __init__(self):
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        left_image = sample['left_image']
        right_image = sample['right_image']

        new_right_image = self.transform(right_image)
        new_left_image = self.transform(left_image)

        sample = {'left_image': new_left_image, 'right_image': new_right_image}

        return sample

@registry.register_transforms("RandomErasing")
class RandomErasing(BaseTransforms):
    def __init__(self):
        self.transform = transforms.RandomErasing(p=1, scale = (0.02, 0.2), ratio = (0.3, 3.3))

    def __call__(self, sample):
        left_image = sample['left_image']
        right_image = sample['right_image']
        
        k = np.random.uniform(0, 1, 1)
        if k > 0.5:
            new_right_image = self.transform(right_image)
            new_left_image = self.transform(left_image)
        
            sample = {'left_image': new_left_image, 'right_image': new_right_image}
        else:
            sample = {'left_image': left_image, 'right_image': right_image}
       
        return sample

@registry.register_transforms("RandomRotation")
class RandomRotation(BaseTransforms):
    def __init__(self, degrees=(20,180)):
        self.degrees = degrees
        self.transform = transforms.RandomRotation(self.degrees, interpolation=transforms.functional.InterpolationMode.BILINEAR)

    def __call__(self, sample):
        left_image = sample['left_image']
        right_image = sample['right_image']
        
        k = np.random.uniform(0, 1, 1)
        if k > 0.5:
            new_right_image = self.transform(right_image)
            new_left_image = self.transform(left_image)
        
            sample = {'left_image': new_left_image, 'right_image': new_right_image}
        else:
            sample = {'left_image': left_image, 'right_image': right_image}
       
        return sample

@registry.register_transforms("RandomAffine")
class RandomAffine(BaseTransforms):
    def __init__(self, degrees = (20,180)):
        self.degrees = degrees
        self.transform = transforms.RandomAffine(self.degrees, translate= (0.1,0.1), scale=(0.5,0.75), shear=0,
                                                 interpolation=transforms.functional.InterpolationMode.BILINEAR)

    def __call__(self, sample):
        left_image = sample['left_image']
        right_image = sample['right_image']
        
        k = np.random.uniform(0, 1, 1)
        if k > 0.5:
            new_right_image = self.transform(right_image)
            new_left_image = self.transform(left_image)
        
            sample = {'left_image': new_left_image, 'right_image': new_right_image}
        else:
            sample = {'left_image': left_image, 'right_image': right_image}
       
        return sample

@registry.register_transforms("RandomPerspective")
class RandomPerspective(BaseTransforms):
    def __init__(self):
        self.transform = transforms.RandomPerspective(distortion_scale=0.2, p=1,
                                                 interpolation=transforms.functional.InterpolationMode.BILINEAR)

    def __call__(self, sample):
        left_image = sample['left_image']
        right_image = sample['right_image']
        
        k = np.random.uniform(0, 1, 1)
        if k > 0.5:
            new_right_image = self.transform(right_image)
            new_left_image = self.transform(left_image)
        
            sample = {'left_image': new_left_image, 'right_image': new_right_image}
        else:
            sample = {'left_image': left_image, 'right_image': right_image}
       
        return sample

@registry.register_transforms("RandomHorizontalFlip")
class RandomHorizontalFlip(BaseTransforms):
    def __init__(self):
        self.transform = transforms.RandomHorizontalFlip(p=1)

    def __call__(self, sample):
        left_image = sample['left_image']
        right_image = sample['right_image']
        
        k = np.random.uniform(0, 1, 1)
        if k > 0.5:
            new_right_image = self.transform(right_image)
            new_left_image = self.transform(left_image)
        
            sample = {'left_image': new_left_image, 'right_image': new_right_image}
        else:
            sample = {'left_image': left_image, 'right_image': right_image}
       
        return sample

# random colour jitter.
@registry.register_transforms("RandomAugment")
class AugmentImageBlock(BaseTransforms):
    def __init__(self, augment_parameters, do_augmentation):
        self.do_augmentation = do_augmentation
        self.gamma_low = augment_parameters[0]  # 0.8
        self.gamma_high = augment_parameters[1]  # 1.2
        self.brightness_low = augment_parameters[2]  # 0.5
        self.brightness_high = augment_parameters[3]  # 2.0
        self.color_low = augment_parameters[4]  # 0.8
        self.color_high = augment_parameters[5]  # 1.2

    def run_rand_gamma(self, fnc, img):
        img_new = img ** fnc
        return img_new

    def run_rand_brightness(self, fnc, img):
        img_new = img * fnc
        return img_new

    def run_rand_color_shift(self, fnc, img):
        for i in range(3):
            img[i, :, :] *= fnc[i]
        return img

    def run_rand_saturation(self, img):
        img_new = torch.clamp(img, 0, 1)
        return img_new

    def __call__(self, sample):
        left_image = sample['left_image']
        right_image = sample['right_image']
        
        p = np.random.uniform(0, 1, 1)
        
        if self.do_augmentation:
            if p > 0.5:
                # randomly shift gamma
                random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
                left_image_aug = self.run_rand_gamma(random_gamma, left_image)
                right_image_aug = self.run_rand_gamma(random_gamma, right_image)

                # randomly shift brightness
                random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
                left_image_aug =  self.run_rand_brightness(random_brightness, left_image_aug) 
                right_image_aug =  self.run_rand_brightness(random_brightness, right_image_aug)

                # randomly shift color
                random_colors = np.random.uniform(self.color_low, self.color_high, 3)
                left_image_aug = self.run_rand_color_shift(random_colors, left_image_aug)
                right_image_aug = self.run_rand_color_shift(random_colors, right_image_aug)

                # saturate
                left_image_aug = self.run_rand_saturation(left_image_aug)
                right_image_aug = self.run_rand_saturation(right_image_aug)

                sample = {'left_image': left_image_aug, 'right_image': right_image_aug}

        else:
            sample = {'left_image': left_image, 'right_image': right_image}

        return sample

@registry.register_transforms("GaussianBlur")
class GaussianBlur(BaseTransforms):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, do_augmentation, sigma=[.1, 2.]):
        self.do_augmentation= do_augmentation
        self.sigma = sigma

    def run_transform(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img_new = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img_new

    def __call__(self, sample):
        left_image = sample['left_image']
        right_image = sample['right_image']
        
        p = np.random.uniform(0, 1, 1)

        if self.do_augmentation:
            if p > 0.5:
                new_right_image = self.run_transform(right_image)
                new_left_image = self.run_transform(left_image)
                
                sample = {'left_image': new_left_image, 'right_image': new_right_image}

        else:
            sample = {'left_image': left_image, 'right_image': right_image}
        
        return sample

@registry.register_transforms("Solarize")
class Solarize(BaseTransforms):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __init__(self, version= "torch", threshold=192.0, do_augmentation=True):
        self.do_augmentation= do_augmentation
        self.transform = transforms.RandomSolarize(threshold, p=0.5)
        self.version= version

    def run_transform(self, img):
        if self.version=="torch":
            img_new = self.transform(img)
        else:
            img_new = ImageOps.solarize(img)
        return img_new

    def __call__(self, sample):
        left_image = sample['left_image']
        right_image = sample['right_image']
        
        p = np.random.uniform(0, 1, 1)

        if self.do_augmentation:
            if p > 0.5:
                new_right_image = self.run_transform(right_image)
                new_left_image = self.run_transform(left_image)
                
                sample = {'left_image': new_left_image, 'right_image': new_right_image}

        else:
            sample = {'left_image': left_image, 'right_image': right_image}
        
        return sample