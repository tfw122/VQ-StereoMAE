import torch
from torch.utils.data.dataset import Dataset
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
import numpy as np 
import random
import math
import logging
import os
import re

from datasets.base_dataset import BaseDataset
from imports.registry import registry

from omegaconf import OmegaConf
from pathlib import Path
from glob import glob
import copy 
import os.path as osp

import modules.raft.frame_utils 
from modules.raft.augmentor import FlowAugmentor, SparseFlowAugmentor
from datasets.dataset_zoo.downstream.base_stereo_dataset import *


@registry.register_datamodule("middlebury_downstream")
class Middlebury(StereoDataset):
    def __init__(self, config, aug_params=None, root='../data/middlebury', split='train', sub_split='F'):
        super(Middlebury, self).__init__(config, aug_params, sparse=True, reader=frame_utils.readDispMiddlebury)
        assert os.path.exists(root)
        assert sub_split in ["F", "H", "Q", "all", "2014"]
        if sub_split == "2014": # datasets/Middlebury/2014/Pipes-perfect/im0.png
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E","L",""]:
                    self.image_list += [ [str(scene / "im0.png"), str(scene / f"im1{s}.png")] ]
                    self.disparity_list += [ str(scene / "disp0.pfm") ]
        else:
            lines = list(map(osp.basename, glob(os.path.join(root, "trainingF/*"))))
            lines = list(filter(lambda p: any(s in p.split('/') for s in Path(os.path.join(root, "official_train.txt")).read_text().splitlines()), lines))
            image1_list = sorted([os.path.join(root,  f'training{sub_split}', f'{name}/im0.png') for name in lines])
            image2_list = sorted([os.path.join(root, f'training{sub_split}', f'{name}/im1.png') for name in lines])
            disp_list = sorted([os.path.join(root,  f'training{sub_split}', f'{name}/disp0GT.pfm') for name in lines])
            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, sub_split]
        
            if split=='train':
                for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                    self.image_list += [ [img1, img2] ]
                    self.disparity_list += [ disp ]
            
            else:
                self.image_list = []
                self.disparity_list= []
            
        logging.info(f"Added {len(self.disparity_list)} from middlebury")


@registry.register_datamodule("middlebury_custom_downstream")
class MiddleburyCustom(StereoDataset):
    def __init__(self, config, aug_params=None, root='../data/middlebury', split='train', sub_split='all'):
        super(MiddleburyCustom, self).__init__(config, aug_params, sparse=True, reader=frame_utils.readDispMiddlebury)
        assert os.path.exists(root)
        assert sub_split in ["F", "H", "Q", "all", "2014"]
        if sub_split == "2014": # datasets/Middlebury/2014/Pipes-perfect/im0.png
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E","L",""]:
                    self.image_list += [ [str(scene / "im0.png"), str(scene / f"im1{s}.png")] ]
                    self.disparity_list += [ str(scene / "disp0.pfm") ]
        else:
            linesF = list(map(osp.basename, glob(os.path.join(root, "trainingF/*"))))
            #linesF = list(filter(lambda p: any(s in p.split('/') for s in Path(os.path.join(root, "MiddEval3/official_train.txt")).read_text().splitlines()), linesF))
            
            linesH = list(map(osp.basename, glob(os.path.join(root, "trainingH/*"))))
            
            linesQ = list(map(osp.basename, glob(os.path.join(root, "trainingQ/*"))))
            

            image1_listF = sorted([os.path.join(root,  f'trainingF', f'{name}/im0.png') for name in linesF])
            image1_listH = sorted([os.path.join(root,  f'trainingH', f'{name}/im0.png') for name in linesH])
            image1_listQ = sorted([os.path.join(root, f'trainingQ', f'{name}/im0.png') for name in linesQ])
            
            image2_listF = sorted([os.path.join(root,  f'trainingF', f'{name}/im1.png') for name in linesF])
            image2_listH = sorted([os.path.join(root, f'trainingH', f'{name}/im1.png') for name in linesH])
            image2_listQ = sorted([os.path.join(root,  f'trainingQ', f'{name}/im1.png') for name in linesQ])
            
            disp_listF = sorted([os.path.join(root,  f'trainingF', f'{name}/disp0GT.pfm') for name in linesF])
            disp_listH = sorted([os.path.join(root,  f'trainingH', f'{name}/disp0GT.pfm') for name in linesH])
            disp_listQ = sorted([os.path.join(root,  f'trainingQ', f'{name}/disp0GT.pfm') for name in linesQ])

            image1_list= image1_listF + image1_listH + image1_listQ
            image2_list= image2_listF + image2_listH + image2_listQ
            disp_list= disp_listF + disp_listH + disp_listQ
            
            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, sub_split]
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]

            if split=='train':
                for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                    self.image_list += [ [img1, img2] ]
                    self.disparity_list += [ disp ]
            
            else:
                self.image_list = []
                self.disparity_list= []
            
        logging.info(f"Added {len(self.disparity_list)} from middlebury_custom")