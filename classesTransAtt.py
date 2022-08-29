# KvHoeve
# 8/8/2022
# transient attributes classes


# Importing needed libraries
from __future__ import print_function, division
import os
import torch
import torch.cuda
from torch.utils.data import Dataset
from torch import nn
import numpy as np


# =============== functions for classes ==============

def pil_loader(path):
    """Opens an image in RGB. Opens path as file to avoid ResourceWarning 
    taken from (https://github.com/python-pillow/Pillow/issues/835) """
    # import package
    from PIL import Image
    
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def clean_tsv(path):
    """Takes the transient attributes annotations tsv file, reads it and
    cleans the file so that only the neccessary""" 
    # import packages
    import pandas as pd
    
    df = pd.read_csv(path, header=None, sep='\t')
    
    # only keep the columns containing the information needed
    # change the column numbers to the attributes you want to keep.
    att_df = df.iloc[:, 1:]
    
    # The file contains attribute scores and confidences, 
    # the following only keeps the scores. 
    rm_conf = att_df.applymap(lambda x: x[:x.find(',')])
    # concatenate img path column and score columns
    final_df = pd.concat([df[0], rm_conf], axis=1)
    
    return final_df
        

# =============== classes ==============
# This was created by KvHoeve using this tutorial by PyTorch,
# accessed on 11/5/2022 at https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

attributes = [
         "dirty",
         "daylight",
         "night",
         "sunrisesunset",
         "dawndusk",
         "sunny",
         "clouds",
         "fog",
         "storm",
         "snow",
         "warm",
         "cold",
         "busy",
         "beautiful",
         "flowers",
         "spring",
         "summer",
         "autumn",
         "winter",
         "glowing",
         "colorful",
         "dull",
         "rugged",
         "midday",
         "dark",
         "bright",
         "dry",
         "moist",
         "windy",
         "rain",
         "ice",
         "cluttered",
         "soothing",
         "stressful",
         "exciting",
         "sentimental",
         "mysterious",
         "boring",
         "gloomy",
         "lush"       
 ]
 
class TransAttributes(Dataset):
    
    annotations_file = "annotations.tsv"
    
    def __init__(self,                 
                 root,
                 img_root="imageLD",
                 lbl_root="annotations",
                 split_root = "holdout_split",
                 split = "train",
                 transform=None):
        """Args:
                root = base path to correct directory
                img_root = path to folder containing images
                lbl_root = path to folder containing labels
                split_root = path to folder to the image paths for the split 
                """

        self.root = root
        self.img_root = img_root
        self.lbl_root = lbl_root
        self.split_root = split_root
        self.transform = transform
        self.full_lbl = clean_tsv(os.path.join(root, lbl_root, "annotations.tsv"))
        
        if split == "train":
            self.file = os.path.join(self.root, self.split_root, "training.txt")
            with open(self.file, "r") as f:
                self.indices = f.read().splitlines()
        elif split == "test":
            self.file = os.path.join(self.root, self.split_root, "test.txt")
            with open(self.file, "r") as f:
                self.indices = f.read().splitlines()
            
        self.labels = self.full_lbl.loc[self.full_lbl[0].isin(self.indices)]
        

    def __len__(self):
        return len(self.labels)
    
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.img_root, self.labels.iloc[idx, 0])
        image = pil_loader(img_path)
        label = self.labels.iloc[idx,1:]
        label = torch.from_numpy(np.array(label, dtype=float))
        if self.transform:
            image = self.transform(image)
        return image, label, img_path


class TACAM(nn.Module):
    def __init__(self):
        """ A newly defined model architecture that uses ResNet50 as a base,
        but has no fully vonnected layers. Instead, it uses convolutions and
        a global average pool to create Class Activation Maps.
        Lastly, another avreage pool layer creates the score per attribute."""        
        super(TACAM, self).__init__()
        
        self.base = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.base.fc = nn.Identity()
        self.base.avgpool = nn.Identity()
        self.cam = nn.Sequential(nn.Conv2d(2048, 40, 1), 
                                 nn.AdaptiveAvgPool2d(224))
        self.score = nn.Sequential(nn.AdaptiveAvgPool2d(1))
        
    def forward(self, img):
        x = self.base(img)
        maps = self.cam(torch.reshape(x, (len(img), 2048, 7, 7)))
        scores = self.score(maps)        
        
        return maps, scores
             
            
        
        
    