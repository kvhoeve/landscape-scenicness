# AADB Dataset classses
# Kvhoeve
# 26/7/22
# GNU General Public License v3.0

# =============== functions for classes ==============
# reading the mat annotations file into a pandas dataframe

# Importing needed libraries
from __future__ import print_function, division
import os
import torch
import torch.cuda
from torch.utils.data import Dataset
from torch import nn
import numpy as np
from mat4py import loadmat



def mat_to_pddf(root, mat_name='AADBinfo.mat', dataset = "train"):
    """The AADB annotations file is in matlab format.
    In this function it is converted to 2 pandas dataframes with columns for:
    testName, testScore in pd_test and trainName, trainScore in pd_train"""

    # import packages
    from mat4py import loadmat
    import pandas as pd
    
    # load in the mat file
    mat_file = loadmat(os.path.join(root, mat_name))
    
    # extract infromation in dictionaries
    for c in range(0,2,2):
        key_name = list(mat_file.keys())
        val_dict1 = mat_file.pop(key_name[c])
        val_dict2 = mat_file.pop(key_name[c+1])
        if dataset == "train" :
            pd_frame = pd.DataFrame(mat_file)
        else:
            dict_sub = {key_name[c]: val_dict1, key_name[c + 1]: val_dict2}
            pd_frame = pd.DataFrame(dict_sub)

    return pd_frame


def pil_loader(path):
    """Opens an image in RGB. Opens path as file to avoid ResourceWarning 
    taken from (https://github.com/python-pillow/Pillow/issues/835) by Kanoc
    on 9/5/22"""
    # import package
    from PIL import Image
    
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# =============== classes ==============
# creating dataloader by first defining the dataset
# This was created by KvHoeve using this tutorial by PyTorch,
# accessed on 11/5/2022 at https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# With adaptations from isaaccorley on 28/7/2022
# https://github.com/isaaccorley/deep-aesthetics-pytorch/blob/main/torch_aesthetics/aadb.py

attributes = [
      "score",
      "balancing_elements",
      "color_harmony",
      "content",
      "depth_of_field",
      "light",
      "motion_blur",
      "object",
      "repetition",
      "rule_of_thirds",
      "symmetry",
      "vivid_color"
  ]

class AADB(Dataset):


    splits = {
        "train": {"idx": 0, "file": "imgListTrainRegression_score.txt"},
        "test": {"idx": 1, "file": "imgListTestNewRegression_score.txt"},
        "val": {"idx": 2, "file": "imgListValidationRegression_score.txt"}
    }

    labels_file = "attMat.mat"

    def __init__(self,
                 root,
                 img_root="datasetImages_originalSize",
                 lbl_root="imgListFiles_label",
                 split="train",
                 transform=None):
        """Args:
                img_root = path to file containing images
                lbl_root = path to file containing labels
                split = default on train for accessing train, validation or test datasets
                """

        self.root = root
        self.img_root = img_root
        self.lbl_root = lbl_root
        self.split = split
        self.transform = transform
        self.files, self.labels = self.load_split(split)

    def load_split(self, split):
        # Load labels
        assert split in ["train", "val", "test"]
        labels_path = os.path.join(self.root, self.lbl_root, self.labels_file)
        labels = loadmat(labels_path)["dataset"]
        labels = labels[self.splits[split]["idx"]]
################################################        
      #  idx_list =[]
      #  tens_list=[]
      #  for i in range(len(labels)):
      #      if labels[i][10] > 0.0:
      #          idx_list.append(i)
      #          tens_list.append(labels[i][6])
###################### tens_list veranderen ##################        
        labels = np.array(labels)
        labels = np.delete(labels, [6], 1)

        # Load file paths
        files_path = os.path.join(self.root, self.lbl_root, self.splits[split]["file"])
        with open(files_path, "r") as f:
            files = f.read().strip().splitlines()
            files = [f.split()[0] for f in files]
 #############################################################           
          #  file_list=[]
          #  for d in idx_list[:100]:
          #      file_list.append(files[d])
 ######################## vergeet niet file_list te veranderen! #####################################           
            files = [os.path.join(self.img_root, f) for f in files]
            

        return files, labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root, self.files[idx])
        x = pil_loader(img_name)
        ############### vergeet niet np.asarray weg te halen############
     #   y = torch.from_numpy(np.asarray(self.labels[idx]))
        y = torch.from_numpy(self.labels[idx])
        if self.transform:
             x = self.transform(x)

        return x, y, img_name

class AADBCAM(nn.Module):
    def __init__(self):
        """ A newly defined model architecture that uses ResNet50 as a base,
        but has no fully vonnected layers. Instead, it uses convolutions and
        a global average pool to create Class Activation Maps.
        Lastly, another avreage pool layer creates the score per attribute."""        
        super(AADBCAM, self).__init__()
        
        self.base = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.base.fc = nn.Identity()
        self.base.avgpool = nn.Identity()
        self.cam = nn.Sequential(nn.Conv2d(2048, 12, 1))
        self.score = nn.Sequential(nn.Tanh(),
                                   nn.AdaptiveAvgPool2d(1))
        
    def forward(self, img):
        x = self.base(img)
        maps = self.cam(torch.reshape(x, (len(img), 2048, 7, 7)))
        scores = self.score(maps)        
        
        return maps, scores


# Loss classes adapted from isaaccorley on 30/8/2022 at
# https://github.com/isaaccorley/deep-aesthetics-pytorch/blob/main/torch_aesthetics/losses.py

class RankLoss(nn.Module):
    """y_pred are the predictions from the model, y_true are the labels
    from the dataset (ground truth). Both consist of a tuple containig two 
    tensors.
    This class prepares a target; a tensor in the shape of y_true[0]
    containing values of 1, but will be come -1 where y_true[0] is smaller 
    than y_true[1]. This will create the ranks. This is then put into the 
    standard pytorch margin rnaking loss function. 
    """
    def __init__(self, margin):
        super().__init__()
        self.margin=margin

    def forward(self, y_pred, y_true):
       
        device, dtype = y_pred[0].device, y_pred[0].dtype

        target = torch.ones_like(y_true[0]).to(device).to(dtype)

        # Set indices where y_true1 < y_true2 to -1
        target[y_true[0] < y_true[1]] = -1.0

        return nn.MarginRankingLoss(
            y_pred[0],
            y_pred[1],
            target,
            margin=self.margin
        )


class RegRankLoss(nn.Module):
    """This function calls two loss functions, MSEloss (regression loss)
    and marginRankingLoss. It then adds these two together to
    a combined loss."""
    def __init__(self, margin):
        super().__init__()
        self.reg_loss = nn.MSELoss(reduction="mean")
        self.rank_loss = nn.MarginRankingLoss(margin)
        

    def forward(self, y_pred, y_true):
        
        loss_reg = (
            self.reg_loss(y_pred[0], y_true[0]) +
            self.reg_loss(y_pred[1], y_true[1])
        ) / 2.0
        
        device, dtype = y_pred[0].device, y_pred[0].dtype
        
        # create a target for all the ranks
        target = torch.ones_like(y_true[0]).to(device).to(dtype)

        # Set indices where y_true1 < y_true2 to -1
        target[y_true[0] < y_true[1]] = -1.0
        
        loss_rank = self.rank_loss(y_pred[0], y_pred[1], target)
        loss = loss_reg + loss_rank
        return loss, loss_reg, loss_rank

