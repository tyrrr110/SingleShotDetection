import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F


def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]

    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.

    B, num_box, num_cat = pred_confidence.shape
    pred_confidence = pred_confidence.reshape((B * num_box, num_cat))
    ann_confidence = ann_confidence.reshape((B * num_box, num_cat))
    pred_box = pred_box.reshape((B * num_box, 4))
    ann_box = ann_box.reshape((B * num_box, 4))

    indices_obj = torch.where(ann_confidence[:, -1] != 1)
    indices_noobj = torch.where(ann_confidence[:, -1] == 1)

    obj_ann_conf = ann_confidence[indices_obj]
    obj_pred_conf = pred_confidence[indices_obj]
    obj_ann_box = ann_box[indices_obj]
    obj_pred_box = pred_box[indices_obj]
    noobj_ann_conf = ann_confidence[indices_noobj]
    noobj_pred_conf = pred_confidence[indices_noobj]    

    #For confidence (class labels), use cross entropy (F.cross_entropy)
    #You can try F.binary_cross_entropy and see which loss is better
    L_cls = F.cross_entropy(obj_pred_conf, target=obj_ann_conf) 
    L_cls += 3 * F.cross_entropy(noobj_pred_conf, target=noobj_ann_conf)

    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    L_box = F.smooth_l1_loss(obj_pred_box, target=obj_ann_box)
    return L_cls + L_box
  

class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()

        self.class_num = class_num

        self.conv_layers_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )  

        self.conv_layers_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv_layers_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.conv_layers_4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv_res5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv_res3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv_res1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv_res10_o1 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_res10_o2 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_res5_o1 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_res5_o2 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_res3_o1 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_res3_o2 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_res1_o1 = nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_res1_o2 = nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv_layers_1(x)
        x = self.conv_layers_2(x)
        x = self.conv_layers_3(x)

        res10 = self.conv_layers_4(x)
        res5 = self.conv_res5(res10)
        res3 = self.conv_res3(res5)
        res1 = self.conv_res1(res3)

        B, _, H, W = res10.shape
        res10_o1 = self.conv_res10_o1(res10).view(B, 16, H*W)
        res10_o2 = self.conv_res10_o2(res10).view(B, 16, H*W)

        B, _, H, W = res5.shape
        res5_o1 = self.conv_res5_o1(res5).view(B, 16, H*W)
        res5_o2 = self.conv_res5_o2(res5).view(B, 16, H*W)

        B, _, H, W = res3.shape
        res3_o1 = self.conv_res3_o1(res3)
        res3_o1 = res3_o1.view(B, 16, H*W)
        res3_o2 = self.conv_res3_o2(res3).view(B, 16, H*W)

        B, _, H, W = res1.shape
        res1_o1 = self.conv_res1_o1(res1).view(B, 16, H*W)
        res1_o2 = self.conv_res1_o2(res1).view(B, 16, H*W)

        out_bbox = torch.concat([res10_o1, res5_o1, res3_o1, res1_o1], axis=2)
        B, _, _ = out_bbox.shape
        out_bbox = out_bbox.permute(0, 2, 1)
        bboxes = out_bbox.reshape((B, 540, 4))

        out_conf = torch.concat([res10_o2, res5_o2, res3_o2, res1_o2], axis=2)
        B, _, _ = out_conf.shape
        out_conf = out_conf.permute(0, 2, 1)
        out_conf = out_conf.reshape((B, 540, 4))
        confidence = torch.softmax(out_conf, dim=2)
        
        return confidence,bboxes
