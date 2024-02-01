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

    pred_confidence = pred_confidence.reshape((-1, pred_confidence.size(2)))
    ann_confidence = ann_confidence.reshape((-1, ann_confidence.size(2)))
    pred_box = pred_box.reshape((-1, pred_box.size(2)))
    ann_box = ann_box.reshape((-1, ann_box.size(2)))

    indices_obj = torch.where(sum(ann_confidence[:]) >= 1)
    indices_noobj = torch.where(sum(ann_confidence[:]) < 1)

    #For confidence (class labels), use cross entropy (F.cross_entropy)
    #You can try F.binary_cross_entropy and see which loss is better
    L_cls = F.cross_entropy(pred_confidence[indices_obj], target=ann_confidence[indices_obj])
    L_cls += 3 * F.cross_entropy(pred_confidence[indices_noobj], target=ann_confidence[indices_noobj])

    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    L_box = F.smooth_l1_loss(pred_box[indices_obj], target=ann_box[indices_obj])
    return L_cls + L_box


class nonParallelBlock(nn.Module):
    def __init__(self, cin, cout):
        super(nonParallelBlock, self).__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU()
        )
        self.layers2 = nn.Sequential(
            nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers2(x)
        return x
    

class nonParallelStep(nn.Module):
    def __init__(self, channels=(3, 64, 128, 256, 512)):
        super(nonParallelStep, self).__init__()

        self.nonParallelBlocks = nn.ModuleList(
            [nonParallelBlock(channels[i], channels[i+1])
                for i in range(len(channels)-1)])
        
        self.lastBlock = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels[-2]),
            nn.ReLU()
        )

    def forward(self, x):
        for block in self.nonParallelBlocks:
            x = block(x)
        x = self.lastBlock(x)
        return x    

class SSD(nn.Module):

    def __init__(self, class_num = 4):
        super(SSD, self).__init__()
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        
        self.nonParallelStep = nonParallelStep()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv10to5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv5to3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv3to1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv_dim_reduce = nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0)
        self.br_conv_dim_reduce = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        # x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
        
        x = self.nonParallelStep(x)
        out_bbox = []
        out_conf = []

        out_10_bbox = self.br_conv_dim_reduce(x)
        out_bbox.append(out_10_bbox.reshape((out_10_bbox.size(0), out_10_bbox.size(1), -1)))
        out_10_conf = self.br_conv_dim_reduce(x)
        out_conf.append(out_10_conf.reshape((out_10_conf.size(0), out_10_conf.size(1), -1)))

        x = self.conv1x1(x)
        x = self.conv10to5(x)

        out_5_bbox = self.br_conv_dim_reduce(x)
        out_bbox.append(out_5_bbox.reshape((out_5_bbox.size(0), out_5_bbox.size(1), -1)))
        out_5_conf = self.br_conv_dim_reduce(x)
        out_conf.append(out_5_conf.reshape((out_5_conf.size(0), out_5_conf.size(1), -1)))

        x = self.conv1x1(x)
        x = self.conv5to3(x)

        out_3_bbox = self.br_conv_dim_reduce(x)
        out_bbox.append(out_3_bbox.reshape((out_3_bbox.size(0), out_3_bbox.size(1), -1)))
        out_3_conf = self.br_conv_dim_reduce(x)
        out_conf.append(out_3_conf.reshape((out_3_conf.size(0), out_3_conf.size(1), -1)))

        x = self.conv1x1(x)
        x = self.conv3to1(x)

        out_1_bbox = self.conv_dim_reduce(x)
        out_bbox.append(out_1_bbox.reshape((out_1_bbox.size(0), out_1_bbox.size(1), -1)))
        out_1_conf = self.conv_dim_reduce(x)
        out_conf.append(out_1_conf.reshape((out_1_conf.size(0), out_1_conf.size(1), -1)))

        # concat
        bboxes = torch.cat(out_bbox, dim=2)
        bboxes = bboxes.permute(0,2,1).reshape(bboxes.size(0), -1, self.class_num)
        confidence = torch.cat(out_conf, dim=2)
        confidence = confidence.permute(0,2,1).reshape(confidence.size(0), -1, self.class_num)
        #should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        
        return confidence,bboxes
