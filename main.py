import argparse
import os
import numpy as np
import time
import cv2

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

from dataset import *
from model import *
from utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    #please google how to use argparse
    #a short intro:
    #to train: python main.py
    #to test:  python main.py --test


    class_num = 4 #cat dog person background

    num_epochs = 100
    batch_size = 32


    boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])


    #Create network
    network = SSD(class_num)
    network.cuda()
    cudnn.benchmark = True


    if not args.test:
        dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = True, test = False, image_size=320)
        dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, test = False, image_size=320)
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
        
        optimizer = optim.Adam(network.parameters(), lr = 1e-4)
        #feel free to try other optimizers and parameters.
        
        start_time = time.time()

        for epoch in range(num_epochs):
            #TRAINING
            network.train()

            avg_loss = 0
            avg_count = 0
            for i, data in enumerate(dataloader, 0):
                images_, ann_box_, ann_confidence_ = data
                images = images_.cuda()
                ann_box = ann_box_.cuda()
                ann_confidence = ann_confidence_.cuda()

                optimizer.zero_grad()
                pred_confidence, pred_box = network(images)
                loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
                loss_net.backward()
                optimizer.step()
                
                avg_loss += loss_net.data
                avg_count += 1

            print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))
            
            #visualize
            pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
            pred_box_ = pred_box[0].detach().cpu().numpy()
            visualize_pred("train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
            
            
            #VALIDATION
            network.eval()
            
            # split the dataset into 90% training and 10% validation
            # use the training set to train and the validation set to evaluate
            
            for i, data in enumerate(dataloader_test, 0):
                images_, ann_box_, ann_confidence_ = data
                images = images_.cuda()
                ann_box = ann_box_.cuda()
                ann_confidence = ann_confidence_.cuda()

                pred_confidence, pred_box = network(images)
                
                pred_confidence_ = pred_confidence.detach().cpu().numpy()
                pred_box_ = pred_box.detach().cpu().numpy()
                
                #optional: implement a function to accumulate precision and recall to compute mAP or F1.
                #update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,precision_,recall_,thres)
            
            #visualize
            pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
            pred_box_ = pred_box[0].detach().cpu().numpy()
            visualize_pred("val", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
            
            #optional: compute F1
            #F1score = 2*precision*recall/np.maximum(precision+recall,1e-8)
            #print(F1score)
            
            #save weights
            if epoch%10==9:
                #save last network
                print('saving net...')
                torch.save(network.state_dict(), 'network.pth')


    else:
        #TEST
        dataset_test = COCO("data/test/images/", "data/test/annotations/", class_num, boxs_default, train = False, test = True, image_size=320)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
        network.load_state_dict(torch.load('network.pth'))
        network.eval()
        
        for i, data in enumerate(dataloader_test, 0):
            images_, ann_box_, ann_confidence_ = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            pred_confidence, pred_box = network(images)

            pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
            pred_box_ = pred_box[0].detach().cpu().numpy()
            
            #pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
            
            #save predicted bounding boxes and classes to a txt file.
            #you will need to submit those files for grading this assignment
            save_results_to_txt('result.txt', pred_confidence_, pred_box_)
            
            visualize_pred("test", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
            cv2.waitKey(1000)

def save_results_to_txt(file_path, pred_confidence, pred_box, threshold=0.5):
    # Assuming pred_confidence and pred_box are numpy arrays
    # pred_confidence shape: (num_boxes, num_classes)
    # pred_box shape: (num_boxes, 4)

    # Apply threshold to filter out low-confidence predictions
    mask = pred_confidence[:, :-1].max(axis=1) > threshold
    pred_confidence = pred_confidence[mask]
    pred_box = pred_box[mask]

    with open(file_path, 'w') as file:
        for i in range(len(pred_confidence)):
            class_index = np.argmax(pred_confidence[i, :-1])  # Ignore background class (index 3)
            confidence = pred_confidence[i, class_index]
            bbox = pred_box[i]
            line = f"{class_index} {confidence:.4f} {bbox[0]:.4f} {bbox[1]:.4f} {bbox[2]:.4f} {bbox[3]:.4f}\n"
            file.write(line)

if __name__ == '__main__':
    main()