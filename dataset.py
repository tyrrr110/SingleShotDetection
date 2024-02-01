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
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image

#generate default bounding boxes
def default_box_generator(layers=[10,5,3,1], large_scale=[0.2,0.4,0.6,0.8], small_scale=[0.1,0.3,0.5,0.7]):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    #create an numpy array "boxes" to store default bounding boxes
    #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    #the second dimension 4 means each cell has 4 default bounding boxes.
    #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    box_num = 4*(10*10+5*5+3*3+1*1)

    boxes = np.empty((10*10+5*5+3*3+1*1, 4, 8), np.float32)
    idx = 0
    for l in range(len(layers)):
        for y in range(layers[l]):
            for x in range(layers[l]):
                x_center, y_center = (x+0.5)/layers[l], (y+0.5)/layers[l]
                # box 0
                box_width, box_height = small_scale[l], small_scale[l]
                boxes[idx][0][0], boxes[idx][0][1] = x_center, y_center
                boxes[idx][0][2], boxes[idx][0][3] = box_width, box_height
                boxes[idx][0][4], boxes[idx][0][5] = max(x_center - (box_width/2), 0), max(y_center - (box_height/2), 0) #x_min, y_min
                boxes[idx][0][6], boxes[idx][0][7] = min(x_center + (box_width/2), layers[l]), min(y_center + (box_height/2), layers[l]) #x_max, y_max
                # box 1
                box_width, box_height = large_scale[l], large_scale[l]
                boxes[idx][1][0], boxes[idx][1][1] = x_center, y_center
                boxes[idx][1][2], boxes[idx][1][3] = box_width, box_height
                boxes[idx][1][4], boxes[idx][1][5] = max(x_center - (box_width/2), 0), max(y_center - (box_height/2), 0) #x_min, y_min
                boxes[idx][1][6], boxes[idx][1][7] = min(x_center + (box_width/2), layers[l]), min(y_center + (box_height/2), layers[l]) #x_max, y_max
                # box 2
                box_width, box_height = large_scale[l]*np.sqrt(2), large_scale[l]/np.sqrt(2)
                boxes[idx][2][0], boxes[idx][2][1] = x_center, y_center
                boxes[idx][2][2], boxes[idx][2][3] = box_width, box_height
                boxes[idx][2][4], boxes[idx][2][5] = max(x_center - (box_width/2), 0), max(y_center - (box_height/2), 0) #x_min, y_min
                boxes[idx][2][6], boxes[idx][2][7] = min(x_center + (box_width/2), layers[l]), min(y_center + (box_height/2), layers[l]) #x_max, y_max
                # box 3
                box_width, box_height = large_scale[l]/np.sqrt(2), large_scale[l]*np.sqrt(2)
                boxes[idx][3][0], boxes[idx][3][1] = x_center, y_center
                boxes[idx][3][2], boxes[idx][3][3] = box_width, box_height
                boxes[idx][3][4], boxes[idx][3][5] = max(x_center - (box_width/2), 0), max(y_center - (box_height/2), 0) #x_min, y_min
                boxes[idx][3][6], boxes[idx][3][7] = min(x_center + (box_width/2), layers[l]), min(y_center + (box_height/2), layers[l]) #x_max, y_max
                # print(x+y*l) # index
                idx += 1
    boxes = boxes.reshape(box_num, -1)
    return boxes


#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)



def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min,y_min,x_max,y_max)
    
    ious_true = ious>threshold
    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    for i in range(len(ious)):
        if ious_true[i]:
            gx, gy = (x_min + x_max)/2.0, (y_min + y_max)/2.0
            gw, gh = (x_max - x_min), (y_max - y_min)
            tx, ty = (gx - boxs_default[i][0])/boxs_default[i][2], (gy - boxs_default[i][1])/boxs_default[i][3]
            tw, th = np.log(gw/boxs_default[i][2]), np.log(gh/boxs_default[i][3])
            ann_box[i][:] = [tx,ty,tw,th]
            ann_confidence[i][-1] = 0 
            ann_confidence[i][int(cat_id)] = 1 
    
    ious_true = np.argmax(ious)
    #make sure at least one default bounding box is used
    #update ann_box and ann_confidence (do the same thing as above)
    gx, gy = (x_min + x_max)/2.0, (y_min + y_max)/2.0
    gw, gh = (x_max - x_min), (y_max - y_min)
    tx, ty = (gx - boxs_default[ious_true][0])/boxs_default[ious_true][2], (gy - boxs_default[ious_true][1])/boxs_default[ious_true][3]
    tw, th = np.log(gw/boxs_default[ious_true][2]), np.log(gh/boxs_default[ious_true][3])
    ann_box[ious_true][:] = [tx,ty,tw,th]
    ann_confidence[ious_true][-1] = 0 
    ann_confidence[ious_true][int(cat_id)] = 1 


class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, test = False, image_size=320):
        self.train = train
        self.test = test
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size
        
        #you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train
        img_train, img_validation = train_test_split(self.img_names, test_size=0.1, random_state=42)
        if self.test:
            pass
        elif self.train:
            self.img_names = img_train
        else:
            self.img_names = img_validation
        
        # transforms
        self.basic_transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.augmentation_transform = transforms.Compose([
            transforms.RandomRotation(20), # p = 20%
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomResizedCrop((320, 320), scale=(0.1, 0.5), ratio=(0.75, 1.25)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        
        img_name = self.imgdir+'/'+self.img_names[index]
        ann_name = self.anndir+'/'+self.img_names[index][:-3]+"txt"
        
        #1. prepare the image [3,320,320], by reading image "img_name" first.
        image = Image.open(img_name)
        # ? H, W before resized ?
        H, W = np.array(image).shape[:-1]

        # TRAIN
        if not self.test:
            #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
            anns = np.loadtxt(ann_name, dtype=np.float32)
            #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
            if len(anns.shape) == 1:
                class_id = anns[0]
                x_min, y_min, x_max, y_max = (anns[1])/W, (anns[2])/H, (anns[1]+anns[3])/W, (anns[2]+anns[4])/H
                match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
            else:
                for ann in anns:
                    class_id = ann[0]
                    x_min, y_min, x_max, y_max = (ann[1])/W, (ann[2])/H, (ann[1]+ann[3])/W, (ann[2]+ann[4])/H
                    match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
            #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
            if self.train:
                image = self.augmentation_transform(image)
            else:
                image = self.basic_transform(image)

        # TEST 
        else:
            image = self.basic_transform(image)
        
        #to use function "match":
        #match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        #where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        
        #note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        #For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        
        return image, ann_box, ann_confidence
