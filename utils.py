import numpy as np
import cv2
from dataset import iou


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    # image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image = ((np.transpose(image_, (1,2,0)) + 1) * 127.5).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #image1: draw ground truth bounding boxes on image1
                gt_x, gt_y =  boxs_default[i][2] * ann_box[i][0] + boxs_default[i][0], boxs_default[i][3] * ann_box[i][1] + boxs_default[i][1]
                gt_w, gt_h = boxs_default[i][2] * np.exp(ann_box[i][2]), boxs_default[i][3] * np.exp(ann_box[i][3])
                gt_start, gt_end = (int((gt_x - gt_w/2.0) * 320), int((gt_y - gt_h/2.0) * 320)), (int((gt_x + gt_w/2.0) * 320), int((gt_y + gt_h/2.0) * 320)) #[xmin, ymin, xmax, ymax]
                cv2.rectangle(image1, gt_start, gt_end, colors[j], thickness=2)
                
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                cv2.rectangle(image2, (int(boxs_default[i][4] * 320), int(boxs_default[i][5] * 320)), (int(boxs_default[i][6] * 320), int(boxs_default[i][7] * 320)), colors[j], thickness=2)
                
                #you can use cv2.rectangle as follows:
                #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                #end_point = (x2, y2) #bottom right corner
                #color = colors[j] #use red green blue to represent different classes
                #thickness = 2
                #cv2.rectangle(image?, start_point, end_point, color, thickness)
    
    #pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.5:
                #image3: draw network-predicted bounding boxes on image3
                pred_x, pred_y =  boxs_default[i][2] * pred_box[i][0] + boxs_default[i][0], boxs_default[i][3] * pred_box[i][1] + boxs_default[i][1]
                pred_w, pred_h = boxs_default[i][2] * np.exp(pred_box[i][2]), boxs_default[i][3] * np.exp(pred_box[i][3])
                pred_start, pred_end = (int((pred_x - pred_w/2.0) * 320), int((pred_y - pred_h/2.0) * 320)), (int((pred_x + pred_w/2.0) * 320), int((pred_y + pred_h/2.0) * 320)) #[xmin, ymin, xmax, ymax]
                cv2.rectangle(image3, pred_start, pred_end, colors[j], thickness=2)

                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                cv2.rectangle(image4, (int(boxs_default[i][4] * 320), int(boxs_default[i][5] * 320)), (int(boxs_default[i][6] * 320), int(boxs_default[i][7] * 320)), colors[j], thickness=2)
    
    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    # cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    # cv2.waitKey(1)

    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.
    cv2.imwrite(windowname+".png",image)


def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.5, threshold=0.5):
    #TODO: non maximum suppression
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    pass

def generate_mAP():
    #TODO: Generate mAP
    pass








