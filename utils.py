import numpy as np
import cv2
from dataset import iou
import copy
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch

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


def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.3, threshold=0.6):
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.    
    
    abs_box = np.zeros((len(box_),8))
    px, py, pw, ph = boxs_default[:,0], boxs_default[:,1], boxs_default[:,2], boxs_default[:,3]
    #convert box_ to absolute box_
    abs_box[:,0] = box_[:,0]*pw + px
    abs_box[:,1] = box_[:,1]*ph + py
    abs_box[:,2] = pw*np.exp(box_[:,2])
    abs_box[:,3] = ph*np.exp(box_[:,3])
    abs_box[:,4] = abs_box[:,0] - abs_box[:,2]/2.0   # min_x
    abs_box[:,5] = abs_box[:,1] - abs_box[:,3]/2.0   # min_y
    abs_box[:,6] = abs_box[:,0] + abs_box[:,2]/2.0   # min_x
    abs_box[:,7] = abs_box[:,1] + abs_box[:,3]/2.0   # min_y
    
    non_suppressed_boxes = set()
    confidence_suppressed = copy.deepcopy(confidence_)
    while True:
        # bounding box in to_check list with the highest probability in class cat, dog or person
        high_prob_boxes = np.argmax(confidence_, axis=0)[:-1]
        if confidence_[high_prob_boxes[0],0] < threshold and confidence_[high_prob_boxes[1],1] < threshold and confidence_[high_prob_boxes[2],2] < threshold:
            break
        for j, i in enumerate(high_prob_boxes):
            if confidence_[i][j] > threshold:
                non_suppressed_boxes.add(i)
                confidence_[i][j] = 0
                i_box = abs_box[i]

                # suppress boxes with high IOU with i_box
                ious = iou(abs_box, i_box[4], i_box[5], i_box[6], i_box[7])
                ious = ious > overlap
                for o in range(len(ious)):
                    if ious[o]:
                        confidence_[o][:] = [0, 0, 0, 1]       

    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    for i in range(len(confidence_suppressed)):
        if i not in non_suppressed_boxes:
            confidence_suppressed[i][:] = [0, 0, 0, 1]
    with open("debugging_nms", "a") as f:
        idx = list(non_suppressed_boxes)
        f.write(np.array_str(confidence_suppressed[idx]))

    with open("nms_box", "a") as f:
        idx = list(non_suppressed_boxes)
        f.write(np.array_str(abs_box[idx]))

    return confidence_suppressed


def generate_mAP(pred_confidence, pred_box, ann_confidence, ann_box, image_, num_classes=4, class_names=["cat", "dog", "person", "background"]):
    #input:
    #- pred_confidence: Predicted class labels from SSD, [num_of_boxes, num_classes]
    #- pred_box: Predicted bounding boxes from SSD, [num_of_boxes, 4]
    #- ann_confidence: Ground truth class labels, [num_of_boxes, num_classes]
    # - ann_box: Ground truth bounding boxes, [num_of_boxes, 4]
    # - image_: Input image to the network
    # - num_classes: Number of classes (including background).
    # - class_names: List of class names (optional, used for plotting).

    # Returns:
    # - mAP: Mean Average Precision.
    precision_list = []
    recall_list = []
    average_precision_list = []

    for class_index in range(num_classes-1):  # excludes background class
        true_class = ann_confidence[:, class_index]
        pred_scores_class = pred_confidence[:, class_index]
        pred_boxes_class = pred_box[:, class_index * 4: (class_index + 1) * 4]

        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(true_class, pred_scores_class)

        # Compute average precision
        ap = average_precision_score(true_class, pred_scores_class)

        # Store precision, recall, and average precision for the current class
        precision_list.append(precision)
        recall_list.append(recall)
        average_precision_list.append(ap)

        # Plot precision-recall curve for the current class
        if class_names is not None:
            plt.plot(recall, precision, label=f"{class_names[class_index]} (AP = {ap:.2f})")
        else:
            plt.plot(recall, precision, label=f"Class {class_index} (AP = {ap:.2f})")

    # Compute mAP as the mean of average precision across all classes
    mAP = np.mean(average_precision_list)

    # Plot settings
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Mean Average Precision (mAP): {mAP:.4f}")

    return mAP
