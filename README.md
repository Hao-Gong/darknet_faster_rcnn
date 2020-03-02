![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

# Faster_rcnn model added under Darknet #

[fasterrcnn]<br/>
classes=20<br/>
rois_nms_thresh=0.7<br/>
rois_min_area_thresh=16<br/>
train_rcnn_flg=1<br/>
rpn_sample_num=256<br/>
rpn_sample_pos_ratio=0.25<br/>
rpn_iou_pos_thresh=0.7<br/>
rpn_iou_neg_thresh=0.3<br/>
downsample_ratio=16<br/>
anchor_scale=4, 8 ,16,32<br/>
anchor_ratio=0.5,1,2<br/>
train_pre_nms_num=12000<br/>
train_post_nms_num=2000<br/>
test_pre_nms_num=6000<br/>
test_post_nms_num=300<br/>
rois_sample_num=128<br/>
rois_sample_ratio=0.25<br/>
roialign_pooling_height=7<br/>
roialign_pooling_width=7<br/>
pos_iou_thresh=0.5<br/>
neg_iou_thresh_hi=0.5<br/>
neg_iou_thresh_lo=0<br/>

# How to train and test faster_rcnn? #
## train on pretrained resnet50 model ##
./darknet faster_rcnn train cfg/voc.data cfg/resnet50_faster_rcnn.cfg resnet50.weights -pretrain<br/>

Using -pretrain, the model will initialize the rpn and rcnn randomly.<br/>

### imageNet pretrained resnet50 and resnet152 model download link ###
https://pjreddie.com/media/files/resnet50.weights<br/>
https://pjreddie.com/media/files/resnet152.weights<br/>

## continue training the model ##
./darknet faster_rcnn train cfg/voc.data cfg/resnet50_faster_rcnn.cfg backup/resnet50_faster_rcnn.backup<br/>

# Future work #
1. adding the nms_gpu function for the rois nms
2. FPN structure will be added soon
3. adding focal loss function for unbalanced fg and bg 
4. disable the delta and weights_update memory calloc when the only inference using
5. testing the performance on the VOC datasets

Project modified by Hao. March. 2020