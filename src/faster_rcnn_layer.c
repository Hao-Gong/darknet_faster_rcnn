#include "faster_rcnn_layer.h"
#include "convolutional_layer.h"
#include "connected_layer.h"
#include "shortcut_layer.h"
#include "softmax_layer.h"
#include "avgpool_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include "darknet.h"
#include <stdio.h>
#include <time.h>
#include "roialign_pooling.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
// #include <vector>

void generate_anchor_box(layer l){
    int anchor_index=0;

    float h_ratio=l.rpn_bbox_pred->out_h*l.downsample_ratio;
    float w_ratio=l.rpn_bbox_pred->out_w*l.downsample_ratio;
    for(int ratio_index=0;ratio_index<l.anchor_ratio_num;ratio_index++){
        for(int scale_index=0;scale_index<l.anchor_scale_num;scale_index++){
            for(int h=0;h<l.rpn_bbox_pred->out_h;h++){
                for(int w=0;w<l.rpn_bbox_pred->out_w;w++){
                    float centre_x=w*l.downsample_ratio+l.downsample_ratio/2;
                    float centre_y=h*l.downsample_ratio+l.downsample_ratio/2;
                    float height=l.downsample_ratio*l.anchor_scale[scale_index]*sqrt(l.anchor_ratio[ratio_index]);
                    float width=l.downsample_ratio*l.anchor_scale[scale_index]*sqrt(1/l.anchor_ratio[ratio_index]);

                    l.anchor_space[anchor_index*5]=centre_x/w_ratio;
                    l.anchor_space[anchor_index*5+1]=centre_y/h_ratio;
                    l.anchor_space[anchor_index*5+2]=width/w_ratio;
                    l.anchor_space[anchor_index*5+3]=height/h_ratio;

                    if(centre_x-width/2>=0&&centre_y-height/2>=0&&centre_x+width/2<=l.rpn_bbox_pred->out_w*l.downsample_ratio&&centre_y+height/2<=l.rpn_bbox_pred->out_h*l.downsample_ratio){
                        l.anchor_space[anchor_index*5+4]=1;
                    }else{
                        l.anchor_space[anchor_index*5+4]=-1;
                    }
                    anchor_index++;
                }
            }
        }
    }

}

float smooth_l1_loss(float scale,float t,float p){
    float delta =scale*(t-p);
    if(delta>scale){
        return scale;
    }else if(delta<-scale)
    {
        return -scale;
    }else{
        return delta;
    }
}

void init_convolutional_layer_weights(layer l,float scale){
    for(int i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    #ifdef GPU
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
    #endif
}
void init_connectec_layer_weights(layer l,float scale){
    for(int i = 0; i < l.outputs*l.inputs; ++i){
        l.weights[i] = scale*rand_uniform(-1, 1);
    }
    #ifdef GPU
    cuda_push_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    #endif
}

layer make_faster_rcnn_layer(int batch, int w, int h, int n, int classes,int adam,faster_rcnn_params f_param)
{
    int i;
    layer l = {0};
    l.type = FASTER_RCNN;

    l.n = n;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.classes=classes;
    l.anchor_scale_num=f_param.anchor_scale_num;
    l.anchor_ratio_num=f_param.anchor_ratio_num;
    // Resnet downsample ratio is 16
    l.downsample_ratio=f_param.downsample_ratio;
    l.cost = calloc(1, sizeof(float));
    l.anchor_scale=calloc(l.anchor_scale_num, sizeof(float));
    l.anchor_ratio=calloc(l.anchor_ratio_num, sizeof(float));
    for(i=0;i<l.anchor_scale_num;i++){
        l.anchor_scale[i]=f_param.anchor_scale[i];
        // printf("scale:%d %f\n",i,l.anchor_scale[i]);
    }
    for(i=0;i<l.anchor_ratio_num;i++){
        l.anchor_ratio[i]=f_param.anchor_ratio[i];
        // printf("ratio:%d %f\n",i,l.anchor_ratio[i]);
    }

    l.rois_nms_thresh=f_param.rois_nms_thresh;
    l.rois_min_area_thresh=f_param.rois_min_area_thresh;
    l.train_rcnn_flg=f_param.train_rcnn_flg;
    l.rpn_sample_num=f_param.rpn_sample_num;
    l.rpn_sample_pos_ratio=f_param.rpn_sample_pos_ratio;
    l.rpn_sample_pos_max_num=l.rpn_sample_pos_ratio*l.rpn_sample_num;
    l.rpn_iou_neg_thresh=f_param.rpn_iou_neg_thresh;
    l.rpn_iou_pos_thresh=f_param.rpn_iou_pos_thresh;
    // int rpn_sample_neg_max_num;
    l.train_pre_nms_num=f_param.train_pre_nms_num;
    l.train_post_nms_num=f_param.train_post_nms_num;
    l.test_pre_nms_num=f_param.test_pre_nms_num;
    l.test_post_nms_num=f_param.test_post_nms_num;
    l.rois_sample_num=f_param.rois_sample_num;
    l.rois_sample_ratio=f_param.rois_sample_ratio;
    l.roialign_pooling_height=f_param.roialign_pooling_height;
    l.roialign_pooling_width=f_param.roialign_pooling_width;
    l.pos_iou_thresh=f_param.pos_iou_thresh;
    l.neg_iou_thresh_hi=f_param.neg_iou_thresh_hi;
    l.neg_iou_thresh_lo=f_param.neg_iou_thresh_lo;

    l.rpn_conv1= malloc(sizeof(layer));
    l.rpn_cls_score= malloc(sizeof(layer));
    l.rpn_bbox_pred= malloc(sizeof(layer));

    l.roi_head_layers_num=4;
    l.roi_head_layers= malloc(l.roi_head_layers_num*sizeof(layer));
    l.roi_cls_score= malloc(sizeof(layer));
    l.roi_bbox_pred= malloc(sizeof(layer));
    l.rois_cls_score_softmax= malloc(sizeof(layer));

    fprintf(stderr,"#####rpn#####\n");
    int rpn_id=0;
    fprintf(stderr,"rpn:%d    ",rpn_id);
    *(l.rpn_conv1)=make_convolutional_layer(batch, h, w, l.n, 512, 1, 3, 1, 1, RELU, 1, 0,0, adam);
    fprintf(stderr,"rpn:%d    ",++rpn_id);
    *(l.rpn_cls_score)=make_convolutional_layer(batch, l.rpn_conv1->out_h,  l.rpn_conv1->out_w, 512,l.anchor_scale_num*l.anchor_ratio_num, 1, 1, 1, 0, LOGISTIC, 0, 0,0, adam);
    fprintf(stderr,"rpn:%d    ",++rpn_id);
    *(l.rpn_bbox_pred)=make_convolutional_layer(batch,  l.rpn_conv1->out_h,  l.rpn_conv1->out_w, 512, l.anchor_scale_num*l.anchor_ratio_num*4, 1, 1, 1, 0, LINEAR, 0, 0,0, adam);
    init_convolutional_layer_weights(*(l.rpn_conv1),0.01);
    init_convolutional_layer_weights(*(l.rpn_bbox_pred),0.01);
    init_convolutional_layer_weights(*(l.rpn_cls_score),0.01);
    l.rpn_anchor_box_num=l.rpn_bbox_pred->out_h*l.rpn_bbox_pred->out_w*l.anchor_ratio_num*l.anchor_scale_num;
    l.anchor_space=calloc(l.rpn_anchor_box_num*5, sizeof(float));
    generate_anchor_box(l);

    fprintf(stderr,"#####rcnn#####\n");
    int rcnn_id=0;
    fprintf(stderr,"rcnn:%d    roi_align_pooling size:  %dx%dx%d\n",rcnn_id, l.roialign_pooling_height,  l.roialign_pooling_width, l.n);
    fprintf(stderr,"rcnn:%d    ",++rcnn_id);
    l.roi_head_layers[0]=make_convolutional_layer(l.rois_sample_num*batch,  l.roialign_pooling_height,  l.roialign_pooling_width, l.n, 512, 1, 1, 1, 0, LEAKY, 1, 0,0, adam);
    fprintf(stderr,"rcnn:%d    ",++rcnn_id);
    l.roi_head_layers[1]=make_convolutional_layer(l.rois_sample_num*batch,  l.roi_head_layers[0].out_h,  l.roi_head_layers[0].out_w, l.roi_head_layers[0].out_c, 512, 1, 3, 1, 1, LEAKY, 1, 0,0, adam);
    fprintf(stderr,"rcnn:%d    ",++rcnn_id);
    l.roi_head_layers[2]=make_convolutional_layer(l.rois_sample_num*batch,  l.roi_head_layers[1].out_h,  l.roi_head_layers[1].out_w, l.roi_head_layers[1].out_c, 2048, 1, 1, 1, 0, LINEAR, 1, 0,0, adam);
    // fprintf(stderr,"rcnn:%d    ",++rcnn_id);
    // l.roi_head_layers[3]= make_shortcut_layer2(l.rois_sample_num*batch, rcnn_id-4, l.roi_head_layers[2].out_w,  l.roi_head_layers[2].out_h, l.roi_head_layers[2].out_c,  l.roialign_pooling_width,  l.roialign_pooling_height, l.n,LEAKY,1, 1);
    // fprintf(stderr,"rcnn:%d    ",++rcnn_id);
    // l.roi_head_layers[4]=make_convolutional_layer(l.rois_sample_num*batch, l.roi_head_layers[3].out_h,  l.roi_head_layers[3].out_w, l.roi_head_layers[3].out_c,  512, 1, 1, 1, 0, LEAKY, 1, 0,0, adam);
    // fprintf(stderr,"rcnn:%d    ",++rcnn_id);
    // l.roi_head_layers[5]=make_convolutional_layer(l.rois_sample_num*batch,  l.roi_head_layers[4].out_h,  l.roi_head_layers[4].out_w, l.roi_head_layers[4].out_c, 512, 1, 3, 1, 1, LEAKY, 1, 0,0, adam);
    // fprintf(stderr,"rcnn:%d    ",++rcnn_id);
    // l.roi_head_layers[6]=make_convolutional_layer(l.rois_sample_num*batch,  l.roi_head_layers[5].out_h,  l.roi_head_layers[5].out_w, l.roi_head_layers[5].out_c, 2048, 1, 1, 1, 0, LINEAR, 1, 0,0, adam);
    // fprintf(stderr,"rcnn:%d    ",++rcnn_id);
    // l.roi_head_layers[7]= make_shortcut_layer2(l.rois_sample_num*batch, rcnn_id-4, l.roi_head_layers[6].out_w,  l.roi_head_layers[6].out_h, l.roi_head_layers[6].out_c,  l.roi_head_layers[3].out_w,  l.roi_head_layers[3].out_h, l.roi_head_layers[3].out_c, LEAKY,1, 1);
    // fprintf(stderr,"rcnn:%d    ",++rcnn_id);
    // l.roi_head_layers[8]=make_convolutional_layer(l.rois_sample_num*batch, l.roi_head_layers[7].out_h,  l.roi_head_layers[7].out_w, l.roi_head_layers[7].out_c,  512, 1, 1, 1, 0, LEAKY, 1, 0,0, adam);
    // fprintf(stderr,"rcnn:%d    ",++rcnn_id);
    // l.roi_head_layers[9]=make_convolutional_layer(l.rois_sample_num*batch,  l.roi_head_layers[8].out_h,  l.roi_head_layers[8].out_w, l.roi_head_layers[8].out_c, 512, 1, 3, 1, 1, LEAKY, 1, 0,0, adam);
    // fprintf(stderr,"rcnn:%d    ",++rcnn_id);
    // l.roi_head_layers[10]=make_convolutional_layer(l.rois_sample_num*batch,  l.roi_head_layers[9].out_h,  l.roi_head_layers[9].out_w, l.roi_head_layers[9].out_c, 2048, 1, 1, 1, 0, LINEAR, 1, 0,0, adam);
    // fprintf(stderr,"rcnn:%d    ",++rcnn_id);
    // l.roi_head_layers[11]= make_shortcut_layer2(l.rois_sample_num*batch, rcnn_id-4, l.roi_head_layers[10].out_w,  l.roi_head_layers[10].out_h, l.roi_head_layers[10].out_c,  l.roi_head_layers[7].out_w,  l.roi_head_layers[7].out_h, l.roi_head_layers[7].out_c, LEAKY,1, 1);
    fprintf(stderr,"rcnn:%d    ",++rcnn_id);
    l.roi_head_layers[rcnn_id-1]=make_avgpool_layer(l.rois_sample_num*batch,  l.roi_head_layers[rcnn_id-2].out_h,  l.roi_head_layers[rcnn_id-2].out_w, l.roi_head_layers[rcnn_id-2].out_c);

    fprintf(stderr,"rcnn:%d    ",++rcnn_id);
    *(l.roi_cls_score)= make_connected_layer(l.rois_sample_num*batch,  l.roi_head_layers[l.roi_head_layers_num-1].out_h*l.roi_head_layers[l.roi_head_layers_num-1].out_w*l.roi_head_layers[l.roi_head_layers_num-1].out_c, l.classes+1,  LINEAR, 0, adam);
    fprintf(stderr,"rcnn:%d    ",++rcnn_id);
    *(l.roi_bbox_pred)= make_connected_layer(l.rois_sample_num*batch, l.roi_head_layers[l.roi_head_layers_num-1].out_h*l.roi_head_layers[l.roi_head_layers_num-1].out_w*l.roi_head_layers[l.roi_head_layers_num-1].out_c, (l.classes+1)*4,  LINEAR, 0, adam);
    fprintf(stderr,"rcnn:%d    ",++rcnn_id);
    *(l.rois_cls_score_softmax)= make_softmax_layer(l.rois_sample_num*batch, l.classes+1, 1);
    l.rois_cls_score_softmax->temperature = 1;
    l.rois_cls_score_softmax->w = 1;
    l.rois_cls_score_softmax->h = 1;
    l.rois_cls_score_softmax->c = l.classes+1;
    l.rois_cls_score_softmax->spatial = 0;
    l.rois_cls_score_softmax->noloss =  0;

    init_connectec_layer_weights(*(l.roi_bbox_pred),0.001);
    init_connectec_layer_weights(*(l.roi_cls_score),0.01);

    l.rois=calloc(l.batch*l.rois_sample_num*5, sizeof(float));
    l.rois_target=calloc(l.batch*l.rois_sample_num*4, sizeof(float));
    l.rois_target_label=calloc(l.batch*l.rois_sample_num*(classes+1), sizeof(float));

    l.roialign_pooling_output=calloc( l.batch*l.n*l.roialign_pooling_height*l.roialign_pooling_width*l.rois_sample_num, sizeof(float));
    l.roialign_pooling_output_delta=calloc( l.batch*l.n*l.roialign_pooling_height*l.roialign_pooling_width*l.rois_sample_num, sizeof(float));
    l.conv1_delta_add=calloc( l.batch*l.rpn_conv1->outputs, sizeof(float));

    l.roi_head_layers_delta_add=calloc(l.roi_head_layers[l.roi_head_layers_num-1].batch* l.roi_head_layers[l.roi_head_layers_num-1].outputs, sizeof(float));

    l.forward = forward_faster_rcnn_layer;
    l.backward = backward_faster_rcnn_layer;
    l.update = update_faster_rcnn_layer;

    #ifdef GPU
        l.forward_gpu = forward_faster_rcnn_layer_gpu;
        l.backward_gpu = backward_faster_rcnn_layer_gpu;
        l.update_gpu = update_faster_rcnn_layer_gpu;
        l.conv1_delta_add_gpu=cuda_make_array(l.conv1_delta_add, l.batch*l.rpn_conv1->outputs);
        l.rois_gpu=cuda_make_array(l.rois, l.batch*l.rois_sample_num*5);
        l.roialign_pooling_output_gpu=cuda_make_array(l.roialign_pooling_output,l.batch*l.n*l.roialign_pooling_height*l.roialign_pooling_width*l.rois_sample_num);
        l.roialign_pooling_output_delta_gpu=cuda_make_array(l.roialign_pooling_output_delta,l.batch*l.n*l.roialign_pooling_height*l.roialign_pooling_width*l.rois_sample_num);
        l.rois_target_label_gpu=cuda_make_array(l.rois_target_label, l.batch*l.rois_sample_num*(classes+1));
        // l.rois_net_delta_add_gpu=cuda_make_array(l.rois_net_delta_add, l.batch*l.n*l.h*l.w);
        l.roi_head_layers_delta_add_gpu=cuda_make_array(l.roi_head_layers_delta_add,l.roi_head_layers[l.roi_head_layers_num-1].batch* l.roi_head_layers[l.roi_head_layers_num-1].outputs);
    #endif
    // printf("***********exit the make_faster_rcnn_layer\n");
    return l;
}

void forward_faster_rcnn_layer(const layer l, network net){
    // printf("########### forward_faster_rcnn_layer\n");
    if(l.rpn_cls_score->delta){
            fill_cpu(l.rpn_cls_score->outputs * l.rpn_cls_score->batch, 0, l.rpn_cls_score->delta, 1);
    }
    if(l.rpn_bbox_pred->delta){
            fill_cpu(l.rpn_bbox_pred->outputs * l.rpn_bbox_pred->batch, 0, l.rpn_bbox_pred->delta, 1);
    }
    if(l.roi_bbox_pred->delta){
        fill_cpu(l.roi_bbox_pred->outputs * l.roi_bbox_pred->batch, 0,l.roi_bbox_pred->delta, 1);
    }
    
#ifndef GPU
    if(!net.train){

        if(l.roialign_pooling_output_delta){
            fill_cpu(l.batch*l.n*l.roialign_pooling_height*l.roialign_pooling_width*l.rois_sample_num,0, l.roialign_pooling_output_delta,1);
        }
        for(int i=0;i<l.roi_head_layers_num;i++){
            if(l.roi_head_layers[i].delta){
                fill_cpu(l.roi_head_layers[i].outputs * l.roi_head_layers[i].batch, 0,l.roi_head_layers[i].delta, 1);
            }
            fill_cpu(l.roi_head_layers[l.roi_head_layers_num-1].batch*l.roi_head_layers[l.roi_head_layers_num-1].outputs, 0,l.roi_head_layers_delta_add, 1);
        }
        if(l.roi_cls_score->delta){
            fill_cpu(l.roi_cls_score->outputs * l.roi_cls_score->batch, 0,l.roi_cls_score->delta, 1);
        }
        if(l.roi_bbox_pred->delta){
            fill_cpu(l.roi_bbox_pred->outputs * l.roi_bbox_pred->batch, 0,l.roi_bbox_pred->delta, 1);
        }
        if(l.rpn_conv1->delta){
                fill_cpu(l.rpn_conv1->outputs * l.rpn_conv1->batch, 0, l.rpn_conv1->delta, 1);
                fill_cpu(l.rpn_conv1->outputs * l.rpn_conv1->batch, 0, l.conv1_delta_add, 1);
        }
    }

    l.rpn_conv1->train=net.train;
    forward_convolutional_layer_pure(*(l.rpn_conv1), net.input,net.workspace);
    l.rpn_cls_score->train=net.train;
    forward_convolutional_layer_pure(*(l.rpn_cls_score), l.rpn_conv1->output,net.workspace);
    l.rpn_bbox_pred->train=net.train;
    forward_convolutional_layer_pure(*(l.rpn_bbox_pred), l.rpn_conv1->output,net.workspace);
#endif


    if(net.train){
        fill_cpu( l.batch*l.rois_sample_num*(l.classes+1), 0, l.rois_target_label, 1);
        fill_cpu( l.batch*l.rois_sample_num*5, 0, l.rois, 1);
        fill_cpu( l.batch*l.rois_sample_num*4, 0, l.rois_target, 1);

    }

    float *truth_roi;
    truth_roi=(float*)calloc(l.roi_bbox_pred->outputs*l.roi_bbox_pred->batch, sizeof(float));

    // #######################################rpn region############################################
    *(l.cost) = 0;
    for(int b=0;b<l.batch;b++){
        if(net.train){
            int bbox_num=0;
                for(int t = 0; t < l.max_boxes; ++t){
                    box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
                    if(!truth.x) break;
                    bbox_num++;
                }
                float *total_iou_cal;
                total_iou_cal=(float*)calloc(l.rpn_anchor_box_num*bbox_num, sizeof(float));
                int *total_iou_label;
                total_iou_label=(int*)calloc(l.rpn_anchor_box_num*bbox_num, sizeof(int));

                float *pos_anchor_index_set;
                pos_anchor_index_set=(float*)calloc(l.rpn_anchor_box_num*bbox_num, sizeof(float));
                int pos_anchor_index_set_num=0;
                float *neg_anchor_index_set;
                neg_anchor_index_set=(float*)calloc(l.rpn_anchor_box_num*bbox_num, sizeof(float));
                int neg_anchor_index_set_num=0;

                for(int t = 0; t < bbox_num; ++t){
                    box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
                    // if(!truth.x) break;
                    float max_iou=0;
                    int max_iou_index=0;
                    for(int box_id=0;box_id<l.rpn_anchor_box_num;box_id++){
                        if(l.anchor_space[box_id*5+4]==-1){
                            total_iou_cal[l.rpn_anchor_box_num*t+box_id]=0;
                            total_iou_label[l.rpn_anchor_box_num*t+box_id]=0;
                        }else{
                            box anchor;

                            anchor.x=l.anchor_space[box_id*5+0];
                            anchor.y=l.anchor_space[box_id*5+1];
                            anchor.w=l.anchor_space[box_id*5+2];
                            anchor.h=l.anchor_space[box_id*5+3];
                            // printf("bbox x:%f y:%f w:%f h:%f network h:%d w:%d c:%d\n", anchor.x,  anchor.y, anchor.w, anchor.h,net.h,net.w,net.c);
                            float _iou=box_iou(anchor, truth);
                            total_iou_cal[l.rpn_anchor_box_num*t+box_id]=_iou;
                            if(_iou>=l.rpn_iou_pos_thresh){
                                total_iou_label[l.rpn_anchor_box_num*t+box_id]=1+t;
                                // printf("_iou>0.7 anchor iou:%f:   box_id:%d\n",total_iou_cal[l.rpn_anchor_box_num*t+box_id],box_id);
                            }else if(_iou<=l.rpn_iou_neg_thresh){
                                total_iou_label[l.rpn_anchor_box_num*t+box_id]=-1-t;
                            }else{
                                total_iou_label[l.rpn_anchor_box_num*t+box_id]=0;
                            }

                            if(max_iou<_iou){
                                max_iou=_iou;
                                max_iou_index=l.rpn_anchor_box_num*t+box_id;
                            }
                        }
                    }
                    total_iou_label[max_iou_index]=1+t;
                    // printf("max iou anchor iou:%f:   box_id:%d\n",total_iou_cal[max_iou_index],max_iou_index);
                }
                for(int box_id=0;box_id<l.rpn_anchor_box_num*bbox_num;box_id++){
                    if(total_iou_label[box_id]>0){
                        pos_anchor_index_set[pos_anchor_index_set_num]=box_id;
                        pos_anchor_index_set_num++;
                    }
                    if(total_iou_label[box_id]<0){
                        neg_anchor_index_set[neg_anchor_index_set_num]=box_id;
                        neg_anchor_index_set_num++;
                    }
                }
                // printf("pos anchor num:%d neg anchor num:%d netual:%d\n",pos_anchor_index_set_num,neg_anchor_index_set_num,l.rpn_anchor_box_num*bbox_num-pos_anchor_index_set_num-neg_anchor_index_set_num);

                if(pos_anchor_index_set_num>l.rpn_sample_pos_max_num){
                    for(int rand_count=0;rand_count<l.rpn_sample_pos_max_num;rand_count++){
                        int random_index=rand()%pos_anchor_index_set_num+rand_count;
                        int _tmp=pos_anchor_index_set[random_index];
                        // printf ("%d %d\t",random_index,_tmp);
                        for(int ii=random_index;ii>rand_count;ii--){
                            pos_anchor_index_set[ii]=pos_anchor_index_set[ii-1];
                        }
                        pos_anchor_index_set[rand_count]=_tmp;
                        pos_anchor_index_set_num--;
                    }
                    pos_anchor_index_set_num=l.rpn_sample_pos_max_num;
                }

                int neg_sample_num=l.rpn_sample_num-pos_anchor_index_set_num;
                if(neg_anchor_index_set_num>neg_sample_num){
                    for(int rand_count=0;rand_count<neg_sample_num;rand_count++){
                        int random_index=rand()%neg_anchor_index_set_num+rand_count;
                        int _tmp=neg_anchor_index_set[random_index];
                        // printf ("%d %d\t",random_index,_tmp);
                        for(int ii=random_index;ii>rand_count;ii--){
                            neg_anchor_index_set[ii]=neg_anchor_index_set[ii-1];
                        }
                        neg_anchor_index_set[rand_count]=_tmp;
                        neg_anchor_index_set_num--;
                    }
                    neg_anchor_index_set_num=neg_sample_num;
                }
                // printf("rpn pos num:%d neg num:%d \t",pos_anchor_index_set_num,neg_anchor_index_set_num);
                // float pos_delta_scale=sqrt(1.0/(float)pos_anchor_index_set_num);
                // float neg_delta_scale=sqrt(1.0/(float)neg_anchor_index_set_num);
                float pos_delta_scale=1.0;
                float neg_delta_scale=1.0;
                // float pos_delta_scale=1.0/(pos_anchor_index_set_num+1);
                // float neg_delta_scale=1.0-pos_delta_scale;
                for(int i=0;i<pos_anchor_index_set_num;i++){
                    int box_id_plus=pos_anchor_index_set[i];
                    int box_id=box_id_plus%l.rpn_anchor_box_num;
                    // printf("NOW anchor iou:%f:   box_id:%d\n",total_iou_cal[box_id],box_id);
                    box anchor;

                    anchor.x=l.anchor_space[box_id*5+0];
                    anchor.y=l.anchor_space[box_id*5+1];
                    anchor.w=l.anchor_space[box_id*5+2];
                    anchor.h=l.anchor_space[box_id*5+3];
                    
                    int stride=l.rpn_bbox_pred->out_h*l.rpn_bbox_pred->out_w;
                    int channel_id=(int)(box_id/(l.rpn_bbox_pred->out_h*l.rpn_bbox_pred->out_w));
                    int loc=(int)(box_id%(l.rpn_bbox_pred->out_h*l.rpn_bbox_pred->out_w));
                    int index=b*l.rpn_bbox_pred->out_h*l.rpn_bbox_pred->out_w*l.rpn_bbox_pred->out_c+4*channel_id*stride+loc;
                    box truth = float_to_box(net.truth + (total_iou_label[box_id_plus]-1)*(4 + 1) + b*l.truths, 1);

                    float tx=(truth.x-anchor.x)/anchor.w;
                    float ty=(truth.y-anchor.y)/anchor.h;
                    float tw=log(truth.w/anchor.w);
                    float th=log(truth.h/anchor.h);

                    l.rpn_bbox_pred->delta[index + 0*stride] = smooth_l1_loss(pos_delta_scale,tx ,l.rpn_bbox_pred->output[index + 0*stride]);
                    l.rpn_bbox_pred->delta[index + 1*stride] = smooth_l1_loss(pos_delta_scale ,ty ,l.rpn_bbox_pred->output[index + 1*stride]);
                    l.rpn_bbox_pred->delta[index + 2*stride] = smooth_l1_loss(pos_delta_scale,tw ,l.rpn_bbox_pred->output[index + 2*stride]);
                    l.rpn_bbox_pred->delta[index + 3*stride] = smooth_l1_loss(pos_delta_scale ,th , l.rpn_bbox_pred->output[index + 3*stride]);

                    index=b*l.rpn_cls_score->out_h*l.rpn_cls_score->out_w*l.rpn_cls_score->out_c+channel_id*stride+loc;
                    l.rpn_cls_score->delta[index]=pos_delta_scale*(1.0-l.rpn_cls_score->output[index]);
                    // printf("%f \t",l.rpn_cls_score->output[index]);
                }
                //printf("\n");

                for(int i=0;i<neg_anchor_index_set_num;i++){
                    int box_id_plus=neg_anchor_index_set[i];
                    int box_id=box_id_plus%l.rpn_anchor_box_num;
                    int stride=l.rpn_bbox_pred->out_h*l.rpn_bbox_pred->out_w;
                    int channel_id=(int)(box_id/(l.rpn_bbox_pred->out_h*l.rpn_bbox_pred->out_w));
                    int loc=(int)(box_id%(l.rpn_bbox_pred->out_h*l.rpn_bbox_pred->out_w));
                    int index=b*l.rpn_cls_score->out_h*l.rpn_cls_score->out_w*l.rpn_cls_score->out_c+channel_id*stride+loc;
                    l.rpn_cls_score->delta[index]=-neg_delta_scale*l.rpn_cls_score->output[index];
                    //printf("%f \t",l.rpn_cls_score->output[index]);
                }

                free(total_iou_cal);
                free(total_iou_label);
                free(pos_anchor_index_set);
                free(neg_anchor_index_set);
        }
 

        //copy the obj and decode the output to detection format
        detection *dets_rpn = calloc(l.rpn_anchor_box_num, sizeof(detection));
        for(int i=0;i<l.rpn_anchor_box_num;i++){
            box anchor;
            int box_id=i;
            anchor.x=l.anchor_space[box_id*5+0];
            anchor.y=l.anchor_space[box_id*5+1];
            anchor.w=l.anchor_space[box_id*5+2];
            anchor.h=l.anchor_space[box_id*5+3];

            int stride=l.rpn_bbox_pred->out_h*l.rpn_bbox_pred->out_w;
            int channel_id=(int)(box_id/(l.rpn_bbox_pred->out_h*l.rpn_bbox_pred->out_w));
            int loc=(int)(box_id%(l.rpn_bbox_pred->out_h*l.rpn_bbox_pred->out_w));
            int index=b*l.rpn_bbox_pred->out_h*l.rpn_bbox_pred->out_w*l.rpn_bbox_pred->out_c+4*channel_id*stride+loc;
            int index_obj=b*l.rpn_cls_score->out_h*l.rpn_cls_score->out_w*l.rpn_cls_score->out_c+channel_id*stride+loc;

            float tx=l.rpn_bbox_pred->output[index + 0*stride];
            float ty=l.rpn_bbox_pred->output[index + 1*stride];
            float tw=l.rpn_bbox_pred->output[index + 2*stride];
            float th=l.rpn_bbox_pred->output[index + 3*stride];

            dets_rpn[i].bbox.x=anchor.w*tx+anchor.x;
            dets_rpn[i].bbox.y=anchor.h*ty+anchor.y;
            dets_rpn[i].bbox.w=exp(tw)*anchor.w;
            dets_rpn[i].bbox.h=exp(th)*anchor.h;

            if(dets_rpn[i].bbox.x-dets_rpn[i].bbox.w/2<0||dets_rpn[i].bbox.y-dets_rpn[i].bbox.h/2<0||dets_rpn[i].bbox.x+dets_rpn[i].bbox.w/2>1||dets_rpn[i].bbox.y+dets_rpn[i].bbox.h/2>1){
                dets_rpn[i].objectness=0;
            }else{
                dets_rpn[i].objectness=l.rpn_cls_score->output[index_obj];
            }

        }

        if(net.train&&l.train_rcnn_flg){
            do_nms_obj_rcnn(dets_rpn, l.rpn_anchor_box_num,l.train_pre_nms_num, 0.7,16,net);

            int pos_rois_counter=0;
            int pos_rois_num=l.rois_sample_num*l.rois_sample_ratio;

            for(int i=0;i<l.train_post_nms_num;i++){
                float _best_iou=0;
                int _class;
                box _best_truth_box;
                for(int t = 0; t < l.max_boxes; ++t){
                    box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
                    if(!truth.x) break;
                    float _iou=box_iou(truth,dets_rpn[i].bbox);
                    if(_iou>=l.pos_iou_thresh&&_iou>_best_iou){
                        _best_iou=_iou;
                        _best_truth_box=truth;
                        _class = net.truth[t*(4 + 1) + b*l.truths + 4];
                        // printf("cls:%d\n",_class);
                    }
                }
                if(_best_iou>=l.pos_iou_thresh){
                    int _index=(l.rois_sample_num*b+pos_rois_counter)*5;
                    int _index_rois_target=(l.rois_sample_num*b+pos_rois_counter)*4;

                    l.rois[_index]=b;

                    l.rois[_index+1]=(dets_rpn[i].bbox.x-dets_rpn[i].bbox.w/2)*net.w;
                    l.rois[_index+2]=(dets_rpn[i].bbox.y-dets_rpn[i].bbox.h/2)*net.h;
                    l.rois[_index+3]=(dets_rpn[i].bbox.x+dets_rpn[i].bbox.w/2)*net.w;
                    l.rois[_index+4]=(dets_rpn[i].bbox.y+dets_rpn[i].bbox.h/2)*net.h;

                    l.rois_target_label[(l.rois_sample_num*b+pos_rois_counter)*(l.classes+1)+_class+1]=1;

                    l.rois_target[_index_rois_target]=(_best_truth_box.x-dets_rpn[i].bbox.x)/dets_rpn[i].bbox.w;
                    l.rois_target[_index_rois_target+1]=(_best_truth_box.y-dets_rpn[i].bbox.y)/dets_rpn[i].bbox.h;
                    l.rois_target[_index_rois_target+2]=log(_best_truth_box.w/dets_rpn[i].bbox.w);
                    l.rois_target[_index_rois_target+3]=log(_best_truth_box.h/dets_rpn[i].bbox.h);

                    truth_roi[_index_rois_target]=_best_truth_box.x;
                    truth_roi[_index_rois_target+1]=_best_truth_box.y;
                    truth_roi[_index_rois_target+2]=_best_truth_box.w;
                    truth_roi[_index_rois_target+3]=_best_truth_box.h;
                    // printf("index:%d iou:%f i:%d class_id:%d obj:%f\n",l.rois_sample_num*b+pos_rois_counter,_best_iou,i,_class+1,dets_rpn[i].objectness);      
                    // printf("index:%d iou:%f i:%d class_id:%f obj:%f\n",l.rois_sample_num*b+pos_rois_counter,_best_iou,i,l.rois_target_label[l.rois_sample_num*b+pos_rois_counter],dets_rpn[i].objectness);      
                    pos_rois_counter++;
                }
                if(pos_rois_counter>=pos_rois_num){
                    break;
                }
            }
            pos_rois_num=pos_rois_counter;
            int neg_rois_num=l.rois_sample_num-pos_rois_num;
            int neg_rois_counter=0;
            // for(int i=0;i<l.train_post_nms_num;i++){
            for(int i=l.train_post_nms_num-1;i>=0;i--){
                float _best_iou=0;
                int _class;
                for(int t = 0; t < l.max_boxes; ++t){
                    box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
                    if(!truth.x) break;
                    float _iou=box_iou(truth,dets_rpn[i].bbox);
                    if(_iou>=l.pos_iou_thresh&&_iou>_best_iou){
                        _best_iou=_iou;
                    }
                }
                if(_best_iou<l.neg_iou_thresh_hi&&_best_iou>=l.neg_iou_thresh_lo){
                    int _index=(l.rois_sample_num*b+pos_rois_num+neg_rois_counter)*5;
                    l.rois[_index]=b;
                    l.rois[_index+1]=(dets_rpn[i].bbox.x-dets_rpn[i].bbox.w/2)*net.w;
                    l.rois[_index+2]=(dets_rpn[i].bbox.y-dets_rpn[i].bbox.h/2)*net.h;
                    l.rois[_index+3]=(dets_rpn[i].bbox.x+dets_rpn[i].bbox.w/2)*net.w;
                    l.rois[_index+4]=(dets_rpn[i].bbox.y+dets_rpn[i].bbox.h/2)*net.h;
                    l.rois_target_label[(l.rois_sample_num*b+pos_rois_num+neg_rois_counter)*(l.classes+1)]=1; 
                    neg_rois_counter++;
                }
                if(neg_rois_counter>=neg_rois_num){
                    break;
                }
            }
            // printf("rois pos_num:%d  neg_num:%d \t",pos_rois_num,neg_rois_counter);
        }else if(!net.train){

            do_nms_obj_rcnn(dets_rpn, l.rpn_anchor_box_num,l.test_pre_nms_num, l.rois_nms_thresh,l.rois_min_area_thresh,net);
            for(int i=0;i<l.rois_sample_num;i++){
                    int _index=(l.rois_sample_num*b+i)*5;
                    l.rois[_index]=b;
                    l.rois[_index+1]=(dets_rpn[i].bbox.x-dets_rpn[i].bbox.w/2)*net.w;
                    l.rois[_index+2]=(dets_rpn[i].bbox.y-dets_rpn[i].bbox.h/2)*net.h;
                    l.rois[_index+3]=(dets_rpn[i].bbox.x+dets_rpn[i].bbox.w/2)*net.w;
                    l.rois[_index+4]=(dets_rpn[i].bbox.y+dets_rpn[i].bbox.h/2)*net.h;
                    // printf("rank:%d obj:%f\n",i,dets_rpn[i].objectness);
            }
        }
        free_detections(dets_rpn, l.rpn_anchor_box_num);
    }

    #ifdef GPU
    cuda_push_array(l.rois_gpu, l.rois, l.batch*l.rois_sample_num*5);
    if(net.train&&l.train_rcnn_flg){
        cuda_push_array(l.rois_target_label_gpu, l.rois_target_label, l.batch*l.rois_sample_num*(l.classes+1));
        ROIAlignForwardLaucher(net.input_gpu, 1/l.downsample_ratio, l.rois_sample_num*l.batch, l.h, l.w,l.n,l.roialign_pooling_height, l.roialign_pooling_width, l.rois_gpu, l.roialign_pooling_output_gpu, 0);
        // cuda_pull_array(l.roialign_pooling_output_gpu, l.roialign_pooling_output, l.batch*l.n*l.roialign_pooling_height*l.roialign_pooling_width*l.rois_sample_num);

        for(int i=0;i<l.roi_head_layers_num;i++){
            l.roi_head_layers[i].train=net.train;
            if(i==0){
                forward_convolutional_layer_pure_gpu(l.roi_head_layers[i],l.roialign_pooling_output_gpu, net.workspace);
            }else if(l.roi_head_layers[i].type==CONVOLUTIONAL){
                forward_convolutional_layer_pure_gpu(l.roi_head_layers[i],l.roi_head_layers[i-1].output_gpu,net.workspace);
            }else if(l.roi_head_layers[i].type==AVGPOOL){
                forward_avgpool_layer_pure_gpu(l.roi_head_layers[i],l.roi_head_layers[i-1].output_gpu);
            }else if(l.roi_head_layers[i].type==SHORTCUT&&i-4<0){
                forward_shortcut_layer_pure_gpu(l.roi_head_layers[i], l.roi_head_layers[i-1].output_gpu,l.roialign_pooling_output_gpu);
            }else if(l.roi_head_layers[i].type==SHORTCUT&&i-4>=0){
                forward_shortcut_layer_pure_gpu(l.roi_head_layers[i], l.roi_head_layers[i-1].output_gpu, l.roi_head_layers[i-4].output_gpu);
            }
        }
        // forward_connected_layer_pure_gpu(*(l.roi_head_layers),l.roialign_pooling_output_gpu);
        forward_connected_layer_pure_gpu(*(l.roi_cls_score),l.roi_head_layers[l.roi_head_layers_num-1].output_gpu);
        forward_connected_layer_pure_gpu(*(l.roi_bbox_pred),l.roi_head_layers[l.roi_head_layers_num-1].output_gpu);
        
        cuda_pull_array(l.roi_bbox_pred->output_gpu, l.roi_bbox_pred->output, l.roi_bbox_pred->batch*l.roi_bbox_pred->outputs);
        float rois_bbox_scale=1;
        for(int b=0;b<l.roi_bbox_pred->batch;b++){
            for(int c=1;c<l.classes+1;c++){
                int _index=(l.classes+1)*b+c;
                if(l.rois_target_label[_index]==1){
                    l.roi_bbox_pred->delta[(_index)*4]=smooth_l1_loss(rois_bbox_scale,l.rois_target[4*b],l.roi_bbox_pred->output[(_index)*4]);
                    l.roi_bbox_pred->delta[(_index)*4+1]=smooth_l1_loss(rois_bbox_scale,l.rois_target[4*b+1],l.roi_bbox_pred->output[(_index)*4+1]);
                    l.roi_bbox_pred->delta[(_index)*4+2]=smooth_l1_loss(rois_bbox_scale,l.rois_target[4*b+2],l.roi_bbox_pred->output[(_index)*4+2]);
                    l.roi_bbox_pred->delta[(_index)*4+3]=smooth_l1_loss(rois_bbox_scale,l.rois_target[4*b+3],l.roi_bbox_pred->output[(_index)*4+3]);
                }
            }
        }
        cuda_push_array(l.roi_bbox_pred->delta_gpu, l.roi_bbox_pred->delta, l.roi_bbox_pred->batch*l.roi_bbox_pred->outputs);

        cuda_pull_array(l.roi_cls_score->output_gpu, l.roi_cls_score->output, l.roi_cls_score->batch*l.roi_cls_score->outputs);
        l.rois_cls_score_softmax->train=net.train;
        forward_softmax_layer_pure(*(l.rois_cls_score_softmax), l.roi_cls_score->output,l.rois_target_label);
        cuda_push_array(l.rois_cls_score_softmax->delta_gpu, l.rois_cls_score_softmax->delta, l.rois_cls_score_softmax->batch*l.rois_cls_score_softmax->outputs);
    }else if(!net.train){

        ROIAlignForwardLaucher(net.input_gpu, 1/l.downsample_ratio, l.rois_sample_num*l.batch, l.h, l.w,l.n,l.roialign_pooling_height, l.roialign_pooling_width, l.rois_gpu, l.roialign_pooling_output_gpu, 0);

        for(int i=0;i<l.roi_head_layers_num;i++){
            l.roi_head_layers[i].train=net.train;
            if(i==0){
                forward_convolutional_layer_pure_gpu(l.roi_head_layers[i],l.roialign_pooling_output_gpu, net.workspace);
            }else if(l.roi_head_layers[i].type==CONVOLUTIONAL){
                forward_convolutional_layer_pure_gpu(l.roi_head_layers[i],l.roi_head_layers[i-1].output_gpu,net.workspace);
            }else if(l.roi_head_layers[i].type==AVGPOOL){
                forward_avgpool_layer_pure_gpu(l.roi_head_layers[i],l.roi_head_layers[i-1].output_gpu);
            }else if(l.roi_head_layers[i].type==SHORTCUT&&i-4<0){
                forward_shortcut_layer_pure_gpu(l.roi_head_layers[i], l.roi_head_layers[i-1].output_gpu,l.roialign_pooling_output_gpu);
            }else if(l.roi_head_layers[i].type==SHORTCUT&&i-4>=0){
                forward_shortcut_layer_pure_gpu(l.roi_head_layers[i], l.roi_head_layers[i-1].output_gpu, l.roi_head_layers[i-4].output_gpu);
            }
        }
        // forward_connected_layer_pure_gpu(*(l.roi_head_layers),l.roialign_pooling_output_gpu);
        forward_connected_layer_pure_gpu(*(l.roi_cls_score),l.roi_head_layers[l.roi_head_layers_num-1].output_gpu);
        forward_connected_layer_pure_gpu(*(l.roi_bbox_pred),l.roi_head_layers[l.roi_head_layers_num-1].output_gpu);
        
        cuda_pull_array(l.roi_bbox_pred->output_gpu, l.roi_bbox_pred->output, l.roi_bbox_pred->batch*l.roi_bbox_pred->outputs);
        cuda_pull_array(l.roi_cls_score->output_gpu, l.roi_cls_score->output, l.roi_cls_score->batch*l.roi_cls_score->outputs);
        l.rois_cls_score_softmax->train=net.train;
        forward_softmax_layer_pure(*(l.rois_cls_score_softmax), l.roi_cls_score->output,l.rois_target_label);
    }

    #endif
    #ifndef GPU
        ROIAlignForwardLaucher_cpu(net.input, 1/l.downsample_ratio, l.rois_sample_num*l.batch, l.h, l.w,l.n,l.roialign_pooling_height, l.roialign_pooling_width, l.downsample_ratio,l.rois, l.roialign_pooling_output);

        for(int i=0;i<l.roi_head_layers_num;i++){
            l.roi_head_layers[i].train=net.train;
            if(i==0){
                forward_convolutional_layer_pure(l.roi_head_layers[i],l.roialign_pooling_output, net.workspace);
            }else if(l.roi_head_layers[i].type==CONVOLUTIONAL){
                forward_convolutional_layer_pure(l.roi_head_layers[i],l.roi_head_layers[i-1].output,net.workspace);
            }else if(l.roi_head_layers[i].type==AVGPOOL){
                forward_avgpool_layer_pure(l.roi_head_layers[i],l.roi_head_layers[i-1].output);
            }else if(l.roi_head_layers[i].type==SHORTCUT&&i-4<0){
                forward_shortcut_layer_pure(l.roi_head_layers[i], l.roi_head_layers[i-1].output,l.roialign_pooling_output);
            }else if(l.roi_head_layers[i].type==SHORTCUT&&i-4>=0){
                forward_shortcut_layer_pure(l.roi_head_layers[i], l.roi_head_layers[i-1].output, l.roi_head_layers[i-4].output);
            }
        }
        // forward_connected_layer_pure_gpu(*(l.roi_head_layers),l.roialign_pooling_output_gpu);
        forward_connected_layer_pure(*(l.roi_cls_score),l.roi_head_layers[l.roi_head_layers_num-1].output);
        forward_connected_layer_pure(*(l.roi_bbox_pred),l.roi_head_layers[l.roi_head_layers_num-1].output);
        l.rois_cls_score_softmax->train=net.train;
        forward_softmax_layer_pure(*(l.rois_cls_score_softmax), l.roi_cls_score->output,l.rois_target_label);

    #endif
    if(net.train&&l.train_rcnn_flg){
        for(int b=0;b<l.roi_bbox_pred->batch;b++){
            int pred_class=0;
            float best_class_score=0;
            int truth_class=0;
            for(int c=0;c<l.classes+1;c++){
                int _index=(l.classes+1)*b+c;

                if(best_class_score<l.rois_cls_score_softmax->output[_index]){
                    best_class_score=l.rois_cls_score_softmax->output[_index];
                    pred_class=c;
                }
                if(l.rois_target_label[_index]==1){
                    truth_class=c;
                }
            }

            if(truth_class>0||pred_class>0){
                int _index=(l.classes+1)*b+pred_class;
                float tx=l.roi_bbox_pred->output[(_index)*4];
                float ty=l.roi_bbox_pred->output[(_index)*4+1];
                float tw=l.roi_bbox_pred->output[(_index)*4+2];
                float th=l.roi_bbox_pred->output[(_index)*4+3];

                box roi;
                roi.x=(l.rois[b*5+1]+l.rois[b*5+3])/2/net.w;
                roi.y=(l.rois[b*5+2]+l.rois[b*5+4])/2/net.h;
                roi.w=(l.rois[b*5+3]-l.rois[b*5+1])/net.w;
                roi.h=(l.rois[b*5+4]-l.rois[b*5+2])/net.h;

                box bbox_pred;
                bbox_pred.x=roi.w*tx+roi.x;
                bbox_pred.y=roi.h*ty+roi.y;
                bbox_pred.w=exp(tw)*roi.w;
                bbox_pred.h=exp(th)*roi.h;

                box truth;
                truth.x=truth_roi[b*4];
                truth.y=truth_roi[b*4+1];
                truth.w=truth_roi[b*4+2];
                truth.h=truth_roi[b*4+3];

                float _iou=box_iou(truth,bbox_pred);
                // printf("pred_class:%d truth_class:%d _iou:%f\n",pred_class,truth_class,_iou);
            }
        }

        *(l.cost) = pow(mag_array(l.rpn_bbox_pred->delta, l.rpn_bbox_pred->outputs * l.rpn_bbox_pred->batch), 2)+pow(mag_array(l.rpn_cls_score->delta, l.rpn_cls_score->outputs * l.rpn_cls_score->batch), 2);
        printf("rpn total loss:%f\t",l.cost[0]);
        *(l.cost) = pow(mag_array(l.rpn_bbox_pred->delta, l.rpn_bbox_pred->outputs * l.rpn_bbox_pred->batch), 2);
        printf("rpn bbox loss:%f\t",l.cost[0]);
        *(l.cost) = pow(mag_array(l.rpn_cls_score->delta, l.rpn_cls_score->outputs * l.rpn_cls_score->batch), 2);
        printf("rpn obj loss:%f\t",l.cost[0]);
        printf("roi cls loss:%f\t",l.rois_cls_score_softmax->cost[0]);
        *(l.cost) = pow(mag_array(l.roi_bbox_pred->delta, l.roi_bbox_pred->batch*l.roi_bbox_pred->outputs), 2);
        printf("roi bbox loss:%f\n",l.cost[0]);
    }else if(net.train&&!l.train_rcnn_flg){
        *(l.cost) = pow(mag_array(l.rpn_bbox_pred->delta, l.rpn_bbox_pred->outputs * l.rpn_bbox_pred->batch), 2)+pow(mag_array(l.rpn_cls_score->delta, l.rpn_cls_score->outputs * l.rpn_cls_score->batch), 2);
        printf("rpn total loss:%f\t",l.cost[0]);
        *(l.cost) = pow(mag_array(l.rpn_bbox_pred->delta, l.rpn_bbox_pred->outputs * l.rpn_bbox_pred->batch), 2);
        printf("rpn bbox loss:%f\t",l.cost[0]);
        *(l.cost) = pow(mag_array(l.rpn_cls_score->delta, l.rpn_cls_score->outputs * l.rpn_cls_score->batch), 2);
        printf("rpn obj loss:%f\n",l.cost[0]);
    }


    free(truth_roi);
}

void backward_faster_rcnn_layer(const layer l, network net){

    backward_convolutional_layer_pure(*(l.rpn_cls_score), l.rpn_conv1->output,l.conv1_delta_add,net.workspace);
    backward_convolutional_layer_pure(*(l.rpn_bbox_pred), l.rpn_conv1->output,l.rpn_conv1->delta,net.workspace);
    shortcut_cpu(l.batch,l.rpn_conv1->out_w, l.rpn_conv1->out_h, l.rpn_conv1->out_c, l.conv1_delta_add,l.rpn_conv1->w,l.rpn_conv1->h, l.rpn_conv1->c, 1, 1, l.rpn_conv1->delta);
    backward_convolutional_layer_pure(*(l.rpn_conv1), net.input,net.delta,net.workspace);
}

void update_faster_rcnn_layer(const layer l,update_args a){
    update_convolutional_layer(*(l.rpn_cls_score),a);
    update_convolutional_layer(*(l.rpn_bbox_pred),a);
    update_convolutional_layer(*(l.rpn_conv1),a);
}

#ifdef GPU

void forward_faster_rcnn_layer_gpu(const layer l, network net)
{
    if(l.rpn_conv1->delta_gpu){
        fill_gpu(l.rpn_conv1->outputs * l.rpn_conv1->batch, 0,l.rpn_conv1->delta_gpu, 1);
        fill_gpu(l.rpn_conv1->outputs * l.rpn_conv1->batch, 0,l.conv1_delta_add_gpu, 1);
    }
    if(l.rpn_cls_score->delta_gpu){
        fill_gpu(l.rpn_cls_score->outputs * l.rpn_cls_score->batch, 0,l.rpn_cls_score->delta_gpu, 1);
    }
    if(l.rpn_bbox_pred->delta_gpu){
        fill_gpu(l.rpn_bbox_pred->outputs * l.rpn_bbox_pred->batch, 0,l.rpn_bbox_pred->delta_gpu, 1);
    }
    if(l.roialign_pooling_output_delta_gpu){
        fill_gpu(l.batch*l.n*l.roialign_pooling_height*l.roialign_pooling_width*l.rois_sample_num,0, l.roialign_pooling_output_delta_gpu,1);
    }
    for(int i=0;i<l.roi_head_layers_num;i++){
        if(l.roi_head_layers[i].delta_gpu){
            fill_gpu(l.roi_head_layers[i].outputs * l.roi_head_layers[i].batch, 0,l.roi_head_layers[i].delta_gpu, 1);
        }
        fill_gpu(l.roi_head_layers[l.roi_head_layers_num-1].batch*l.roi_head_layers[l.roi_head_layers_num-1].outputs, 0,l.roi_head_layers_delta_add_gpu, 1);
    }

    if(l.roi_cls_score->delta_gpu){
        fill_gpu(l.roi_cls_score->outputs * l.roi_cls_score->batch, 0,l.roi_cls_score->delta_gpu, 1);
    }
    if(l.roi_bbox_pred->delta_gpu){
        fill_gpu(l.roi_bbox_pred->outputs * l.roi_bbox_pred->batch, 0,l.roi_bbox_pred->delta_gpu, 1);
    }
        // l.rois_net_delta_add_gpu=cuda_make_array(l.rois_net_delta_add, l.batch*l.n*l.h*l.w);
        // l.roi_head_layers_delta_add_gpu=cuda_make_array(l.roi_head_layers_delta_add,l.roi_head_layers->batch* l.roi_head_layers->outputs);
    l.rpn_conv1->train=net.train;
    forward_convolutional_layer_pure_gpu(*(l.rpn_conv1), net.input_gpu,net.workspace);
    l.rpn_cls_score->train=net.train;
    forward_convolutional_layer_pure_gpu(*(l.rpn_cls_score), l.rpn_conv1->output_gpu,net.workspace);
    l.rpn_bbox_pred->train=net.train;
    forward_convolutional_layer_pure_gpu(*(l.rpn_bbox_pred), l.rpn_conv1->output_gpu,net.workspace);

    cuda_pull_array(l.rpn_cls_score->output_gpu, l.rpn_cls_score->output, l.rpn_cls_score->batch*l.rpn_cls_score->outputs);
    cuda_pull_array(l.rpn_bbox_pred->output_gpu,l.rpn_bbox_pred->output, l.rpn_bbox_pred->batch*l.rpn_bbox_pred->outputs);
    forward_faster_rcnn_layer(l, net);
    cuda_push_array(l.rpn_cls_score->delta_gpu, l.rpn_cls_score->delta, l.rpn_cls_score->batch*l.rpn_cls_score->outputs);
    cuda_push_array(l.rpn_bbox_pred->delta_gpu, l.rpn_bbox_pred->delta, l.rpn_bbox_pred->batch*l.rpn_bbox_pred->outputs);
}

void backward_faster_rcnn_layer_gpu(const layer l, network net)
{
    backward_convolutional_layer_pure_gpu(*(l.rpn_bbox_pred), l.rpn_conv1->output_gpu, l.conv1_delta_add_gpu,net.workspace);
    // backward_convolutional_layer_pure_gpu(*(l.rpn_bbox_pred), l.rpn_conv1->output_gpu,l.rpn_conv1->delta_gpu,net.workspace);
    backward_convolutional_layer_pure_gpu(*(l.rpn_cls_score), l.rpn_conv1->output_gpu, l.rpn_conv1->delta_gpu,net.workspace);
    shortcut_gpu(l.batch,l.rpn_conv1->out_w, l.rpn_conv1->out_h, l.rpn_conv1->out_c, l.conv1_delta_add_gpu,l.rpn_conv1->out_w, l.rpn_conv1->out_h, l.rpn_conv1->out_c, 1, 1, l.rpn_conv1->delta_gpu);
    backward_convolutional_layer_pure_gpu(*(l.rpn_conv1), net.input_gpu,net.delta_gpu,net.workspace);


    if(l.train_rcnn_flg){
        backward_softmax_layer_pure_gpu(*(l.rois_cls_score_softmax),l.roi_cls_score->delta_gpu);
        backward_connected_layer_pure_gpu(*(l.roi_cls_score),l.roi_head_layers[l.roi_head_layers_num-1].output_gpu,l.roi_head_layers_delta_add_gpu);
        backward_connected_layer_pure_gpu(*(l.roi_bbox_pred),l.roi_head_layers[l.roi_head_layers_num-1].output_gpu,l.roi_head_layers[l.roi_head_layers_num-1].delta_gpu);
        shortcut_gpu(l.roi_head_layers[l.roi_head_layers_num-1].batch,l.roi_head_layers[l.roi_head_layers_num-1].out_w, l.roi_head_layers[l.roi_head_layers_num-1].out_h, l.roi_head_layers[l.roi_head_layers_num-1].out_c, l.roi_head_layers_delta_add_gpu,l.roi_head_layers[l.roi_head_layers_num-1].out_w, l.roi_head_layers[l.roi_head_layers_num-1].out_h, l.roi_head_layers[l.roi_head_layers_num-1].out_c, 1, 1, l.roi_head_layers[l.roi_head_layers_num-1].delta_gpu);

        for(int i=l.roi_head_layers_num-1;i>=0;i--){
            if(i==0){
                backward_convolutional_layer_pure_gpu(l.roi_head_layers[i],l.roialign_pooling_output_gpu,l.roialign_pooling_output_delta_gpu,net.workspace);
            }else if(l.roi_head_layers[i].type==CONVOLUTIONAL){
                backward_convolutional_layer_pure_gpu(l.roi_head_layers[i],l.roi_head_layers[i-1].output_gpu,l.roi_head_layers[i-1].delta_gpu,net.workspace);
            }else if(l.roi_head_layers[i].type==AVGPOOL){
                backward_avgpool_layer_pure_gpu(l.roi_head_layers[i],l.roi_head_layers[i-1].delta_gpu);
            }else if(l.roi_head_layers[i].type==SHORTCUT&&i-4<0){
                backward_shortcut_layer_pure_gpu(l.roi_head_layers[i], l.roi_head_layers[i-1].delta_gpu,l.roialign_pooling_output_delta_gpu);
            }else if(l.roi_head_layers[i].type==SHORTCUT&&i-4>=0){
                backward_shortcut_layer_pure_gpu(l.roi_head_layers[i], l.roi_head_layers[i-1].delta_gpu, l.roi_head_layers[i-4].delta_gpu);
            }
        }

        // backward_connected_layer_pure_gpu(*(l.roi_head_layers),l.roialign_pooling_output_gpu,l.roialign_pooling_output_delta_gpu);                         
        ROIAlignBackwardLaucher(l.roialign_pooling_output_delta_gpu, 1/l.downsample_ratio, l.rois_sample_num*l.batch,  l.h, l.w,l.n,l.roialign_pooling_height, l.roialign_pooling_width, l.rois_gpu, net.delta_gpu, 0);
    }

}

void update_faster_rcnn_layer_gpu(layer l,update_args a){
    update_convolutional_layer_gpu(*(l.rpn_cls_score),a);
    update_convolutional_layer_gpu(*(l.rpn_bbox_pred),a);
    update_convolutional_layer_gpu(*(l.rpn_conv1),a);
    if(l.train_rcnn_flg){
        update_connected_layer_gpu(*(l.roi_cls_score),a);
        update_connected_layer_gpu(*(l.roi_bbox_pred),a);
        for(int i=0;i<l.roi_head_layers_num;i++){
            if(l.roi_head_layers[i].type==CONVOLUTIONAL){
            update_convolutional_layer_gpu(l.roi_head_layers[i],a);
            }
        }
    }
}
#endif
