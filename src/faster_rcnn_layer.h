#ifndef FASTER_RCNN_LAYER_H
#define FASTER_RCNN_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_faster_rcnn_layer(int batch, int w, int h, int n,int classes,int adam,faster_rcnn_params f_param);

void forward_faster_rcnn_layer(const layer l, network net);
void backward_faster_rcnn_layer(const layer l, network net);
void update_faster_rcnn_layer(layer l,update_args a);

#ifdef GPU
void forward_faster_rcnn_layer_gpu(const layer l, network net);
void backward_faster_rcnn_layer_gpu(const layer l, network net);
void update_faster_rcnn_layer_gpu(layer l,update_args a);
#endif

#endif