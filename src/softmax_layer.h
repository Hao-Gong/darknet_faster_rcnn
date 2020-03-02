#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "layer.h"
#include "network.h"

typedef layer softmax_layer;

void softmax_array(float *input, int n, float temp, float *output);
softmax_layer make_softmax_layer(int batch, int inputs, int groups);
void forward_softmax_layer(const softmax_layer l, network net);
void backward_softmax_layer(const softmax_layer l, network net);
void forward_softmax_layer_pure(const softmax_layer l, float* input,float* truth);
void backward_softmax_layer_pure(const softmax_layer l, float* delta);

#ifdef GPU
void pull_softmax_layer_output(const softmax_layer l);
void forward_softmax_layer_gpu(const softmax_layer l, network net);
void backward_softmax_layer_gpu(const softmax_layer l, network net);
void forward_softmax_layer_pure_gpu(const softmax_layer l, float* input_gpu,float *truth_gpu,float *truth);
void backward_softmax_layer_pure_gpu(const softmax_layer l, float* delta_gpu);
#endif

#endif
