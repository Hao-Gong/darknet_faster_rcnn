#ifndef SHORTCUT_LAYER_H
#define SHORTCUT_LAYER_H

#include "layer.h"
#include "network.h"

layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2);
layer make_shortcut_layer2(int batch, int index, int w, int h, int c, int w2, int h2, int c2,ACTIVATION activation,float alpha, float beta);
void forward_shortcut_layer(const layer l, network net);
void backward_shortcut_layer(const layer l, network net);
void resize_shortcut_layer(layer *l, int w, int h);
void forward_shortcut_layer_pure(const layer l, float* input,float* output);
void backward_shortcut_layer_pure(const layer l, float* delta,float* delta2);
#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network net);
void backward_shortcut_layer_gpu(const layer l, network net);
void forward_shortcut_layer_pure_gpu(const layer l, float*  direct_input_gpu,float* res_output_gpu);
void backward_shortcut_layer_pure_gpu(const layer l, float* direct_delta_gpu,float* res_delta_gpu);
#endif

#endif
