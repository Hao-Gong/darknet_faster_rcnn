#ifndef ROIALIGN_POOLING_H
#define ROIALIGN_POOLING_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

int ROIAlignForwardLaucher_cpu(const float* bottom_data, const float spatial_scale, const int num_rois, const int height, const int width,
		const int channels, const int aligned_height, const int aligned_width, const int sampling_ratio,const float* bottom_rois, float* top_data);

#ifdef GPU
	// int ROIAlignForwardLaucher(const float* bottom_data, const float spatial_scale, const int num_rois, const int height, const int width,
	// 	const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* top_data,const int sampling_ratio);
	// int ROIAlignBackwardLaucher(const float* top_diff, const float spatial_scale, const int num_rois, const int height, const int width,
	// 	const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* bottom_diff,const int sampling_ratio);
	int ROIAlignBackwardLaucher(const float* top_diff, const float spatial_scale, const int num_rois, const int height, const int width,
		const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* bottom_diff, cudaStream_t stream);
    int ROIAlignForwardLaucher(const float* bottom_data, const float spatial_scale, const int num_rois, const int height, const int width,
		const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* top_data, cudaStream_t stream);
#endif

#endif

