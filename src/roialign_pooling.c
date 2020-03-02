#include "roialign_pooling.h"


// implementation taken from Caffe2
typedef struct {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  float w1;
  float w2;
  float w3;
  float w4;
}PreCalc;

void pre_calc_for_bilinear_interpolate(
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int iy_upper,
    const int ix_upper,
    float roi_start_h,
    float roi_start_w,
    float bin_size_h,
    float bin_size_w,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    PreCalc* pre_calc) {
  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int iy = 0; iy < iy_upper; iy++) {
        const float yy = roi_start_h + ph * bin_size_h + (iy + .5f) * bin_size_h /(roi_bin_grid_h); // e.g., 0.5, 1.5
        for (int ix = 0; ix < ix_upper; ix++) {
          const float xx = roi_start_w + pw * bin_size_w +(ix + .5f) * bin_size_w /(roi_bin_grid_w);

          float x = xx;
          float y = yy;
          // deal with: inverse elements are out of feature map boundary
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            PreCalc pc;
            pc.pos1 = 0;
            pc.pos2 = 0;
            pc.pos3 = 0;
            pc.pos4 = 0;
            pc.w1 = 0;
            pc.w2 = 0;
            pc.w3 = 0;
            pc.w4 = 0;
            pre_calc[pre_calc_index] = pc;
            pre_calc_index += 1;
            continue;
          }

          if (y <= 0) {
            y = 0;
          }
          if (x <= 0) {
            x = 0;
          }

          int y_low = (int)y;
          int x_low = (int)x;
          int y_high;
          int x_high;

          if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = y_low;
          } else {
            y_high = y_low + 1;
          }

          if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = x_low;
          } else {
            x_high = x_low + 1;
          }

          float ly = y - y_low;
          float lx = x - x_low;
          float hy = 1. - ly, hx = 1. - lx;
          float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

          // save weights and indeces
          PreCalc pc;
          pc.pos1 = y_low * width + x_low;
          pc.pos2 = y_low * width + x_high;
          pc.pos3 = y_high * width + x_low;
          pc.pos4 = y_high * width + x_high;
          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc[pre_calc_index] = pc;

          pre_calc_index += 1;
        }
      }
    }
  }
}


void ROIAlignForward_cpu_kernel(
    const int nthreads,
    const float* bottom_data,
    const float spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const float* bottom_rois,
    float* top_data) {
  //AT_ASSERT(roi_cols == 4 || roi_cols == 5);
  int roi_cols = 5;

  int n_rois = nthreads / channels / pooled_width / pooled_height;
  // (n, c, ph, pw) is an element in the pooled output
  // can be parallelized using omp
  // #pragma omp parallel for num_threads(32)
  for (int n = 0; n < n_rois; n++) {
    int index_n = n * channels * pooled_width * pooled_height;

    // roi could have 4 or 5 columns
    const float* offset_bottom_rois = bottom_rois + n * roi_cols;
    int roi_batch_ind = 0;
    if (roi_cols == 5) {
      roi_batch_ind = offset_bottom_rois[0];
      offset_bottom_rois++;
    }

    // Do not using rounding; this implementation detail is critical
    float roi_start_w = offset_bottom_rois[0] * spatial_scale;
    float roi_start_h = offset_bottom_rois[1] * spatial_scale;
    float roi_end_w = offset_bottom_rois[2] * spatial_scale;
    float roi_end_h = offset_bottom_rois[3] * spatial_scale;
    // T roi_start_w = round(offset_bottom_rois[0] * spatial_scale);
    // T roi_start_h = round(offset_bottom_rois[1] * spatial_scale);
    // T roi_end_w = round(offset_bottom_rois[2] * spatial_scale);
    // T roi_end_h = round(offset_bottom_rois[3] * spatial_scale);

    // Force malformed ROIs to be 1x1
    float roi_width = (roi_end_w - roi_start_w>1.)?roi_end_w - roi_start_w:1.;
    float roi_height =(roi_end_h - roi_start_h>1.)?roi_end_h - roi_start_h:1.;
    float bin_size_h =roi_height / pooled_height;
    float bin_size_w = roi_width/ pooled_width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

	// 		float bin_size_h = roi_height / (pooled_height - 1.);
	// 		float bin_size_w = roi_width / (pooled_width - 1.);
    // printf("%d %f\n",roi_bin_grid_h,bin_size_h );

    // We do average (integral) pooling inside a bin
    const float count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    // we want to precalculate indeces and weights shared by all chanels,
    // this is the key point of optimiation
    PreCalc* pre_calc=calloc(roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height,sizeof(PreCalc));

    pre_calc_for_bilinear_interpolate(
        height,
        width,
        pooled_height,
        pooled_width,
        roi_bin_grid_h,
        roi_bin_grid_w,
        roi_start_h,
        roi_start_w,
        bin_size_h,
        bin_size_w,
        roi_bin_grid_h,
        roi_bin_grid_w,
        pre_calc);

      for (int c = 0; c < channels; c++) {
      int index_n_c = index_n + c * pooled_width * pooled_height;
      const float* offset_bottom_data =
          bottom_data + (roi_batch_ind * channels + c) * height * width;
      int pre_calc_index = 0;

      for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
          int index = index_n_c + ph * pooled_width + pw;

          float output_val = 0.;
          for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
              PreCalc pc = pre_calc[pre_calc_index];
              output_val += pc.w1 * offset_bottom_data[pc.pos1] +
                  pc.w2 * offset_bottom_data[pc.pos2] +
                  pc.w3 * offset_bottom_data[pc.pos3] +
                  pc.w4 * offset_bottom_data[pc.pos4];

              pre_calc_index += 1;
            }
          }
          output_val /= count;
          

          top_data[index] = output_val;
        } // for pw
      } // for ph
    } // for c
    free(pre_calc);
  } // for n
}

	int ROIAlignForwardLaucher_cpu(const float* bottom_data, const float spatial_scale, const int num_rois, const int height, const int width,
		const int channels, const int aligned_height, const int aligned_width, const int sampling_ratio,const float* bottom_rois, float* top_data) {

		const int output_size = num_rois * aligned_height * aligned_width * channels;   //要处理的总任务数量，即pooling完之后featuremap的大小
        ROIAlignForward_cpu_kernel(
            output_size,
            bottom_data,
            spatial_scale,
            channels,
            height,
            width,
            aligned_height,
            aligned_width,
            sampling_ratio,
            bottom_rois,
            top_data) ;

		return 1;
	}

// at::Tensor ROIAlign_forward_cpu(const at::Tensor& input,
//                                 const at::Tensor& rois,
//                                 const float spatial_scale,
//                                 const int pooled_height,
//                                 const int pooled_width,
//                                 const int sampling_ratio) {

//   auto num_rois = rois.size(0);
//   auto channels = input.size(1);
//   auto height = input.size(2);
//   auto width = input.size(3);

//   auto output = at::empty({num_rois, channels, pooled_height, pooled_width}, input.options());
//   auto output_size = num_rois * pooled_height * pooled_width * channels;

//   if (output.numel() == 0) {
//     return output;
//   }

//   AT_DISPATCH_FLOATING_TYPES(input.type(), "ROIAlign_forward", [&] {
//     ROIAlignForward_cpu_kernel<scalar_t>(
//          output_size,
//          input.data<scalar_t>(),
//          spatial_scale,
//          channels,
//          height,
//          width,
//          pooled_height,
//          pooled_width,
//          sampling_ratio,
//          rois.data<scalar_t>(),
//          output.data<scalar_t>());
//   });
//   return output;
// }
