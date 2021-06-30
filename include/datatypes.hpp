#pragma once

#include <opencv2/core/cuda.hpp>

using cv::cuda::GpuMat;

struct Configuration
{
    // sub-sampling: Number of pyramid layers for each data frame
    int num_layers {3};
    // bilateral filtering
    int kernel_size {5};        // values are for debugging need to be changed later
    float sigma_color {1.f};
    float sigma_spatial {1.f};
    // Cut off depth values that are further away (set depth to 0)
    float max_depth {1000.f};

};

struct PreprocessedData
{
    std::vector<GpuMat> depth_pyramid;
    std::vector<GpuMat> filtered_depth_pyramid;
    //std::vector<GpuMat> color_pyramid;              //TODO: check if this is needed
    GpuMat color_map;

    std::vector<GpuMat> vertex_pyramid;
    std::vector<GpuMat> normal_pyramid;

    PreprocessedData(const size_t& size): depth_pyramid(size), filtered_depth_pyramid(size), vertex_pyramid(size), normal_pyramid(size) {} // set number of subsampled pyramid layers 
};

struct CameraIntrinsics
{
    int img_width, img_height;
    float fx, fy, cx, cy;       // focal lengths and center point

    // get camera parameters at certain pyramid level
    CameraIntrinsics getCameraIntrinsics(const int layer) const {
        if (layer == 0) return *this;

        const float scale_factor = powf(0.5f, static_cast<float>(layer));
        return (CameraIntrinsics) { img_width >> layer,
                                    img_height >> layer,
                                    fx * scale_factor,
                                    fy * scale_factor,
                                    (cx + 0.5f) * scale_factor - 0.5f,
                                    (cy + 0.5f) * scale_factor - 0.5f };
    }
};
