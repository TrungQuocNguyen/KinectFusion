#pragma once

#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

using cv::cuda::GpuMat;


struct Configuration
{
    Configuration(){}
    Configuration(const int num_layers, const int kernel_size, const float sigma_color, const float sigma_spatial, const float max_depth) :
        num_layers(num_layers), kernel_size(kernel_size), sigma_color(sigma_color), sigma_spatial(sigma_spatial), max_depth(max_depth) {}
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

    // Constructor
    CameraIntrinsics() {}
    CameraIntrinsics(const int& img_width_, const int& img_height_, const float& fx_, const float& fy_, const float& cx_, const float& cy_){
        img_width = img_width_;
        img_height = img_height_;
        fx = fx_;
        fy = fy_;
        cx = cx_;
        cy = cy_;
    }

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


struct VolumeData {
    GpuMat tsdf_volume;  // (volume_size * volume_size, volume_size)
    int3 volume_size;
    float voxel_scale;

    VolumeData() {}
    
    VolumeData(const int3 _volume_size, const float _voxel_scale) :
        tsdf_volume(cv::cuda::createContinuous(_volume_size.y * _volume_size.z, _volume_size.x, CV_16SC2)),
        volume_size(_volume_size), voxel_scale(_voxel_scale)
    {
        tsdf_volume.setTo(0);
    }
};
