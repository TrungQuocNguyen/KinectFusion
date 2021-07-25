#pragma once

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

#include "datatypes.hpp"

// Forward declarartions
void compute_vertex_map(const GpuMat& depth_map, GpuMat& vertex_map, GpuMat& valid_vertex_mask, const CameraIntrinsics camera_params, const float max_depth);
void compute_normal_map(const GpuMat& vertex_map, GpuMat& normal_map, GpuMat& valid_vertex_mask);

void surface_measurement(PreprocessedData& data,
                         const cv::Mat& depth,
                         const cv::Mat& img,
                         const int& num_layers,
                         const int& kernel_size,
                         const float& sigma_color,
                         const float& sigma_spatial,
                         const CameraIntrinsics& camera_params,
                         const float& max_depth){
    
    // Allocate GPU memory
    //data.color_map = cv::cuda::createContinuous(camera_params.img_height, camera_params.img_width, CV_8UC3);
    for (int i = 0; i < num_layers; i++) {
        const int width = camera_params.getCameraIntrinsics(i).img_width;
        const int height = camera_params.getCameraIntrinsics(i).img_height;
        data.depth_pyramid[i] = cv::cuda::createContinuous(height, width, CV_32FC1);
        data.filtered_depth_pyramid[i] = cv::cuda::createContinuous(height, width, CV_32FC1);
        data.color_pyramid[i] = cv::cuda::createContinuous(height, width, CV_8UC3);
        data.vertex_pyramid[i] = cv::cuda::createContinuous(height, width, CV_32FC3);
        data.normal_pyramid[i] = cv::cuda::createContinuous(height, width, CV_32FC3);
        data.valid_vertex_mask[i] = cv::cuda::createContinuous(height, width, CV_8UC1);
    }

    data.depth_pyramid[0].upload(depth);
    //data.color_map.upload(img);
    data.color_pyramid[0].upload(img);

    if (!data.depth_pyramid[0].isContinuous()){ 
        data.depth_pyramid[0] = data.depth_pyramid[0].clone();
    }
    
    cv::cuda::Stream stream;
    // Step 1: Subsample depth (and color image???) to get pyramids (different scales of the images)
    for (int i = 0; i < num_layers - 1; i++)
    {
        cv::cuda::pyrDown(data.depth_pyramid[i], data.depth_pyramid[i+1], stream);
        cv::cuda::pyrDown(data.color_pyramid[i], data.color_pyramid[i+1], stream);
    }
    // Step 2: Smooth the depth image with bilateral filtering
    for (int i = 0; i < num_layers; i++)
    {
        cv::cuda::bilateralFilter(data.depth_pyramid[i], data.filtered_depth_pyramid[i], kernel_size, sigma_color, sigma_spatial, cv::BORDER_DEFAULT, stream);
    }
    stream.waitForCompletion();    

    // Step 3: Compute vertex and normal maps 
    for (int i = 0; i < num_layers; i++)
    {
        compute_vertex_map(data.filtered_depth_pyramid[i], data.vertex_pyramid[i], data.valid_vertex_mask[i], camera_params.getCameraIntrinsics(i), max_depth);
        compute_normal_map(data.vertex_pyramid[i], data.normal_pyramid[i], data.valid_vertex_mask[i]);
    }

}