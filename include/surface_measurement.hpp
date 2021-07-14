#pragma once

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

#include "datatypes.hpp"

// Forward declarartions
void compute_vertex_map(const GpuMat& depth_map, GpuMat& vertex_map, const CameraParameters cam, const float max_depth);
void compute_normal_map(const GpuMat& vertex_map, GpuMat& normal_map);

void surface_measurement(
    PreprocessedData& data,
    const cv::Mat& depth, const cv::Mat& img,
    const int& num_layers,
    const int& kernel_size, const float& sigma_color, const float& sigma_spatial,
    const CameraParameters& cam, const float& max_depth
)
{
    // Allocate GPU memory
    data.color_map = cv::cuda::createContinuous(cam.height, cam.width, CV_8UC3);
    for (int i = 0; i < num_layers; i++) 
    {
        const int width = cam.getCameraParameters(i).width;
        const int height = cam.getCameraParameters(i).height;
        data.depth_pyramid[i] = cv::cuda::createContinuous(height, width, CV_32FC1);
        // data.filtered_depth_pyramid[i] = cv::cuda::createContinuous(height, width, CV_32FC1);
        data.vertex_pyramid[i] = cv::cuda::createContinuous(height, width, CV_32FC3);
        data.normal_pyramid[i] = cv::cuda::createContinuous(height, width, CV_32FC3);
    }

    data.depth_pyramid[0].upload(depth);
    data.color_map.upload(img);

    if (!data.depth_pyramid[0].isContinuous())
    { 
        data.depth_pyramid[0] = data.depth_pyramid[0].clone();
    }
    
    cv::cuda::Stream stream;
    // Step 1: Subsample depth (and color image???) to get pyramids (different scales of the images)
    for (int i = 0; i < num_layers - 1; i++)
    {
        cv::cuda::pyrDown(data.depth_pyramid[i], data.depth_pyramid[i+1], stream);
    }
    // Step 2: Smooth the depth image with bilateral filtering
    for (int i = 0; i < num_layers; i++)
    {
        cv::cuda::bilateralFilter(data.depth_pyramid[i], data.depth_pyramid[i], kernel_size, sigma_color, sigma_spatial, cv::BORDER_DEFAULT, stream);
    }
    stream.waitForCompletion();    

    // Step 3: Compute vertex and normal maps 
    for (int i = 0; i < num_layers; i++)
    {
        compute_vertex_map(data.depth_pyramid[i], data.vertex_pyramid[i], cam.getCameraParameters(i), max_depth);
        compute_normal_map(data.vertex_pyramid[i], data.normal_pyramid[i]);
    }
}