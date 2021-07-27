#pragma once

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

#include "datatypes.hpp"

// Forward declarartions
void computeVertexMap(const GpuMat& depth_map, const CameraParameters& cam, GpuMat& vertex_map);
void computeNormalMap(const GpuMat& vertex_map, GpuMat& normal_map);

void surfaceMeasurement(
    const cv::Mat& depth, const cv::Mat& img,
    const int& num_layers, const int& kernel_size, const float& sigma_color, const float& sigma_spatial,
    const CameraParameters& cam,
    FrameData& data
)
{
    data.depth_pyramid[0].upload(depth);
    data.color_map.upload(img);
    
    cv::cuda::Stream stream;

    // Step 1: Subsample depth get pyramids (different scales of the images)
    for (int i = 0; i < num_layers - 1; i++)
    {
        cv::cuda::pyrDown(data.depth_pyramid[i], data.depth_pyramid[i + 1], stream);
    }
    
    // Step 2: Smooth the depth image with bilateral filtering
    for (int i = 0; i < num_layers; i++)
    {
        cv::cuda::bilateralFilter(
            data.depth_pyramid[i], data.depth_pyramid[i], 
            kernel_size, sigma_color, sigma_spatial, cv::BORDER_DEFAULT, stream
        );
    }

    // Step 3: Compute vertex and normal maps 
    for (int i = 0; i < num_layers; i++)
    {
        computeVertexMap(data.depth_pyramid[i], cam.getCameraParameters(i), data.vertex_pyramid[i]);
        computeNormalMap(data.vertex_pyramid[i], data.normal_pyramid[i]);
    }
}