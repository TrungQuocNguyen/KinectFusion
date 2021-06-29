#pragma once

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

using cv::cuda::GpuMat;


PreprocessedData surface_measurement(const cv::Mat raw_depth_map, size_t num_levels, size_t kernel_size, float sigma_color, float sigma_spatial){
    assert (num_levels > 0);
    PreprocessedData data(num_levels);
    data.depth_pyramid[0].upload(raw_depth_map);
    
    cv::cuda::Stream stream;
    // Step 1: Subsample depth (and color image???) to get pyramids (different scales of the images)
    for (size_t i = 0; i < num_levels - 1; i++)
    {
        cv::cuda::pyrDown(data.depth_pyramid[i], data.depth_pyramid[i+1], stream);
    }
    // Step 2: Smooth the depth image with bilateral filtering
    for (size_t i = 0; i < num_levels; i++)
    {
        cv::cuda::bilateralFilter(data.depth_pyramid[i], data.filtered_depth_pyramid[i], kernel_size, sigma_color, sigma_spatial, cv::BORDER_DEFAULT, stream);
    }
    stream.waitForCompletion();    

    // Step 3: Compute vertex maps 
    
    // Step 4: Compute normal maps

    
    return data;
}