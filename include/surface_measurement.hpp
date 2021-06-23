#pragma once

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

using cv::cuda::GpuMat;


/*struct PreprocessedData{
std::vector<GpuMat> depth_pyramid;
std::vector<GpuMat> filtered_depth_pyramid;
std::vector<GpuMat> color_pyramid;

std::vector<GpuMat> vertex_pyramid;
std::vector<GpuMat> normal_pyramid;
}*/

struct PreprocessedData{
    GpuMat depth_pyramid;
    GpuMat color_pyramid;

    GpuMat filtered_depth_pyramid;

    GpuMat vertex_pyramid;
    GpuMat normal_pyramid;
};


//PreprocessedData surface_measurement(cv::Mat& raw_depth_map);
PreprocessedData surface_measurement(const cv::Mat raw_depth_map){
    PreprocessedData data;
    data.depth_pyramid.upload(raw_depth_map);
    
    // Step 1: Smooth the depth image with bilateral filtering
    cv::cuda::bilateralFilter(data.depth_pyramid, data.filtered_depth_pyramid, 10, 20.f, 20.f);

    // Step 2: Compute vertex maps 
    
    // Step 3: Compute normal maps

    
    return data;
}