#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev/common.hpp>
#include <opencv2/opencv.hpp>
#include "datatypes.hpp"


void raycast_tsdf(
    const TSDFData &tsdf_data,
    const CameraParameters &cam,
    const Eigen::Matrix4f T_c_w,
    const float trancation_distance,
    GpuMat &vertex_map, GpuMat &normal_map
);


void surface_prediction(
    const TSDFData &volume,
    const CameraParameters &cam,
    const Eigen::Matrix4f &T_c_w,
    const float trancation_distance,
    const int num_levels,
    ModelData &model_data
)
{
    bool flag_predict_all = true;
    if (flag_predict_all)
    {
        for (int level = 0; level < num_levels; ++level)
        {
            raycast_tsdf(
                volume, cam.getCameraParameters(level), T_c_w, trancation_distance, 
                model_data.vertex_pyramid[level], model_data.normal_pyramid[level]
            );
        }
    }
    else
    {
        raycast_tsdf(
            volume, cam.getCameraParameters(0), T_c_w, trancation_distance,
            model_data.vertex_pyramid[0], model_data.normal_pyramid[0]
        );

        for (int i = 0; i < num_levels - 1; i++)
        {
            cv::cuda::pyrDown(model_data.vertex_pyramid[i], model_data.vertex_pyramid[i+1]);
            cv::cuda::pyrDown(model_data.normal_pyramid[i], model_data.normal_pyramid[i+1]);
        }
    }
}


void raycast_tsdf_using_depth(
    const TSDFData &tsdf_data, const cv::cuda::GpuMat &depth,
    const CameraParameters &cam,
    const Eigen::Matrix4f T_c_w,
    const float trancation_distance,
    GpuMat &vertex_map, GpuMat &normal_map
);


void surface_prediction_using_depth(
    const TSDFData &volume, const std::vector<cv::cuda::GpuMat> &depth_pyramid,
    const CameraParameters &cam,
    const Eigen::Matrix4f &T_c_w,
    const float trancation_distance,
    const int num_levels,
    ModelData &model_data
)
{
    bool flag_predict_all = true;
    if (flag_predict_all)
    {
        for (int level = 0; level < num_levels; ++level)
        {
            raycast_tsdf_using_depth(
                volume, depth_pyramid[level],
                cam.getCameraParameters(level), T_c_w, trancation_distance, 
                model_data.vertex_pyramid[level], model_data.normal_pyramid[level]
            );
        }
    }
    else
    {
        raycast_tsdf_using_depth(
            volume, depth_pyramid[0],
            cam.getCameraParameters(0), T_c_w, trancation_distance,
            model_data.vertex_pyramid[0], model_data.normal_pyramid[0]
        );

        for (int i = 0; i < num_levels - 1; i++)
        {
            cv::cuda::pyrDown(model_data.vertex_pyramid[i], model_data.vertex_pyramid[i+1]);
            cv::cuda::pyrDown(model_data.normal_pyramid[i], model_data.normal_pyramid[i+1]);
        }
    }
}