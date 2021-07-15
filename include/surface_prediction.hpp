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
    for (int level = 0; level < num_levels; ++level)
    {
        raycast_tsdf(
            volume, cam.getCameraParameters(level), T_c_w, trancation_distance, 
            model_data.vertex_pyramid[level], model_data.normal_pyramid[level]
        );
    }
}