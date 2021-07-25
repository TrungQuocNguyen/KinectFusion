#pragma once

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "datatypes.hpp"

using Vector3f_da = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using Matrix3f_da = Eigen::Matrix<float, 3, 3, Eigen::DontAlign>;


void calculate_Ab(
    const cv::cuda::GpuMat& prev_vertex_map, const cv::cuda::GpuMat& prev_normal_map,
    const cv::cuda::GpuMat& vertex_map_current, const cv::cuda::GpuMat& normal_map_current,
    const Matrix3f_da& prev_rotation, const Vector3f_da& prev_translation,
    const Matrix3f_da& rotation, const Vector3f_da& translation,
    const CameraParameters& cam,
    const float& distance_threshold, const float& angle_threshold,
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& AtA, Eigen::Matrix<double, 6, 1>& Atb
);


bool pose_estimation(
    const ModelData& model_data, const PreprocessedData& data, 
    const CameraParameters& cam,
    const int& num_levels, const float& distance_threshold, const float& angle_threshold,
    const std::vector<int>& num_iterations,
    Eigen::Matrix4f& pose  // this should be set as previous pose
)
{
    const Eigen::Vector3f prev_global_translation {pose.block<3, 1>(0, 3)};
    const Eigen::Matrix3f prev_global_rotation {pose.block<3, 3>(0, 0)};
    Eigen::Vector3f global_translation {prev_global_translation};
    Eigen::Matrix3f global_rotation {prev_global_rotation};

    // ICP loop
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> AtA;
    Eigen::Matrix<double, 6, 1> Atb;
    bool flag_success = false;
    for (int level = num_levels - 1; level >= 0; --level)
    {
        flag_success = false;
        // printf("level : %d, %d\n", level, num_iterations[level]);
        for (int i = 0; i < num_iterations[level]; ++i) 
        {
            calculate_Ab(
                model_data.vertex_pyramid[level], model_data.normal_pyramid[level], 
                data.vertex_pyramid[level], data.normal_pyramid[level],
                prev_global_rotation, prev_global_translation,
                global_rotation, global_translation, 
                cam.getCameraParameters(level),
                distance_threshold, sinf(angle_threshold / 180.f * M_PI),
                AtA, Atb
            );
            
            // Solve equation
            double det = AtA.determinant();

            /*
            printf("iter %d: det %f\n", i, det);
            std::cout << AtA << std::endl;
            std::cout << Atb << std::endl;
            */
            
            if (isnan(det)) continue;
            flag_success = true;

            Eigen::Matrix<double, 6, 1> result {AtA.fullPivLu().solve(Atb)};
            Eigen::Matrix3f rotation_c_update = (
                Eigen::AngleAxisd(result[2], Eigen::Vector3d::UnitZ())
                * Eigen::AngleAxisd(result[1], Eigen::Vector3d::UnitY()) 
                * Eigen::AngleAxisd(result[0], Eigen::Vector3d::UnitX())
            ).matrix().cast<float>();
            Eigen::Vector3f translation_c_update = result.tail<3>().cast<float>();
            global_translation = rotation_c_update * global_translation + translation_c_update;
            global_rotation = rotation_c_update * global_rotation;
        }
    }
    if (!flag_success) return false;

    // new pose
    pose.block<3, 3>(0, 0) = global_rotation;
    pose.block<3, 1>(0, 3) = global_translation;

    return true;
}
