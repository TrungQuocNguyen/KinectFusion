#pragma once

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

#include "datatypes.hpp"

using Vector3f_da = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using Matrix3f_da = Eigen::Matrix<float, 3, 3, Eigen::DontAlign>;


void calculate_Ab(
    const cv::cuda::GpuMat& prev_vertex_map, const cv::cuda::GpuMat& prev_normal_map,
    const cv::cuda::GpuMat& vertex_map_current, const cv::cuda::GpuMat& normal_map_current,
    const Matrix3f_da& prev_rotation_inv, const Vector3f_da& prev_translation,
    const Matrix3f_da& rotation, const Vector3f_da& translation,
    const CameraParameters& cam,
    const float& distance_threshold, const float& angle_threshold,
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& A, Eigen::Matrix<double, 6, 1>& b
);


bool pose_estimation(
    const PreprocessedData& data, const ModelData& model_data,
    const CameraParameters& cam,
    const int num_levels, const float distance_threshold, const float angle_threshold,
    const std::vector<int>& num_iterations,
    Eigen::Matrix4f& pose
)
{
    Eigen::Vector3f global_translation = pose.block<3, 1>(0, 3);
    Eigen::Matrix3f global_rotation = pose.block<3, 3>(0, 0);
    Eigen::Vector3f prev_global_translation = pose.block<3, 1>(0, 3);
    Eigen::Matrix3f prev_global_rotation_inv(global_rotation.transpose());

    // ICP loop
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A;
    Eigen::Matrix<double, 6, 1> b;
    for (int level = num_levels - 1; level >= 0; --level)
    {
        for (int i = 0; i < num_iterations[level]; ++i) 
        {

            calculate_Ab(
                model_data.vertex_pyramid[level], model_data.normal_pyramid[level], 
                data.vertex_pyramid[level], data.normal_pyramid[level],
                prev_global_rotation_inv, prev_global_translation,
                global_rotation, global_translation, 
                cam.getCameraParameters(level),
                distance_threshold, sinf(angle_threshold / 180.f * M_PI),
                A, b
            );

            // Solve equation
            if (fabs(A.determinant()) > 100000 && !isnan(A.determinant())) 
            {
                Eigen::Matrix<float, 6, 1> result{ A.fullPivLu().solve(b).cast<float>() };

                // Update
                auto rotation_c_update(
                    Eigen::AngleAxisf(result(0), Eigen::Vector3f::UnitX())
                    * Eigen::AngleAxisf(result(1), Eigen::Vector3f::UnitY()) 
                    * Eigen::AngleAxisf(result(2), Eigen::Vector3f::UnitZ())
                );
                auto translation_c_update = result.tail<3>();

                global_rotation = global_rotation * rotation_c_update;
                global_translation = global_translation * rotation_c_update + translation_c_update;
            }
        }
    }

    // new pose
    pose.block<3, 3>(0, 0) = global_rotation;
    pose.block<3, 1>(0, 3) = global_translation;

    return true;
}
