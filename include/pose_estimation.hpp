#pragma once

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

#include "datatypes.hpp"

using Vector3f_da = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using Matrix3f_da = Eigen::Matrix<float, 3, 3, Eigen::DontAlign>;

// Forward declaration
void estimate_step(const Matrix3f_da& rotation, const Vector3f_da& translation,
    const cv::cuda::GpuMat& vertex_map_current, const cv::cuda::GpuMat& normal_map_current,
    const Matrix3f_da& prev_rotation_inv, const Vector3f_da& prev_translation,
    const CameraIntrinsics& cam_params,
    const cv::cuda::GpuMat& prev_vertex_map, const cv::cuda::GpuMat& prev_normal_map,
    float distance_threshold, float angle_threshold,
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& A, Eigen::Matrix<double, 6, 1>& b);

        bool pose_estimation(Eigen::Matrix4f& pose,
            const PreprocessedData& data,
            const PreprocessedData& prev_data,
            const CameraIntrinsics& cam_params,
            const int pyramid_height,
            const float distance_threshold, const float angle_threshold,
            const std::vector<int>& iterations)
        {
            Eigen::Matrix3f global_rotation = pose.block(0, 0, 3, 3);
            Eigen::Vector3f global_translation = pose.block(0, 3, 3, 1);

            Eigen::Matrix3f prev_global_rotation_inv(global_rotation.inverse());
            Eigen::Vector3f prev_global_translation = pose.block(0, 3, 3, 1);

            // ICP loop
            for (int level = pyramid_height - 1; level >= 0; --level) {
                for (int iteration = 0; iteration < iterations[level]; ++iteration) {
                    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A{};
                    Eigen::Matrix<double, 6, 1> b{};

                    cuda::estimate_step(global_rotation, global_translation,
                        data.vertex_pyramid[level], data.normal_pyramid[level],
                        prev_global_rotation_inv, prev_global_translation,
                        cam_params.level(level),
                        prev_data.vertex_pyramid[level], prev_data.normal_pyramid[level],
                        distance_threshold, sinf(angle_threshold * 3.14159254f / 180.f),
                        A, b);

                    // Solve equation
                    double det = A.determinant();
                    if (fabs(det) < 100000 || std::isnan(det))
                        return false;
                    Eigen::Matrix<float, 6, 1> result{ A.fullPivLu().solve(b).cast<float>() };
                    float alpha = result(0);
                    float beta = result(1);
                    float gamma = result(2);

                    // Update
                    auto camera_rotation_incre(
                        Eigen::AngleAxisf(gamma, Eigen::Vector3f::UnitZ()) *
                        Eigen::AngleAxisf(beta, Eigen::Vector3f::UnitY()) *
                        Eigen::AngleAxisf(alpha, Eigen::Vector3f::UnitX()));
                    auto camera_translation_incre = result.tail<3>();

                    global_translation =
                        camera_rotation_incre * global_translation + camera_translation_incre;
                    global_rotation = camera_rotation_incre * global_rotation;
                }
            }

            // new pose
            pose.block(0, 0, 3, 3) = global_rotation;
            pose.block(0, 3, 3, 1) = global_translation;

            return true;
        }
    }