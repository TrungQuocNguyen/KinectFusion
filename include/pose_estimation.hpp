#pragma once

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

#include "datatypes.hpp"

using Vector3f_da = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using Matrix3f_da = Eigen::Matrix<float, 3, 3, Eigen::DontAlign>;

// Forward declaration
void estimation(const Matrix3f_da& rotation, const Vector3f_da& translation,
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
            const int pyramid_h,
            const float distance_threshold, const float angle_threshold,
            const std::vector<int>& iteration)
        {
            Eigen::Vector3f global_translation = pose.block(0, 3, 3, 1);
            Eigen::Matrix3f global_rotation = pose.block(0, 0, 3, 3);
            Eigen::Vector3f prev_global_translation = pose.block(0, 3, 3, 1);
            Eigen::Matrix3f prev_global_rotation_inv(global_rotation.inverse());

            // ICP loop
            for (int l = pyramid_h - 1; l >= 0; --l) {
                for (int i = 0; i < iteration[l]; ++i) {
                    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A{};
                    Eigen::Matrix<double, 6, 1> b{};

                    cuda::estimation(global_rotation, global_translation, data.vertex_pyramid[l], data.normal_pyramid[l],
                        prev_global_rotation_inv, prev_global_translation, cam_params.getCameraIntrinsics(l),
                        prev_data.vertex_pyramid[l], prev_data.normal_pyramid[l], distance_threshold, sinf(angle_threshold/180.f * 3.14159254f),
                        A, b);

                    // Solve equation
                    if (fabs(A.determinant()) > 100000 && !isnan(A.determinant())) {
                        Eigen::Matrix<float, 6, 1> result{ A.fullPivLu().solve(b).cast<float>() };

                        // Update
                        auto rotation_c_update(Eigen::AngleAxisf(result(0), Eigen::Vector3f::UnitX()) * Eigen::AngleAxisf(result(1), Eigen::Vector3f::UnitY()) *
                            Eigen::AngleAxisf(result(2), Eigen::Vector3f::UnitZ()));
                        auto translation_c_update = result.tail<3>();

                        global_rotation = global_rotation * rotation_c_update;
                        global_translation = global_translation * rotation_c_update + translation_c_update;
                    }
                }
            }

            // new pose
            pose.block(0, 0, 3, 3) = global_rotation;
            pose.block(0, 3, 3, 1) = global_translation;

            return true;
        }
    }