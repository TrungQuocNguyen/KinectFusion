#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev/common.hpp>
#include <Eigen/Core>

using cv::cuda::PtrStep;
using cv::cuda::PtrStepSz;
using cv::cuda::GpuMat;
using cv::cudev::divUp;

using Vector2i_da = Eigen::Matrix<int, 2, 1, Eigen::DontAlign>;
using Vector3i_da = Eigen::Matrix<int, 3, 1, Eigen::DontAlign>;
using Vector3f_da = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using Matrix3f_da = Eigen::Matrix<float, 3, 3, Eigen::DontAlign>;

constexpr int SHORT_MAX = 32767;
constexpr float INV_SHORT_MAX = 0.0000305185f;  // 1.f / SHORT_MAX;
constexpr int MAX_WEIGHT = 128;
