#include "cuda_runtime.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev/common.hpp>
#include <Eigen/Core>
#include "datatypes.hpp"

using cv::cuda::PtrStepSz;


void surface_prediction(
    const VolumeData &volume,
    const CameraIntrinsics &cam,
     const Eigen::Matrix4f &T_c_w,
    const float trancation_distance,
    GpuMat &vertex_map, GpuMat &normal_map
)
{
    
}