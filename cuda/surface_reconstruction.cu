#include "cuda_runtime.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev/common.hpp>
#include <Eigen/Core>
#include "datatypes.hpp"

using cv::cuda::PtrStepSz;
using Vector2i_da = Eigen::Matrix<int, 2, 1, Eigen::DontAlign>;
using Vector3f_da = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using Matrix3f_da = Eigen::Matrix<float, 3, 3, Eigen::DontAlign>;

constexpr int SHORT_MAX = 32767;
constexpr float INV_SHORT_MAX = 0.0000305185f;  // 1.f / SHORT_MAX;
constexpr int MAX_WEIGHT = 128;


__global__ void kernel_update_tsdf(
    const PtrStepSz<float> depth, 
    const CameraIntrinsics cam_params, 
    const Matrix3f_da R_w_c, const Vector3f_da t_w_c,  // world <= current
    const int3 volume_size, const float voxel_scale,
    const float truncation_distance,
    PtrStepSz<short2> tsdf_volume
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= volume_size.x || y >= volume_size.y) return;

    for (int z = 0; z < volume_size.z; ++z) {
        // 0.5f is for the centor of volume
        const Vector3f_da pos_c(
            (static_cast<float>(x) + 0.5f - volume_size.x / 2.f) * voxel_scale,
            (static_cast<float>(y) + 0.5f - volume_size.y / 2.f) * voxel_scale,
            (static_cast<float>(z) + 0.5f - volume_size.z / 2.f) * voxel_scale
        );  // voxel position in current coordinate
        const Vector3f_da pos_w = R_w_c * pos_c + t_w_c;  // voxel position in world coordinate
        if (pos_w[2] <= 0) continue;

        const Vector2i_da uv(
            __float2int_rn(pos_w[0] / pos_w[2] * cam_params.fx + cam_params.cx),
            __float2int_rn(pos_w[1] / pos_w[2] * cam_params.fy + cam_params.cy)
        );
        if (uv[0] < 0 || uv[0] >= depth.cols || uv[1] < 0 || uv[1] >= depth.rows) continue;

        const float d = depth.ptr(uv[1])[uv[0]];
        if (d <= 0.1f) continue;

        const Vector3f_da lambda_vec(pos_w[0] / pos_w[2], pos_w[1] / pos_w[2], 1.f);

        const float sdf = (t_w_c - pos_w).norm() / lambda_vec.norm() - d;
        if (sdf < - truncation_distance) continue;
        
        const float tsdf = fmin(1.f, sdf / truncation_distance);  // F_R_k
        const short weight = 1;  // W_R_k

        const short2 model_voxel = tsdf_volume.ptr(z * volume_size.y + y)[x];  // (tsdf, weight)
        const float model_tsdf = static_cast<float>(model_voxel.x) * INV_SHORT_MAX;  // F_{k-1}
        const short model_weight = model_voxel.y;  // W_{k-1}

        const float updated_tsdf = (model_weight * model_tsdf + weight * tsdf) / (model_weight + weight);

        const int new_tsdf = max(-SHORT_MAX, min(SHORT_MAX, static_cast<int>(updated_tsdf * SHORT_MAX)));
        const int new_weight = min(model_weight + weight, MAX_WEIGHT);

        tsdf_volume.ptr(z * volume_size.y + y)[x] = make_short2(static_cast<short>(new_tsdf), static_cast<short>(new_weight));
    }
}


void surface_reconstruction(
    const cv::cuda::GpuMat& depth,
    const CameraIntrinsics& cam_params,
    const Eigen::Matrix4f& T_c_w,  // current <= world
    const float truncation_distance,
    TSDFData& tsdf_data
)
{
    const dim3 threads(32, 32);
    const dim3 blocks(
        cv::cudev::divUp(volume.volume_size.x, threads.x),
        cv::cudev::divUp(volume.volume_size.y, threads.y)
    );

    kernel_update_tsdf<<<blocks, threads>>>(
        depth,
        cam_params, 
        T_c_w.block(0, 0, 3, 3).transpose(), - T_c_w.block(0, 0, 3, 3).transpose() * T_c_w.block(0, 3, 3, 1),
        tsdf_data.volume_size, tsdf_data.voxel_scale,
        truncation_distance,
        tsdf_data.tsdf
    );

    cudaThreadSynchronize();
}
