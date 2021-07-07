#include "cuda_runtime.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev/common.hpp>
#include <Eigen/Core>
#include "datatypes.hpp"

using cv::cuda::PtrStepSz;
using Vector2i_da = Eigen::Matrix<int, 2, 1, Eigen::DontAlign>;
using Vector3f_da = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using Matrix3f_da = Eigen::Matrix<float, 3, 3, Eigen::DontAlign>;

constexpr float DIVSHORTMAX = 0.0000305185f; //1.f / SHRT_MAX;
constexpr int SHORTMAX = 32767; //SHRT_MAX;
constexpr int MAX_WEIGHT = 128;


__global__ void kernel_update_tsdf(
    const PtrStepSz<float> depth, 
    const int3 volume_size, const float voxel_scale,
    const CameraIntrinsics cam_params, const float truncation_distance,
    const Matrix3f_da rotation, const Vector3f_da translation,
    PtrStepSz<short2> tsdf_volume
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= volume_size.x || y >= volume_size.y) return;

    for (int z = 0; z < volume_size.z; ++z) {
        const Vector3f_da position(
            (static_cast<float>(x) + 0.5f) * voxel_scale,
            (static_cast<float>(y) + 0.5f) * voxel_scale,
            (static_cast<float>(z) + 0.5f) * voxel_scale
        );
        const Vector3f_da camera_pos = rotation * position + translation;

        if (camera_pos.z() <= 0) continue;

        const Vector2i_da uv(
            __float2int_rn(camera_pos.x() / camera_pos.z() * cam_params.fx + cam_params.cx),
            __float2int_rn(camera_pos.y() / camera_pos.z() * cam_params.fy + cam_params.cy)
        );

        if (uv.x() < 0 || uv.x() >= depth.cols || uv.y() < 0 || uv.y() >= depth.rows)
            continue;

        const float d = depth.ptr(uv.y())[uv.x()];

        if (d <= 0) continue;

        const Vector3f_da xylambda(
            (uv.x() - cam_params.cx) / cam_params.fx,
            (uv.y() - cam_params.cy) / cam_params.fy,
            1.f
        );
        const float lambda = xylambda.norm();

        const float sdf = (-1.f) * ((1.f / lambda) * camera_pos.norm() - d);

        if (sdf >= -truncation_distance) {
            const float new_tsdf = fmin(1.f, sdf / truncation_distance);

            short2 voxel_tuple = tsdf_volume.ptr(z * volume_size.y + y)[x];

            const float current_tsdf = static_cast<float>(voxel_tuple.x) * DIVSHORTMAX;
            const int current_weight = voxel_tuple.y;

            const int add_weight = 1;

            const float updated_tsdf = (current_weight * current_tsdf + add_weight * new_tsdf) /
                (current_weight + add_weight);

            const int new_weight = min(current_weight + add_weight, MAX_WEIGHT);
            const int new_value = max(-SHORTMAX, min(SHORTMAX, static_cast<int>(updated_tsdf * SHORTMAX)));

            tsdf_volume.ptr(z * volume_size.y + y)[x] = make_short2(
                static_cast<short>(new_value),
                static_cast<short>(new_weight)
            );
        }
    }
}


void surface_reconstruction(
    const cv::cuda::GpuMat& depth,
    const CameraIntrinsics& cam_params, const float truncation_distance,
    const Eigen::Matrix4f& T_c_w,
    VolumeData& volume
)
{
    const dim3 threads(32, 32);
    const dim3 blocks(
        cv::cudev::divUp(volume.volume_size.x, threads.x),
        cv::cudev::divUp(volume.volume_size.y, threads.y)
    );

    kernel_update_tsdf<<<blocks, threads>>>(
        depth,
        volume.volume_size, volume.voxel_scale,
        cam_params, truncation_distance,
        T_c_w.block(0, 0, 3, 3), T_c_w.block(0, 3, 3, 1),
        volume.tsdf_volume
    );

    cudaThreadSynchronize();
}