#include "cuda_runtime.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev/common.hpp>
#include <Eigen/Core>
#include "datatypes.hpp"

using cv::cuda::PtrStepSz;
using Vector2i_da = Eigen::Matrix<int, 2, 1, Eigen::DontAlign>;
using Vector3f_da = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using Matrix3f_da = Eigen::Matrix<float, 3, 3, Eigen::DontAlign>;
constexpr float DIVSHORTMAX = 0.0000305185f;  // 1.f / SHORT_MAX;

__global__
void extract_points_kernel(
    const PtrStepSz<short2> tsdf_volume, 
    const int3 volume_size, const float voxel_scale,
    PtrStepSz<float3> vertices, PtrStepSz<float3> normals,
    int *point_num
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= volume_size.x - 1 || y >= volume_size.y - 1) return;

    for (int z = 0; z < volume_size.z - 1; ++z) {
        const short2 value = tsdf_volume.ptr(z * volume_size.y + y)[x];

        const float tsdf = static_cast<float>(value.x) * DIVSHORTMAX;
        if (tsdf == 0 || tsdf <= -0.99f || tsdf >= 0.99f) continue;

        short2 vx = tsdf_volume.ptr((z) * volume_size.y + y)[x + 1];
        short2 vy = tsdf_volume.ptr((z) * volume_size.y + y + 1)[x];
        short2 vz = tsdf_volume.ptr((z + 1) * volume_size.y + y)[x];

        if (vx.y <= 0 || vy.y <= 0 || vz.y <= 0) continue;

        const float tsdf_x = static_cast<float>(vx.x) * DIVSHORTMAX;
        const float tsdf_y = static_cast<float>(vy.x) * DIVSHORTMAX;
        const float tsdf_z = static_cast<float>(vz.x) * DIVSHORTMAX;

        const bool is_surface_x = ((tsdf > 0) && (tsdf_x < 0)) || ((tsdf < 0) && (tsdf_x > 0));
        const bool is_surface_y = ((tsdf > 0) && (tsdf_y < 0)) || ((tsdf < 0) && (tsdf_y > 0));
        const bool is_surface_z = ((tsdf > 0) && (tsdf_z < 0)) || ((tsdf < 0) && (tsdf_z > 0));

        if (is_surface_x || is_surface_y || is_surface_z) {
            Eigen::Vector3f normal;
            normal.x() = (tsdf_x - tsdf);
            normal.y() = (tsdf_y - tsdf);
            normal.z() = (tsdf_z - tsdf);
            if (normal.norm() == 0) continue;
            normal.normalize();

            int count = 0;
            if (is_surface_x) count++;
            if (is_surface_y) count++;
            if (is_surface_z) count++;
            int index = atomicAdd(point_num, count);

            /*
            Vector3f_da position(
                (static_cast<float>(x) + 0.5f - volume_size.x / 2.f) * voxel_scale,
                (static_cast<float>(y) + 0.5f - volume_size.y / 2.f) * voxel_scale,
                (static_cast<float>(z) + 0.5f - volume_size.z / 2.f) * voxel_scale
            );
            */

            Vector3f_da position(
                (static_cast<float>(x) + 0.5f) * voxel_scale,
                (static_cast<float>(y) + 0.5f) * voxel_scale,
                (static_cast<float>(z) + 0.5f) * voxel_scale
            );
            if (is_surface_x) {
                position.x() = position.x() - (tsdf / (tsdf_x - tsdf)) * voxel_scale;

                vertices.ptr(0)[index] = float3{position(0), position(1), position(2)};
                normals.ptr(0)[index] = float3{normal(0), normal(1), normal(2)};
                index++;
            }
            if (is_surface_y) {
                position.y() -= (tsdf / (tsdf_y - tsdf)) * voxel_scale;

                vertices.ptr(0)[index] = float3{position(0), position(1), position(2)};;
                normals.ptr(0)[index] = float3{normal(0), normal(1), normal(2)};
                index++;
            }
            if (is_surface_z) {
                position.z() -= (tsdf / (tsdf_z - tsdf)) * voxel_scale;

                vertices.ptr(0)[index] = float3{position(0), position(1), position(2)};;
                normals.ptr(0)[index] = float3{normal(0), normal(1), normal(2)};
                index++;
            }
        }
    }
}

PointCloud extract_points(const TSDFData& volume, const int buffer_size)
{
    CloudData cloud_data { buffer_size };

    dim3 threads(32, 32);
    dim3 blocks(
        (volume.volume_size.x + threads.x - 1) / threads.x,
        (volume.volume_size.y + threads.y - 1) / threads.y
    );

    extract_points_kernel<<<blocks, threads>>>(
        volume.tsdf,
        volume.volume_size, volume.voxel_scale,
        cloud_data.vertices, cloud_data.normals, cloud_data.point_num
    );

    cudaThreadSynchronize();

    return cloud_data.download();
}
