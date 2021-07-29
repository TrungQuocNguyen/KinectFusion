#include <cuda/kernel_common.cuh>
#include <datatypes.hpp>


__global__ void kernel_update_tsdf(
    const PtrStepSz<float> depth, 
    const CameraParameters cam, const Matrix3f_da R_c_w, const Vector3f_da t_c_w,
    const int3 volume_size, const float voxel_scale,
    const float truncation_distance,
    PtrStepSz<short2> tsdf_volume
) 
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= volume_size.x || y >= volume_size.y) return;

    for (int z = 0; z < volume_size.z; ++z)
    {
        // 0.5f is for the centor of volume
        const Vector3f_da voxel_w(
            (static_cast<float>(x) + 0.5f - volume_size.x / 2.f) * voxel_scale,
            (static_cast<float>(y) + 0.5f - volume_size.y / 2.f) * voxel_scale,
            (static_cast<float>(z) + 0.5f - volume_size.z / 2.f) * voxel_scale
        );

        const Vector3f_da voxel_c = R_c_w * voxel_w + t_c_w;
        if (voxel_c[2] <= EPSILON) continue;

        const Vector2i_da uv(
            __float2int_rn(voxel_c[0] / voxel_c[2] * cam.fx + cam.cx),
            __float2int_rn(voxel_c[1] / voxel_c[2] * cam.fy + cam.cy)
        );  // project on current frame
        if (uv[0] < 0 || uv[0] >= depth.cols || uv[1] < 0 || uv[1] >= depth.rows) continue;

        const float d = depth.ptr(uv[1])[uv[0]];
        if (d < cam.min_depth || d > cam.max_depth) continue;  // in mm

        const Vector3f_da lambda((uv[0] - cam.cx) / cam.fx, (uv[1] - cam.cy) / cam.fy, 1.f);
        const float sdf = (d - voxel_c.norm() / lambda.norm()) / truncation_distance;
        
        if (sdf < -2) break;
        float tsdf = sdf > 0 ? fmin(1.f, sdf) : fmax(-1.f, sdf);
        // const short weight = 1;
        short weight = max(static_cast<short>(cam.max_depth / d * 0.5f), 1);

        const short2* model_voxel = &tsdf_volume.ptr(z * volume_size.y + y)[x];  // (tsdf, weight)
        const float model_tsdf = static_cast<float>(model_voxel->x) * INV_SHORT_MAX;
        const short model_weight = model_voxel->y;

        const float updated_tsdf = (model_weight * model_tsdf + weight * tsdf) / (model_weight + weight);

        const int new_tsdf = max(-SHORT_MAX, min(SHORT_MAX, static_cast<int>(updated_tsdf * SHORT_MAX)));
        const int new_weight = min(model_weight + weight, MAX_WEIGHT);

        // save as short
        tsdf_volume.ptr(z * volume_size.y + y)[x] = make_short2(static_cast<short>(new_tsdf), static_cast<short>(new_weight));
    }
}


void surfaceReconstruction(
    const GpuMat& depth,
    const CameraParameters& cam, const Eigen::Matrix4f& T_w_c,
    TSDFData& tsdf_data
)
{
    const dim3 threads(32, 32);
    const dim3 blocks(divUp(tsdf_data.volume_size.x, threads.x), divUp(tsdf_data.volume_size.y, threads.y));
    kernel_update_tsdf<<<blocks, threads>>>(
        depth, cam, 
        T_w_c.block<3, 3>(0, 0).transpose(), - T_w_c.block<3, 3>(0, 0).transpose() * T_w_c.block<3, 1>(0, 3),
        tsdf_data.volume_size, tsdf_data.voxel_scale,
        tsdf_data.truncation_distance,
        tsdf_data.tsdf
    );

    cudaThreadSynchronize();
}
