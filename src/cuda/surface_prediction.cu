#include <cuda/kernel_common.cuh>
#include <datatypes.hpp>


__device__ __forceinline__ float interpolate_trilinearly(
    const Vector3f_da& point, const PtrStep<short2>& volume,
    const int3& volume_size, const float& voxel_scale
) {
    Vector3i_da point_in_grid = point.cast<int>();

    const float vx = (static_cast<float>(point_in_grid.x()) + 0.5f);
    const float vy = (static_cast<float>(point_in_grid.y()) + 0.5f);
    const float vz = (static_cast<float>(point_in_grid.z()) + 0.5f);

    point_in_grid.x() = (point.x() < vx) ? (point_in_grid.x() - 1) : point_in_grid.x();
    point_in_grid.y() = (point.y() < vy) ? (point_in_grid.y() - 1) : point_in_grid.y();
    point_in_grid.z() = (point.z() < vz) ? (point_in_grid.z() - 1) : point_in_grid.z();

    const float a = point.x() - (static_cast<float>(point_in_grid.x()) + 0.5f);
    const float b = point.y() - (static_cast<float>(point_in_grid.y()) + 0.5f);
    const float c = point.z() - (static_cast<float>(point_in_grid.z()) + 0.5f);

    return static_cast<float>(volume.ptr((point_in_grid.z()) * volume_size.y + point_in_grid.y())[point_in_grid.x()].x) * INV_SHORT_MAX * (1 - a) * (1 - b) * (1 - c) +
        static_cast<float>(volume.ptr((point_in_grid.z() + 1) * volume_size.y + point_in_grid.y())[point_in_grid.x()].x) * INV_SHORT_MAX * (1 - a) * (1 - b) * c +
        static_cast<float>(volume.ptr((point_in_grid.z()) * volume_size.y + point_in_grid.y() + 1)[point_in_grid.x()].x) * INV_SHORT_MAX * (1 - a) * b * (1 - c) +
        static_cast<float>(volume.ptr((point_in_grid.z() + 1) * volume_size.y + point_in_grid.y() + 1)[point_in_grid.x()].x) * INV_SHORT_MAX * (1 - a) * b * c +
        static_cast<float>(volume.ptr((point_in_grid.z()) * volume_size.y + point_in_grid.y())[point_in_grid.x() + 1].x) * INV_SHORT_MAX * a * (1 - b) * (1 - c) +
        static_cast<float>(volume.ptr((point_in_grid.z() + 1) * volume_size.y + point_in_grid.y())[point_in_grid.x() + 1].x) * INV_SHORT_MAX * a * (1 - b) * c +
        static_cast<float>(volume.ptr((point_in_grid.z()) * volume_size.y + point_in_grid.y() + 1)[point_in_grid.x() + 1].x) * INV_SHORT_MAX * a * b * (1 - c) +
        static_cast<float>(volume.ptr((point_in_grid.z() + 1) * volume_size.y + point_in_grid.y() + 1)[point_in_grid.x() + 1].x) * INV_SHORT_MAX * a * b * c;
}


__device__ __forceinline__ float get_min_time(
    const float3& volume_max, const Vector3f_da& origin, const Vector3f_da& direction
)
{
    float txmin = ((direction.x() > 0 ? 0.f : volume_max.x) - origin.x()) / direction.x();
    float tymin = ((direction.y() > 0 ? 0.f : volume_max.y) - origin.y()) / direction.y();
    float tzmin = ((direction.z() > 0 ? 0.f : volume_max.z) - origin.z()) / direction.z();

    return fmax(fmax(txmin, tymin), tzmin);
}


__device__ __forceinline__ float get_max_time(
    const float3& volume_max, const Vector3f_da& origin, const Vector3f_da& direction
)
{
    float txmax = ((direction.x() > 0 ? volume_max.x : 0.f) - origin.x()) / direction.x();
    float tymax = ((direction.y() > 0 ? volume_max.y : 0.f) - origin.y()) / direction.y();
    float tzmax = ((direction.z() > 0 ? volume_max.z : 0.f) - origin.z()) / direction.z();

    return fmin(fmin(txmax, tymax), tzmax);
}


__global__ void kernel_raycast_tsdf(
    const PtrStep<short2> tsdf_volume,
    const CameraParameters cam, const Matrix3f_da rotation, const Vector3f_da translation,
    const int3 volume_size, const float voxel_scale, const float truncation_distance,
    PtrStepSz<float3> vertex_map, PtrStep<float3> normal_map
) 
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= vertex_map.cols || y >= vertex_map.rows) return;

    const float3 volume_range = make_float3(
        volume_size.x * voxel_scale, volume_size.y * voxel_scale, volume_size.z * voxel_scale
    );

    const Vector3f_da offset(volume_size.x / 2.f, volume_size.y / 2.f, volume_size.z / 2.f);

    const Vector3f_da pixel_position((x - cam.cx) / cam.fx, (y - cam.cy) / cam.fy, 1.f);
    const Vector3f_da ray_direction = (rotation * pixel_position).normalized();

    float ray_length = fmax(get_min_time(volume_range, translation + offset * voxel_scale, ray_direction), 0.f);
    if (ray_length >= get_max_time(volume_range, translation + offset * voxel_scale, ray_direction)) return;

    ray_length += voxel_scale;

    Vector3f_da grid = (translation + ray_direction * ray_length) / voxel_scale + offset;

    if (grid[0] < 0 || grid[0] > volume_size.x - 1 ||
        grid[1] < 0 || grid[1] > volume_size.y - 1 ||
        grid[2] < 0 || grid[2] > volume_size.z - 1) return;

    float tsdf = static_cast<float>(
        tsdf_volume.ptr(__float2int_rd(grid[2]) * volume_size.y + __float2int_rd(grid[1]))[__float2int_rd(grid[0])].x
    ) * INV_SHORT_MAX;

    const float max_search_length = ray_length + volume_range.x * sqrt(2.f);
    for (; ray_length < max_search_length; ray_length += truncation_distance * 0.5f)
    {
        grid = (translation + ray_direction * (ray_length + truncation_distance * 0.5f)) / voxel_scale + offset;

        if (grid[0] < 0 || grid[0] > volume_size.x - 1 ||
            grid[1] < 0 || grid[1] > volume_size.y - 1 ||
            grid[2] < 0 || grid[2] > volume_size.z - 1) continue;

        const float previous_tsdf = tsdf;
        tsdf = static_cast<float>(
            tsdf_volume.ptr(__float2int_rd(grid[2]) * volume_size.y + __float2int_rd(grid[1]))[__float2int_rd(grid[0])].x
        ) * INV_SHORT_MAX;

        if (previous_tsdf < 0.f && tsdf > 0.f) break;  //Zero crossing from behind
        if (previous_tsdf > 0.f && tsdf < 0.f)
        {
            //Zero crossing
            const float t_star = ray_length - truncation_distance * 0.5f * previous_tsdf / (tsdf - previous_tsdf);
            const auto vertex = translation + ray_direction * t_star;

            const Vector3f_da location_in_grid = vertex / voxel_scale + offset;
            if (location_in_grid.x() < 1 || location_in_grid.x() >= volume_size.x - 1 ||
                location_in_grid.y() < 1 || location_in_grid.y() >= volume_size.y - 1 ||
                location_in_grid.z() < 1 || location_in_grid.z() >= volume_size.z - 1) break;
            
            Vector3f_da normal, shifted;

            shifted = location_in_grid;
            shifted.x() += 1;
            if (shifted.x() >= volume_size.x - 1) break;
            const float Fx1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            shifted = location_in_grid;
            shifted.x() -= 1;
            if (shifted.x() < 1) break;
            const float Fx2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            normal.x() = (Fx1 - Fx2);

            shifted = location_in_grid;
            shifted.y() += 1;
            if (shifted.y() >= volume_size.y - 1) break;
            const float Fy1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            shifted = location_in_grid;
            shifted.y() -= 1;
            if (shifted.y() < 1) break;
            const float Fy2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            normal.y() = (Fy1 - Fy2);

            shifted = location_in_grid;
            shifted.z() += 1;
            if (shifted.z() >= volume_size.z - 1) break;
            const float Fz1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            shifted = location_in_grid;
            shifted.z() -= 1;
            if (shifted.z() < 1) break;
            const float Fz2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            normal.z() = (Fz1 - Fz2);

            if (normal.norm() < EPSILON) break;

            normal.normalize();

            vertex_map.ptr(y)[x] = make_float3(vertex.x(), vertex.y(), vertex.z());
            normal_map.ptr(y)[x] = make_float3(normal.x(), normal.y(), normal.z());
            break;
        }
    }
}


__global__ void kernel_raycast_tsdf_using_depth(
    const PtrStepSz<short2> tsdf_volume, const PtrStepSz<float> depth,
    const CameraParameters cam, const Matrix3f_da rotation, const Vector3f_da translation,
    const int3 volume_size, const float voxel_scale, const float truncation_distance,
    PtrStepSz<float3> vertex_map, PtrStepSz<float3> normal_map
) 
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= vertex_map.cols || y >= vertex_map.rows) return;

    const float3 volume_range = make_float3(
        volume_size.x * voxel_scale, volume_size.y * voxel_scale, volume_size.z * voxel_scale
    );

    const Vector3f_da offset(volume_size.x / 2.f, volume_size.y / 2.f, volume_size.z / 2.f);

    const Vector3f_da pixel_position((x - cam.cx) / cam.fx, (y - cam.cy) / cam.fy, 1.f);
    const Vector3f_da ray_direction = (rotation * pixel_position).normalized();

    float ray_length = depth.ptr(y)[x];
    if (depth.ptr(y)[x] < cam.min_depth || depth.ptr(y)[x] > cam.max_depth)
    {
        ray_length = fmax(get_min_time(volume_range, translation + offset * voxel_scale, ray_direction), 0.f);
    }

    if (ray_length >= get_max_time(volume_range, translation + offset * voxel_scale, ray_direction)) return;

    Vector3f_da grid = (translation + ray_direction * ray_length) / voxel_scale + offset;

    float tsdf = static_cast<float>(
        tsdf_volume.ptr(__float2int_rd(grid[2]) * volume_size.y + __float2int_rd(grid[1]))[__float2int_rd(grid[0])].x
    ) * INV_SHORT_MAX;
    
    while (tsdf < 0)
    {
        ray_length *= 0.8f;
        grid = (translation + ray_direction * ray_length) / voxel_scale + offset;
        tsdf = static_cast<float>(
            tsdf_volume.ptr(__float2int_rd(grid[2]) * volume_size.y + __float2int_rd(grid[1]))[__float2int_rd(grid[0])].x
        ) * INV_SHORT_MAX;
    }

    const float max_search_length = ray_length + volume_range.x * sqrt(2.f);
    for (; ray_length < max_search_length; ray_length += truncation_distance * 0.5f)
    {
        grid = (translation + (ray_direction * ray_length)) / voxel_scale + offset;

        if (grid[0] < 0 || grid[0] > volume_size.x - 1 ||
            grid[1] < 0 || grid[1] > volume_size.y - 1 ||
            grid[2] < 0 || grid[2] > volume_size.z - 1) continue;

        const float previous_tsdf = tsdf;
        tsdf = static_cast<float>(
            tsdf_volume.ptr(__float2int_rd(grid[2]) * volume_size.y + __float2int_rd(grid[1]))[__float2int_rd(grid[0])].x
        ) * INV_SHORT_MAX;

        if (previous_tsdf < 0.f && tsdf > 0.f) break;  //Zero crossing from behind
        if (previous_tsdf > 0.f && tsdf < 0.f)
        {
            //Zero crossing
            const float t_star = ray_length - truncation_distance * 0.5f * previous_tsdf / (tsdf - previous_tsdf);
            const auto vertex = translation + ray_direction * t_star;

            const Vector3f_da location_in_grid = vertex / voxel_scale + offset;
            if (location_in_grid.x() < 1 || location_in_grid.x() >= volume_size.x - 1 ||
                location_in_grid.y() < 1 || location_in_grid.y() >= volume_size.y - 1 ||
                location_in_grid.z() < 1 || location_in_grid.z() >= volume_size.z - 1) break;
            
            Vector3f_da normal, shifted;

            shifted = location_in_grid;
            shifted.x() += 1;
            if (shifted.x() >= volume_size.x - 1) break;
            const float Fx1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            shifted = location_in_grid;
            shifted.x() -= 1;
            if (shifted.x() < 1) break;
            const float Fx2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            normal.x() = (Fx1 - Fx2);

            shifted = location_in_grid;
            shifted.y() += 1;
            if (shifted.y() >= volume_size.y - 1) break;
            const float Fy1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            shifted = location_in_grid;
            shifted.y() -= 1;
            if (shifted.y() < 1) break;
            const float Fy2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            normal.y() = (Fy1 - Fy2);

            shifted = location_in_grid;
            shifted.z() += 1;
            if (shifted.z() >= volume_size.z - 1) break;
            const float Fz1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            shifted = location_in_grid;
            shifted.z() -= 1;
            if (shifted.z() < 1) break;
            const float Fz2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            normal.z() = (Fz1 - Fz2);

            if (normal.norm() == 0) break;

            normal.normalize();

            vertex_map.ptr(y)[x] = make_float3(vertex.x(), vertex.y(), vertex.z());
            normal_map.ptr(y)[x] = make_float3(normal.x(), normal.y(), normal.z());
            break;
        }
    }
}


void raycast_tsdf(
    const TSDFData &tsdf_data,
    const CameraParameters &cam,
    const Eigen::Matrix4f &T_c_w,
    const float &trancation_distance,
    GpuMat &vertex_map, GpuMat &normal_map
)
{
    vertex_map.setTo(0);
    normal_map.setTo(0);

    const dim3 threads(32, 32);
    const dim3 blocks(divUp(tsdf_data.volume_size.x, threads.x), divUp(tsdf_data.volume_size.y, threads.y));

    kernel_raycast_tsdf<<<blocks, threads>>>(
        tsdf_data.tsdf, cam, 
        T_c_w.block<3, 3>(0, 0), T_c_w.block<3, 1>(0, 3),
        tsdf_data.volume_size, tsdf_data.voxel_scale,
        trancation_distance,
        vertex_map, normal_map
    );

    cudaThreadSynchronize();
}


void raycast_tsdf_using_depth(
    const TSDFData &tsdf_data,
    const GpuMat &depth,
    const CameraParameters &cam,
    const Eigen::Matrix4f &T_c_w,
    const float &trancation_distance,
    GpuMat &vertex_map, GpuMat &normal_map
)
{
    vertex_map.setTo(0);
    normal_map.setTo(0);

    const dim3 threads(32, 32);
    const dim3 blocks(divUp(tsdf_data.volume_size.x, threads.x), divUp(tsdf_data.volume_size.y, threads.y));

    kernel_raycast_tsdf_using_depth<<<blocks, threads>>>(
        tsdf_data.tsdf, depth,
        cam, T_c_w.block<3, 3>(0, 0), T_c_w.block<3, 1>(0, 3),
        tsdf_data.volume_size, tsdf_data.voxel_scale,
        trancation_distance,
        vertex_map, normal_map
    );

    cudaThreadSynchronize();
}