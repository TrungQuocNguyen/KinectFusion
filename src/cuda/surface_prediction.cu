#include <cuda/kernel_common.cuh>
#include <datatypes.hpp>


__device__ __forceinline__
float interpolate_trilinearly(
    const Vector3f_da& point, const PtrStep<short2>& volume,
    const int3& volume_size, const float& voxel_scale
) 
{
    Vector3i_da point_in_grid = point.cast<int>();

    const float vx = (static_cast<float>(point_in_grid[0]) + 0.5f);
    const float vy = (static_cast<float>(point_in_grid[1]) + 0.5f);
    const float vz = (static_cast<float>(point_in_grid[2]) + 0.5f);

    point_in_grid.x() = (point[0] < vx) ? (point_in_grid[0] - 1) : point_in_grid[0];
    point_in_grid.y() = (point[1] < vy) ? (point_in_grid[1] - 1) : point_in_grid[1];
    point_in_grid.z() = (point[2] < vz) ? (point_in_grid[2] - 1) : point_in_grid[2];

    const float a = point.x() - (static_cast<float>(point_in_grid.x()) + 0.5f);
    const float b = point.y() - (static_cast<float>(point_in_grid.y()) + 0.5f);
    const float c = point.z() - (static_cast<float>(point_in_grid.z()) + 0.5f);

    const short2* v0 = volume.ptr((point_in_grid[2]) * volume_size.y + point_in_grid[1]);
    const short2* v1 = volume.ptr((point_in_grid[2] + 1) * volume_size.y + point_in_grid[1]);
    const short2* v2 = volume.ptr((point_in_grid[2]) * volume_size.y + point_in_grid[1] + 1);
    const short2* v3 = volume.ptr((point_in_grid[2] + 1) * volume_size.y + point_in_grid[1] + 1);

    return 
        static_cast<float>(v0[point_in_grid[0]].x) * INV_SHORT_MAX * (1 - a) * (1 - b) * (1 - c) +
        static_cast<float>(v1[point_in_grid[0]].x) * INV_SHORT_MAX * (1 - a) * (1 - b) * c +
        static_cast<float>(v2[point_in_grid[0]].x) * INV_SHORT_MAX * (1 - a) * b * (1 - c) +
        static_cast<float>(v3[point_in_grid[0]].x) * INV_SHORT_MAX * (1 - a) * b * c +
        static_cast<float>(v0[point_in_grid[0] + 1].x) * INV_SHORT_MAX * a * (1 - b) * (1 - c) +
        static_cast<float>(v1[point_in_grid[0] + 1].x) * INV_SHORT_MAX * a * (1 - b) * c +
        static_cast<float>(v2[point_in_grid[0] + 1].x) * INV_SHORT_MAX * a * b * (1 - c) +
        static_cast<float>(v3[point_in_grid[0] + 1].x) * INV_SHORT_MAX * a * b * c;
}


__device__ __forceinline__ float get_min_time(const float3& volume_max, const Vector3f_da& origin, const Vector3f_da& direction)
{
    float txmin = ((direction.x() > 0 ? 0.f : volume_max.x) - origin.x()) / direction.x();
    float tymin = ((direction.y() > 0 ? 0.f : volume_max.y) - origin.y()) / direction.y();
    float tzmin = ((direction.z() > 0 ? 0.f : volume_max.z) - origin.z()) / direction.z();

    return fmax(fmax(txmin, tymin), tzmin);
}


__device__ __forceinline__ float get_max_time(const float3& volume_max, const Vector3f_da& origin, const Vector3f_da& direction)
{
    float txmax = ((direction.x() > 0 ? volume_max.x : 0.f) - origin.x()) / direction.x();
    float tymax = ((direction.y() > 0 ? volume_max.y : 0.f) - origin.y()) / direction.y();
    float tzmax = ((direction.z() > 0 ? volume_max.z : 0.f) - origin.z()) / direction.z();

    return fmin(fmin(txmax, tymax), tzmax);
}


__device__ __forceinline__ 
void get_min_pos(const Vector3f_da& volume_hsize, const Vector3f_da& pos, const Vector3f_da& direction, Vector3f_da& min_voxel_pos)
{
    if (direction[0] < EPSILON && direction[0] > - EPSILON)
    {
        min_voxel_pos[0] = - volume_hsize[0];
    }
    else
    {
        min_voxel_pos[0] = ((direction[0] > 0 ? volume_hsize[0] : - volume_hsize[0]) - pos[0]) / direction[0];
    }
    if (direction[1] < EPSILON && direction[1] > - EPSILON)
    {
        min_voxel_pos[1] = - volume_hsize[1];
    }
    else
    {
        min_voxel_pos[1] = ((direction[1] > 0 ? volume_hsize[1] : - volume_hsize[1]) - pos[1]) / direction[1];
    }
    if (direction[2] < EPSILON && direction[2] > - EPSILON)
    {
        min_voxel_pos[2] = - volume_hsize[2];
    }
    else
    {
        min_voxel_pos[2] = ((direction[2] > 0 ? volume_hsize[2] : - volume_hsize[2]) - pos[2]) / direction[2];
    }
}


__device__ __forceinline__ 
void get_max_pos(const Vector3f_da& volume_hsize, const Vector3f_da& pos, const Vector3f_da& direction, Vector3f_da& max_voxel_pos)
{
    if (direction[0] < EPSILON && direction[0] > - EPSILON)
    {
        max_voxel_pos[0] = volume_hsize[0];
    }
    else
    {
        max_voxel_pos[0] = ((direction[0] > 0 ? - volume_hsize[0] : volume_hsize[0]) - pos[0]) / direction[0];
    }
    if (direction[1] < EPSILON && direction[1] > - EPSILON)
    {
        max_voxel_pos[1] = - volume_hsize[1];
    }
    else
    {
        max_voxel_pos[1] = ((direction[1] > 0 ? - volume_hsize[1] : volume_hsize[1]) - pos[1]) / direction[1];
    }
    if (direction[2] < EPSILON && direction[2] > - EPSILON)
    {
        max_voxel_pos[2] = - volume_hsize[2];
    }
    else
    {
        max_voxel_pos[2] = ((direction[2] > 0 ? - volume_hsize[2] : volume_hsize[2]) - pos[2]) / direction[2];
    }
}


__global__
void kernel_raycast_tsdf(
    const PtrStep<short2> tsdf_volume,
    const CameraParameters cam, const Matrix3f_da R_w_c, const Vector3f_da t_w_c,
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

    const Vector3f_da offset(volume_size.x / 2.f * voxel_scale, volume_size.y / 2.f * voxel_scale, volume_size.z / 2.f * voxel_scale);

    const Vector3f_da ray_c((x - cam.cx) / cam.fx, (y - cam.cy) / cam.fy, 1.f);  // in camera coordinate
    const Vector3f_da ray_direction = (R_w_c * ray_c).normalized();  // in world coordinate

    float min_length = 0.f;
    const Vector3f_da voxel_w = t_w_c + offset;

    Vector3f_da min_voxel_pos, max_voxel_pos;
    get_min_pos(offset, t_w_c, ray_direction, min_voxel_pos);
    get_max_pos(offset, t_w_c, ray_direction, max_voxel_pos);
    
    float ray_length = fmax(get_min_time(volume_range, t_w_c + offset, ray_direction), 0.f);
    const float max_length = get_max_time(volume_range, t_w_c + offset, ray_direction);
    if (ray_length >= max_length) return;

    ray_length += voxel_scale / 2.f;

    Vector3f_da grid = (t_w_c + ray_direction * ray_length + offset) / voxel_scale;

    if (grid[0] < 0 || grid[0] > volume_size.x - 1 ||
        grid[1] < 0 || grid[1] > volume_size.y - 1 ||
        grid[2] < 0 || grid[2] > volume_size.z - 1) return;
    
    float tsdf = static_cast<float>(
        tsdf_volume.ptr(__float2int_rd(grid[2]) * volume_size.y + __float2int_rd(grid[1]))[__float2int_rd(grid[0])].x
    ) * INV_SHORT_MAX;
    if (tsdf < 0.f) return;

    const float max_search_length = max_length;
    for (; ray_length < max_search_length; ray_length += truncation_distance * 0.5f)
    {
        grid += ray_direction * truncation_distance * 0.5f / voxel_scale;

        if (grid[0] < 0 || grid[0] > volume_size.x - 1 ||
            grid[1] < 0 || grid[1] > volume_size.y - 1 ||
            grid[2] < 0 || grid[2] > volume_size.z - 1) continue;

        const float previous_tsdf = tsdf;
        tsdf = static_cast<float>(
            tsdf_volume.ptr(__float2int_rd(grid[2]) * volume_size.y + __float2int_rd(grid[1]))[__float2int_rd(grid[0])].x
        ) * INV_SHORT_MAX;

        if (previous_tsdf < 0.f) return;
        if (previous_tsdf > 0.f && tsdf < 0.f)
        {
            //Zero crossing
            const float t_star = ray_length - truncation_distance * 0.5f * previous_tsdf / (tsdf - previous_tsdf);
            const auto vertex_w = t_w_c + ray_direction * t_star;

            const Vector3f_da location_in_grid = (vertex_w + offset) / voxel_scale;
            if (location_in_grid[0] < 1 || location_in_grid[0] >= volume_size.x - 1 ||
                location_in_grid[1] < 1 || location_in_grid[1] >= volume_size.y - 1 ||
                location_in_grid[2] < 1 || location_in_grid[2] >= volume_size.z - 1) return;
            
            Vector3f_da normal_w, shifted;

            shifted = location_in_grid;
            shifted[0] += 1;
            if (shifted[0] >= volume_size.x - 1) break;
            const float Fx1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            shifted = location_in_grid;
            shifted.x() -= 1;
            if (shifted.x() < 1) break;
            const float Fx2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            normal_w[0] = (Fx1 - Fx2);

            shifted = location_in_grid;
            shifted.y() += 1;
            if (shifted.y() >= volume_size.y - 1) break;
            const float Fy1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            shifted = location_in_grid;
            shifted.y() -= 1;
            if (shifted.y() < 1) break;
            const float Fy2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            normal_w[1] = (Fy1 - Fy2);

            shifted = location_in_grid;
            shifted.z() += 1;
            if (shifted.z() >= volume_size.z - 1) break;
            const float Fz1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            shifted = location_in_grid;
            shifted.z() -= 1;
            if (shifted.z() < 1) break;
            const float Fz2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            normal_w[2] = (Fz1 - Fz2);

            if (normal_w.norm() < EPSILON) break;

            normal_w.normalize();

            vertex_map.ptr(y)[x] = make_float3(vertex_w.x(), vertex_w.y(), vertex_w.z());
            normal_map.ptr(y)[x] = make_float3(normal_w.x(), normal_w.y(), normal_w.z());
            break;
        }
    }
}


void raycastTSDF(
    const TSDFData &tsdf_data,
    const CameraParameters &cam,
    const Eigen::Matrix4f &T_w_c,
    GpuMat &vertex_map, GpuMat &normal_map
)
{
    vertex_map.setTo(0);
    normal_map.setTo(0);

    const dim3 threads(32, 32);
    const dim3 blocks(divUp(vertex_map.cols, threads.x), divUp(vertex_map.rows, threads.y));

    kernel_raycast_tsdf<<<blocks, threads>>>(
        tsdf_data.tsdf, cam, 
        T_w_c.block<3, 3>(0, 0), T_w_c.block<3, 1>(0, 3),
        tsdf_data.volume_size, tsdf_data.voxel_scale,
        tsdf_data.truncation_distance,
        vertex_map, normal_map
    );

    cudaThreadSynchronize();
}
