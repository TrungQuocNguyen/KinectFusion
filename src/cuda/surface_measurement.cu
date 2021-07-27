#include <cuda/kernel_common.cuh>
#include <datatypes.hpp>


__global__ void kernel_compute_vertex_map(
    const PtrStepSz<float> depth_map, const CameraParameters cam, PtrStep<float3> vertex_map
)
{
    // Calculate global row and column for each thread
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= depth_map.cols || row >= depth_map.rows) return;

    float d = depth_map.ptr(row)[col];
    if (d > cam.max_depth || d < cam.min_depth)
    {
        vertex_map.ptr(row)[col] = make_float3(0, 0, 0);
    }
    else
    {
        vertex_map.ptr(row)[col] = make_float3(
            (col - cam.cx) * d / cam.fx,
            (row - cam.cy) * d / cam.fy,
            d
        );
    }
}


__global__ void kernel_compute_normal_map(const PtrStepSz<float3> vertex_map, PtrStep<float3> normal_map)
{
    // Calculate global row and column for each threads
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= vertex_map.cols - 1 || row >= vertex_map.rows - 1) return;

    const float3* center = &vertex_map.ptr(row)[col];
    const float3* right = &vertex_map.ptr(row + 1)[col];
    const float3* down = &vertex_map.ptr(row)[col + 1];

    if (center->z <= EPSILON || right->z <= EPSILON || down->z <= EPSILON)
    {
        normal_map.ptr(row)[col] = make_float3(0, 0, 0);
        return;
    }

    const float3 s = make_float3(right->x - center->x, right->y - center->y, right->z - center->z);
    const float3 t = make_float3(down->x - center->x, down->y - center->y, down->z - center->z);

    const float3 cross = make_float3(s.y * t.z - s.z * t.y, s.z * t.x - s.x * t.z, s.x * t.y - s.y * t.x);
    const float norm = sqrt(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z);

    if (norm < EPSILON)
    {
        normal_map.ptr(row)[col] = make_float3(0, 0, 0);
    }
    else
    {
        normal_map.ptr(row)[col] = make_float3(cross.x / norm, cross.y / norm, cross.z / norm);
    }
}


void computeVertexMap(const GpuMat& depth_map, const CameraParameters& cam, GpuMat& vertex_map)
{
    const dim3 blocks(32, 32);
    const dim3 grid(divUp(depth_map.cols, blocks.x), divUp(depth_map.rows, blocks.y));
    kernel_compute_vertex_map<<<grid, blocks>>>(depth_map, cam, vertex_map);

    cudaDeviceSynchronize();
}


void computeNormalMap(const GpuMat& vertex_map, GpuMat& normal_map)
{
    const dim3 blocks(32, 32);
    const dim3 grid(divUp(vertex_map.cols, blocks.x), divUp(vertex_map.rows, blocks.y));
    kernel_compute_normal_map<<<grid, blocks>>>(vertex_map, normal_map);

    cudaDeviceSynchronize();
}
