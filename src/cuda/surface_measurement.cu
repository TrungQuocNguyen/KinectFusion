#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev/common.hpp>
#include <datatypes.hpp>

using cv::cuda::PtrStepSz;
using cv::cuda::GpuMat;


__global__ void kernel_compute_vertex_map(
    const PtrStepSz<float> depth_map, const CameraParameters cam, PtrStepSz<float3> vertex_map
)
{
    // Calculate global row and column for each thread
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= depth_map.cols || row >= depth_map.rows)return;

    float depth_val = depth_map(row, col);

    // Don't use depth values larger than max_depth
    if (depth_val > cam.max_depth || depth_val < cam.min_depth)
    {
        depth_val = 0.f;
    } 

    // from screen to camera space
    vertex_map(row, col) = make_float3(
        (col - cam.cx) * depth_val / cam.fx,
        (row - cam.cy) * depth_val / cam.fy,
        depth_val
    );
}


__global__ void kernel_compute_normal_map(PtrStepSz<float3> vertex_map, PtrStepSz<float3> normal_map)
{
    // Calculate global row and column for each thread
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= vertex_map.cols - 1 || row >= vertex_map.rows - 1)
    {
        if (col == vertex_map.cols - 1 || row == vertex_map.rows - 1)
        {
            normal_map(row, col) = make_float3(0.f, 0.f, 0.f);  // TODO: maybe compute them with vertex_map(row - 1, col) etc.
        }
        return;
    }
    float sx = vertex_map(row + 1, col).x - vertex_map(row, col).x;
    float sy = vertex_map(row + 1, col).y - vertex_map(row, col).y;
    float sz = vertex_map(row + 1, col).z - vertex_map(row, col).z;
    float tx = vertex_map(row, col + 1).x - vertex_map(row, col).x;
    float ty = vertex_map(row, col + 1).y - vertex_map(row, col).y;
    float tz = vertex_map(row, col + 1).z - vertex_map(row, col).z;

    float3 cross_prod = make_float3(sy * tz - sz * ty, sz * tx - sx * tz, sx * ty - sy * tx);

    float norm = sqrt(cross_prod.x * cross_prod.x + cross_prod.y * cross_prod.y + cross_prod.z * cross_prod.z) + .000001f;

    normal_map(row, col) = make_float3(cross_prod.x / norm, cross_prod.y / norm, cross_prod.z / norm);
}


void compute_vertex_map(const GpuMat& filtered_depth_map, const CameraParameters cam, GpuMat& vertex_map)
{
    const dim3 blocks(32, 32);
    const dim3 grid(
        cv::cudev::divUp(filtered_depth_map.cols, blocks.x),
        cv::cudev::divUp(filtered_depth_map.rows, blocks.y)
    );
    kernel_compute_vertex_map<<<grid, blocks>>>(filtered_depth_map, cam, vertex_map);

    cudaDeviceSynchronize();
}


void compute_normal_map(const GpuMat& vertex_map, GpuMat& normal_map)
{
    const dim3 blocks(32, 32);
    const dim3 grid(
        cv::cudev::divUp(vertex_map.cols, blocks.x),
        cv::cudev::divUp(vertex_map.rows, blocks.y)
    );
    kernel_compute_normal_map<<<grid, blocks>>>(vertex_map, normal_map);

    cudaDeviceSynchronize();
}
