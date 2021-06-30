#include <opencv2/core/cuda.hpp>
#include <datatypes.hpp>


using cv::cuda::GpuMat;


__global__ void kernel_compute_vertex_map(const cv::cuda::PtrStepSz<float> depth_map, cv::cuda::PtrStepSz<float3> vertex_map, const CameraIntrinsics camera_params, const float max_depth){
    // Calculate global row and column for each thread
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= depth_map.cols || row >= depth_map.rows)
        return;

    float depth_val = depth_map(row, col);

    // Don't use depth values larger than max_depth
    if (depth_val > max_depth){
        depth_val = 0.f;
    } 

    // from screen to camera space
    vertex_map(row, col) = make_float3((col - camera_params.cx) * depth_val / camera_params.fx,
                                       (row - camera_params.cy) * depth_val / camera_params.fy,
                                       depth_val);
}

__global__ void kernel_compute_normal_map(cv::cuda::PtrStepSz<float3> vertex_map, cv::cuda::PtrStepSz<float3> normal_map){
    // Calculate global row and column for each thread
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= vertex_map.cols - 1 || row >= vertex_map.rows - 1){
        if (col == vertex_map.cols - 1 || row == vertex_map.rows - 1){
            normal_map(row, col) = make_float3(0.f,0.f,0.f);            // TODO: maybe compute them with vertex_map(row - 1, col) etc.
        }
        return;
    }
    float sx = vertex_map(row + 1, col).x - vertex_map(row, col).x;
    float sy = vertex_map(row + 1, col).y - vertex_map(row, col).y;
    float sz = vertex_map(row + 1, col).z - vertex_map(row, col).z;
    float tx = vertex_map(row, col + 1).x - vertex_map(row, col).x;
    float ty = vertex_map(row, col + 1).y - vertex_map(row, col).y;
    float tz = vertex_map(row, col + 1).z - vertex_map(row, col).z;

    float3 cross_prod = make_float3(sy * tz - sz * ty, 
                                    sz * tx - sx * tz,
                                    sx * ty - sy * tx);

    float norm = sqrt(cross_prod.x * cross_prod.x + cross_prod.y * cross_prod.y + cross_prod.z * cross_prod.z) + .000001f;

    normal_map(row, col) = make_float3(cross_prod.x / norm,
                                       cross_prod.y / norm,
                                       cross_prod.z / norm);
}

void compute_vertex_map(const GpuMat& filtered_depth_map, GpuMat& vertex_map, const CameraIntrinsics camera_params, const float max_depth){
    int threads = 32;
    dim3 T(threads, threads, 1);      // number of threads per block (depends on compute capability of your GPU)
    int blocks_x = (filtered_depth_map.cols + T.x - 1) / T.x;
    int blocks_y = (filtered_depth_map.rows + T.y - 1) / T.y;
    dim3 M(blocks_x, blocks_y, 1);       // number of thread blocks (depends on compute capability of your GPU)
    kernel_compute_vertex_map<<< M , T >>>(filtered_depth_map, vertex_map, camera_params, max_depth);
    cudaDeviceSynchronize();
}

void compute_normal_map(const GpuMat& vertex_map, GpuMat& normal_map){
    int threads = 32;
    dim3 T(threads, threads, 1);      // number of threads per block (depends on compute capability of your GPU)
    int blocks_x = (vertex_map.cols + T.x - 1) / T.x;
    int blocks_y = (vertex_map.rows + T.y - 1) / T.y;
    dim3 M(blocks_x, blocks_y, 1);       // number of thread blocks (depends on compute capability of your GPU)
    kernel_compute_normal_map<<< M , T >>>(vertex_map, normal_map);
    cudaDeviceSynchronize();
}

