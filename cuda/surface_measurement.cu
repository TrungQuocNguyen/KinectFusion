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

/*__global__ void kernel_compute_normal_map(const GpuMat& vertex_map, GpuMat& normal_map){
    // Calculate global row and column for each thread
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= vertex_map.cols || row >= vertex_map.rows)
        return;

    normal_map(row, col) = vertex_map() / 1;
}*/

void compute_vertex_map(const GpuMat& filtered_depth_map, GpuMat& vertex_map, const CameraIntrinsics camera_params, const float max_depth){
    int threads = 32;
    dim3 T(threads, threads, 1);      // number of threads per block (depends on compute capability of your GPU)
    int blocks_x = (filtered_depth_map.cols + T.x - 1) / T.x;
    int blocks_y = (filtered_depth_map.rows + T.y - 1) / T.y;
    dim3 M(blocks_x, blocks_y, 1);       // number of thread blocks (depends on compute capability of your GPU)
    kernel_compute_vertex_map<<< M , T >>>(filtered_depth_map, vertex_map, camera_params, max_depth);
    cudaDeviceSynchronize();
}

/*void compute_normal_map(const GpuMat& vertex_map, GpuMat& normal_map){
    int threads = 32;
    dim3 T(threads, threads, 1);      // number of threads per block (depends on compute capability of your GPU)
    int blocks_x = (vertex_map.cols + T.x - 1) / T.x;
    int blocks_y = (vertex_map.rows + T.y - 1) / T.y;
    dim3 M(blocks_x, blocks_y, 1);       // number of thread blocks (depends on compute capability of your GPU)
    kernel_compute_normal_map<<< M , T >>>(vertex_map, normal_map);
    cudaDeviceSynchronize();
}*/

