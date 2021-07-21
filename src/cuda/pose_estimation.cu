#include "cuda_runtime.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev/common.hpp>
#include <Eigen/Core>
#include "datatypes.hpp"
#include "device_launch_parameters.h"

using cv::cuda::PtrStep;
using Vector2i_da = Eigen::Matrix<int, 2, 1, Eigen::DontAlign>;
using Vector3f_da = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using Matrix3f_da = Eigen::Matrix<float, 3, 3, Eigen::DontAlign>;

/*
__global__ void kernel_pose_estimate(
    const PtrStep<float3> prev_vertex_map, const PtrStep<float3> prev_normal_map,
    const PtrStep<float3> vertex_map, const PtrStep<float3> normal_map,
    const Matrix3f_da prev_rotation, const Vector3f_da prev_translation,
    const Matrix3f_da rotation, const Vector3f_da translation,
    const CameraParameters cam,
    const float distance_threshold, const float angle_threshold,
    PtrStep<double> global_buffer
)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > cam.width || y < cam.height) return;

    float row[7];

    Vector3f_da normal;
    normal[0] = normal_map.ptr(y)[x].x;
    if (isnan(normal[0])) return;
    
    Vector3f_da vertex(vertex_map.ptr(y)[x].x, vertex_map.ptr(y)[x].y, vertex_map.ptr(y)[x].z);
    Vector3f_da global_vertex = rotation * vertex + prev_translation;
    Vector3f_da vertex_camera = (global_vertex - prev_translation) * prev_rotation;
    Vector2i_da point(
        __float2int_rd(vertex_camera[0] * cam.fx / vertex_camera[2] + cam.cx + 0.5f),
        __float2int_rd(vertex_camera[1] * cam.fy / vertex_camera[2] + cam.cy + 0.5f)
    );

    if (point[0] < 0 || point[0] >= cam.width || point[1] < 0 || point[1] >= cam.height || vertex_camera[2] < 0) return;

    Vector3f_da prev_global_normal;
    prev_global_normal[0] = prev_normal_map.ptr(point[1])[point[0]].x;

    if (isnan(prev_global_normal[0])) return;
    
    Vector3f_da prev_global_vertex;
    prev_global_vertex[0] = prev_vertex_map.ptr(point[1])[point[0]].x;
    prev_global_vertex[1] = prev_vertex_map.ptr(point[1])[point[0]].y;
    prev_global_vertex[2] = prev_vertex_map.ptr(point[1])[point[0]].z;

    const float distance = (prev_global_vertex - vertex_current_global).norm();
    if (distance <= distance_threshold)
    {
        normal_current[1] = normal_map_current.ptr(y)[x].y;
        normal_current[2] = normal_map_current.ptr(y)[x].z;

        Vector3f_da global_normal = rotation * normal;

        prev_global_normal[1] = prev_normal_map.ptr(point[1])[point[0]].y;
        prev_global_normal[2] = prev_normal_map.ptr(point[1])[point[0]].z;

        const float sine = global_normal.cross(prev_global_normal).norm();
        
        if (sine >= angle_threshold)
        {
            *(Vector3f_da*)&row[0] = global_vertex.cross(prev_global_normal);
            *(Vector3f_da*)&row[3] = prev_global_normal;
            row[6] = prev_global_normal.dot(prev_global_vertex - global_vertex);
        }
        else
        {
            row[0] = row[1] = row[2] = row[3] = row[4] = row[5] = row[6] = 0.f;
        }
    }   

    __shared__ double smem[BLOCK_SIZE_X * BLOCK_SIZE_Y];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int shift = 0;
    for (int i = 0; i < 6; ++i) { 
        for (int j = i; j < 7; ++j) { 
            __syncthreads();
            smem[tid] = row[i] * row[j];
            __syncthreads();

            if (tid == 0)
                global_buffer.ptr(shift++)[gridDim.x * blockIdx.y + blockIdx.x] = smem[0];
        }
    }
}
*/

void calculate_Ab(
    const cv::cuda::GpuMat& prev_vertex_map, const cv::cuda::GpuMat& prev_normal_map,
    const cv::cuda::GpuMat& vertex_map, const cv::cuda::GpuMat& normal_map,
    const Matrix3f_da& prev_rotation_inv, const Vector3f_da& prev_translation,
    const Matrix3f_da& rotation, const Vector3f_da& translation,
    const CameraParameters& cam,
    float distance_threshold, float angle_threshold,
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& A, Eigen::Matrix<double, 6, 1>& b
)
{
    return;
    const dim3 threads(32, 32);
    const dim3 blocks(
        cv::cudev::divUp(vertex_map.cols, threads.x),
        cv::cudev::divUp(vertex_map.rows, threads.y)
    );

    cv::cuda::GpuMat sum_buffer {cv::cuda::createContinuous(27, 1, CV_64FC1)};
    cv::cuda::GpuMat global_buffer {cv::cuda::createContinuous(27, blocks.x * blocks.y, CV_64FC1)};
    /*
    kernel_pose_estimate<<<blocks, threads>>>(
        prev_vertex_map, prev_normal_map,
        vertex_map, normal_map,
        prev_rotation_inv, prev_translation,
        rotation, translation,
        cam,
        distance_threshold, angle_threshold,
        global_buffer
    );

    kernel_reduction<<<27, 512>>>(global_buffer, blocks.x * blocks * y, sum_buffer);
    
    cv::Mat host_data{ 27, 1, CV_64FC1 };
    sum_buffer.download(host_data);

    int shift = 0;
    for (int i = 0; i < 6; ++i)
    {
        for (int j = i; j < 7; ++j)
        {
            double value = host_data.ptr<double>(shift++)[0];
            if (j == 6)
            {
                b.prev_data()[i] = value;
            }
            else
            {
                A.prev_data()[j * 6 + i] = A.prev_data()[i * 6 + j] = value;
            }
        }
    }
    */
}