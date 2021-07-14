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

__global__
void kernel_estimate(const Matrix3f_da rotation,
    const Vector3f_da translation,
    const PtrStep<float3> vertex_map, const PtrStep<float3> normal_map,
    const Matrix3f_da prev_rotation,
    const Vector3f_da prev_translation,
    const CameraIntrinsics cam_params,
    const PtrStep<float3> prev_vertex_map, const PtrStep<float3> prev_normal_map,
    const float distance_threshold, const float angle_threshold, const int cols,
    const int rows,
    PtrStep<double> global_buffer)
{
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;

    float row[7];

    if (col < cols && row < rows) {
        Vector3f_da normal;
        normal.x() = normal_map.ptr(col)[row].x;

        if (!isnan(normal.x())) {
            Vector3f_da vertex;
            vertex.x() = vertex_map.ptr(col)[row].x;
            vertex.y() = vertex_map.ptr(col)[row].y;
            vertex.z() = vertex_map.ptr(col)[row].z;

            Vector3f_da global_vertex = rotation * vertex + prev_translation;
            Vector3f_da vertex_camera = (global_vertex - prev_translation) * prev_rotation;
            Vector2i_da point;
            point.x() = __float2int_rd(vertex_camera.x() * cam_params.fx / vertex_camera.z() + cam_params.cx + 0.5f);
            point.x() = __float2int_rd(vertex_camera.y() * cam_params.fy / vertex_camera.z() + cam_params.cx + 0.5f);

            if (point.x() >= 0 && point.y() >= 0 && point.x() < cols && point.y() < rows &&
                vertex_camera.z() >= 0) {
                Vector3f_da prev_global_normal;
                prev_global_normal.x() = prev_normal_map.ptr(point.y())[point.x()].x;

                if (!isnan(normal_previous_global.x())) {
                    Vector3f_da prev_global_vertex;
                    prev_global_vertex.x() = prev_vertex_map.ptr(point.y())[point.x()].x;
                    prev_global_vertex.y() = prev_vertex_map.ptr(point.y())[point.x()].y;
                    prev_global_vertex.z() = prev_vertex_map.ptr(point.y())[point.x()].z;

                    const float distance = (vertex_previous_global - vertex_current_global).norm();
                    if (distance <= distance_threshold) {
                        normal_current.y() = normal_map_current.ptr(y)[x].y;
                        normal_current.z() = normal_map_current.ptr(y)[x].z;

                        Vector3f_da global_normal = rotation * normal;

                        prev_global_normal.y() = prev_normal_map.ptr(point.y())[point.x()].y;
                        prev_global_normal.z() = prev_normal_map.ptr(point.y())[point.x()].z;

                        const float sine = global_normal.cross(prev_global_normal).norm();
                       

                        if (sine >= angle_threshold) {
                            *(Vector3f_da*)&row[0] = global_vertex.cross(prev_global_normal);
                            *(Vector3f_da*)&row[3] = prev_global_normal;
                            row[6] = prev_global_normal.dot(prev_global_vertex - global_vertex);
                        }
                        else {
                            row[0] = row[1] = row[2] = row[3] = row[4] = row[5] = row[6] = 0.f;
                        }
                    }
                }
            }

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


void estimation(const Matrix3f_da& rotation, const Vector3f_da& translation,
    const cv::cuda::GpuMat& vertex_map, const cv::cuda::GpuMat& normal_map,
    const Matrix3f_da& prev_rotation_inv, const Vector3f_da& prev_translation,
    const CameraIntrinsics& cam_params,
    const cv::cuda::GpuMat& prev_vertex_map, const cv::cuda::GpuMat& prev_normal_map,
    float distance_threshold, float angle_threshold,
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& A, Eigen::Matrix<double, 6, 1>& b)
{
    const int cols = vertex_map.cols;
    const int rows = vertex_map.rows;

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid(1, 1);
    grid.x = static_cast<unsigned int>(std::ceil(cols / block.x));
    grid.y = static_cast<unsigned int>(std::ceil(rows / block.y));

    cv::cuda::GpuMat sum_buffer{ cv::cuda::createContinuous(27, 1, CV_64FC1) };
    cv::cuda::GpuMat global_buffer{ cv::cuda::createContinuous(27, grid.x * grid.y, CV_64FC1) };

    kernel_estimate << <grid, block >> > (rotation, translation,
        vertex_map, normal_map,
        prev_rotation_inv, prev_translation,
        cam_params,
        prev_vertex_map, prev_normal_map,
        distance_threshold, angle_threshold,
        cols, rows,
        global_buffer);


    cv::Mat host_data{ 27, 1, CV_64FC1 };
    sum_buffer.download(host_data);

    int shift = 0;
    for (int i = 0; i < 6; ++i) { 
        for (int j = i; j < 7; ++j) { 
            double value = host_data.ptr<double>(shift++)[0];
            if (j == 6)
                b.prev_data()[i] = value;
            else
                A.prev_data()[j * 6 + i] = A.prev_data()[i * 6 + j] = value;
        }
    }
}



