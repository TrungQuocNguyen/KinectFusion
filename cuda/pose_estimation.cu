#include "cuda_runtime.h"
#include <opencv2/core/cuda.hpp>
#include <datatypes.hpp>
#include <Eigen/Core>
#include <opencv2/cudaarithm.hpp>

using cv::cuda::GpuMat;

__global__ void kernel_pose_estimate(
            const cv::cuda::PtrStepSz<float3> vertex_map,
            const cv::cuda::PtrStepSz<float3> normal_map, 
            const cv::cuda::PtrStepSz<float3> pred_vertex_map, 
            const cv::cuda::PtrStepSz<float3> pred_normal_map,
            const cv::cuda::PtrStepSz<int> valid_vertex_mask,
            cv::cuda::PtrStepSz<float> left_buffer, 
            cv::cuda::PtrStepSz<float> right_buffer,
            const Eigen::Matrix4f T_g_k,
            //const CameraIntrinsics camera_params, 
            const float threshold_dist, 
            const float threshold_angle)
{
    // Calculate global row and column for each thread
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= vertex_map.cols || row >= vertex_map.rows){
        return;
    }

    Eigen::Matrix<float, 6, 1> right;
    Eigen::Matrix<float, 6, 6> left;

    // reject grossly incorrect correspondances
    if (valid_vertex_mask(row, col) == 0)
    {
        right.setZero();
        left.setZero();
        return;
    }
    
    Eigen::Vector3f vertex;
    vertex.x() = vertex_map(row, col).x;
    vertex.y() = vertex_map(row, col).y;
    vertex.z() = vertex_map(row, col).z;

    Eigen::Matrix3f rotation = T_g_k.block<3,3>(0,0);
    Eigen::Vector3f translation = T_g_k.block<3,1>(0,3);
    
    Eigen::Vector3f vertex_current = (rotation * vertex) + translation;

    Eigen::Vector3f vertex_pred;
    vertex_pred.x() = pred_vertex_map(row, col).x;
    vertex_pred.y() = pred_vertex_map(row, col).y;
    vertex_pred.z() = pred_vertex_map(row, col).z;
    
    float vertex_dist = (vertex_current - vertex_pred).norm();
    if (vertex_dist > threshold_dist)
    {
        right.setZero();
        left.setZero();
        return;
    }

    Eigen::Vector3f normal;
    normal.x() = normal_map(row, col).x;
    normal.y() = normal_map(row, col).y;
    normal.z() = normal_map(row, col).z;
    Eigen::Vector3f normal_current = rotation * normal;
    
    Eigen::Vector3f normal_pred;
    normal_pred.x() = pred_normal_map(row, col).x;
    normal_pred.y() = pred_normal_map(row, col).y;
    normal_pred.z() = pred_normal_map(row, col).z;

    float angle_diff = normal_current.dot(normal_pred);
    if (angle_diff > threshold_angle)
    {
        right.setZero();
        left.setZero();
        return;
    }
    
    Eigen::Matrix<float, 3, 6> G;
    G <<  0.f,                -vertex_current[2], vertex_current[1],  1.f, 0.f, 0.f,
        vertex_current[2],  0.f,                -vertex_current[0], 0.f, 1.f, 0.f,
        -vertex_current[1], vertex_current[0],  0.f,                0.f, 0.f, 1.f;

    Eigen::Matrix<float, 6, 1> A_T = G.transpose() * normal_pred;
    float b = normal_pred.dot(vertex_pred - vertex_current);
    
    right << A_T(0, 0) * b, A_T(1, 0) * b, A_T(2, 0) * b, A_T(3, 0) * b, A_T(4, 0) * b, A_T(5, 0) * b;
    left << A_T * A_T.transpose();
    
    for (int i = 0; i < 6; i++)
    {
        right_buffer(row+i, col) = right(i, 0); //A_T(i, 0) * b;
    }

    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            left_buffer(row+i, col+j) = left(i, j);
        }   
    }
}

void compute_pose_estimate(
                const GpuMat& vertex_map, 
                const GpuMat& normal_map, 
                const GpuMat& pred_vertex_map, 
                const GpuMat& pred_normal_map, 
                const GpuMat& valid_vertex_mask, 
                const Eigen::Matrix4f &T_g_k, 
                Eigen::Matrix<float, 6, 6, Eigen::RowMajor>& left, 
                Eigen::Matrix<float, 6, 1>& right, 
                const float& threshold_dist, 
                const float& threshold_angle){

    int threads = 32;
    dim3 T(threads, threads, 1);      // number of threads per block (depends on compute capability of your GPU)
    int blocks_x = (vertex_map.cols + T.x - 1) / T.x;
    int blocks_y = (vertex_map.rows + T.y - 1) / T.y;
    dim3 M(blocks_x, blocks_y, 1);       // number of thread blocks (depends on compute capability of your GPU)

    cv::cuda::GpuMat sum_left = cv::cuda::createContinuous(36, 1, CV_32FC1);
    cv::cuda::GpuMat sum_right = cv::cuda::createContinuous(6, 1, CV_32FC1);

    cv::cuda::GpuMat left_buffer = cv::cuda::createContinuous(36, M.x * M.y, CV_32FC1);
    cv::cuda::GpuMat right_buffer = cv::cuda::createContinuous(6, M.x * M.y, CV_32FC1);

    kernel_pose_estimate<<< M , T >>>(vertex_map, normal_map, pred_vertex_map, pred_normal_map, valid_vertex_mask, left_buffer, right_buffer, T_g_k, threshold_dist, threshold_angle);
    cudaDeviceSynchronize();

    cv::cuda::reduce(left_buffer, sum_left, 1, cv::REDUCE_SUM);
    cv::cuda::reduce(right_buffer, sum_right, 1, cv::REDUCE_SUM);

    cv::Mat left_d;
    cv::Mat right_d;
    sum_left.download(left_d);
    sum_right.download(right_d);

    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            left.data()[i*6+j] = left_d.ptr(i*6+j)[0];
        }
    }
    for (int i = 0; i < 6; i++)
    {
        right.data()[i] = right_d.ptr(i)[0];
    }
    
}