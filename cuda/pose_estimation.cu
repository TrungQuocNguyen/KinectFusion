#include <opencv2/core/cuda.hpp>
#include <datatypes.hpp>
#include <Eigen/Core>
#include <opencv2/cudaarithm.hpp>

using cv::cuda::GpuMat;
using Vector3f_da = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using Matrix3f_da = Eigen::Matrix<float, 3, 3, Eigen::DontAlign>;

__global__ void kernel_pose_estimate(
            const cv::cuda::PtrStepSz<float3> vertex_map,
            const cv::cuda::PtrStepSz<float3> normal_map, 
            const cv::cuda::PtrStepSz<float3> pred_vertex_map, 
            const cv::cuda::PtrStepSz<float3> pred_normal_map,
            const cv::cuda::PtrStepSz<int> valid_vertex_mask,
            cv::cuda::PtrStepSz<float> left_buffer, 
            cv::cuda::PtrStepSz<float> right_buffer,
            const int rows, const int cols,
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
    
    // reject grossly incorrect correspondances
    if (valid_vertex_mask(row, col) == 0)
    {
        return;
    }
    

    Vector3f_da vertex;
    vertex.x() = vertex_map(row, col).x;
    vertex.y() = vertex_map(row, col).y;
    vertex.z() = vertex_map(row, col).z;
    
    Matrix3f_da rotation = T_g_k.block<3,3>(0,0);
    Vector3f_da translation = T_g_k.block<3,1>(0,3);
    
    Vector3f_da vertex_current;
    vertex_current.x() = rotation(0,0)*vertex.x() + rotation(0,1)*vertex.y() + rotation(0,2)*vertex.z() + translation(0,0);
    vertex_current.y() = rotation(1,0)*vertex.x() + rotation(1,1)*vertex.y() + rotation(1,2)*vertex.z() + translation(1,0);
    vertex_current.z() = rotation(2,0)*vertex.x() + rotation(2,1)*vertex.y() + rotation(2,2)*vertex.z() + translation(2,0);
    //printf("Here: %f\n", vertex_current.x());
    
    if(isnan(vertex_current.x()) || isnan(vertex_current.y()) || isnan(vertex_current.z())){
        return;
    }

    Vector3f_da vertex_pred;
    vertex_pred.x() = pred_vertex_map(row, col).x;
    vertex_pred.y() = pred_vertex_map(row, col).y;
    vertex_pred.z() = pred_vertex_map(row, col).z;
    
    if(isnan(vertex_pred.x()) || isnan(vertex_pred.y()) || isnan(vertex_pred.z())){
        return;
    }
    float vertex_dist = (vertex_current - vertex_pred).norm();
    //printf("vertex_dist: %f\n", vertex_dist);
    if (vertex_dist > threshold_dist)
    {
        return;
    }

    Vector3f_da normal;
    normal.x() = normal_map(row, col).x;
    normal.y() = normal_map(row, col).y;
    normal.z() = normal_map(row, col).z;
    //Vector3f_da normal_current = rotation * normal;
    Vector3f_da normal_current;
    normal_current.x() = rotation(0,0)*normal.x() + rotation(0,1)*normal.y() + rotation(0,2)*normal.z();
    normal_current.y() = rotation(1,0)*normal.x() + rotation(1,1)*normal.y() + rotation(1,2)*normal.z();
    normal_current.z() = rotation(2,0)*normal.x() + rotation(2,1)*normal.y() + rotation(2,2)*normal.z();
    

    if(isnan(normal_current.x()) || isnan(normal_current.y()) || isnan(normal_current.z())){
        return;
    }
    
    Vector3f_da normal_pred;
    normal_pred.x() = pred_normal_map(row, col).x;
    normal_pred.y() = pred_normal_map(row, col).y;
    normal_pred.z() = pred_normal_map(row, col).z;

    if(isnan(normal_pred.x()) || isnan(normal_pred.y()) || isnan(normal_pred.z())){
        return;
    }

    //float angle_diff = normal_current.dot(normal_pred);
    /*Vector3f_da temp;
    temp.x() = normal_current[1]*normal_pred[2] + normal_current[2]*normal_pred[1];
    temp.y() = normal_current[2]*normal_pred[0] + normal_current[0]*normal_pred[2];
    temp.z() = normal_current[0]*normal_pred[1] + normal_current[1]*normal_pred[0];*/
    //const float angle_diff = normal_current.cross(normal_pred).norm();
    /*if (isnan(temp.norm()))
    {
        printf("normal_current0: %f\n", normal_current.x());
        printf("normal_current1: %f\n", normal_current.y());
        printf("normal_current2: %f\n", normal_current.z());
        printf("normal_pred0: %f\n", normal_pred[0]);
        printf("normal_pred1: %f\n", normal_pred[1]);
        printf("normal_pred2: %f\n", normal_pred[2]);
        printf("angle_diff: %f\n", temp.norm());

    }*/
    
    /*if (temp.norm() > sinf(threshold_angle * 3.14159254f / 180.f))
    {
        return;
    }*/
    
    //printf("normal_pred: %f\n", normal_pred(0,0));
    //printf("normal_pred: %f\n", normal_pred(1,0));
    //printf("normal_pred: %f\n", normal_pred(2,0));
    
    Eigen::Matrix<float, 6, 1, Eigen::DontAlign> A_T;
    A_T <<  vertex_current[1] * normal_pred[2] - vertex_current[2] * normal_pred[1],
            vertex_current[2] * normal_pred[0] - vertex_current[0] * normal_pred[2],
            vertex_current[0] * normal_pred[1] - vertex_current[1] * normal_pred[0],
            normal_pred[0], normal_pred[1], normal_pred[2];
    
    float b = normal_pred.dot(vertex_pred - vertex_current);
    /*printf("A_T0: %f\n", A_T(0,0));
    printf("A_T1: %f\n", A_T(1,0));
    printf("A_T2: %f\n", A_T(2,0));
    printf("A_T3: %f\n", A_T(3,0));
    printf("A_T4: %f\n", A_T(4,0));
    printf("A_T5: %f\n", A_T(5,0));*/
    //printf("b: %f\n", b);
    
    Eigen::Matrix<float, 6, 1> right;
    Eigen::Matrix<float, 6, 6> left;
    
    //int r = 250;
    //int c = 300;
    //printf("A_T: %f, b: %f\n", A_T(0, 0), b);
    //printf("A_T * b: %f\n", A_T(0, 0) * b);
    /*right(0,0) = A_T(0, 0) * b;
    right(1,0) = A_T(1, 0) * b;
    right(2,0) = A_T(2, 0) * b;
    right(3,0) = A_T(3, 0) * b;
    right(4,0) = A_T(4, 0) * b;
    right(5,0) = A_T(5, 0) * b;*/
    /*if (row == r && col == c)
    {
        printf("A_T: %f, b: %f\n", A_T(0, 0), b);
        printf("A_T0: %f\n", A_T(0,0));
        printf("A_T1: %f\n", A_T(1,0));
        printf("A_T2: %f\n", A_T(2,0));
        printf("A_T3: %f\n", A_T(3,0));
        printf("A_T4: %f\n", A_T(4,0));
        printf("A_T5: %f\n", A_T(5,0));
        printf("A_T * b: %f\n", A_T(0, 0) * b);
        printf("A_T * b: %f\n", A_T(0, 0) * A_T(0,0));
        printf("right1: %f\n", right(0,0));
        printf("right2: %f\n", right(1,0));
        printf("right3: %f\n", right(2,0));
        printf("right4: %f\n", right(3,0));
        printf("right5: %f\n", right(4,0));
        printf("right6: %f\n", right(5,0));
    }*/
    /*printf("right1: %f\n", right(0,0));
    printf("right2: %f\n", right(1,0));
    printf("right3: %f\n", right(2,0));
    printf("right4: %f\n", right(3,0));
    printf("right5: %f\n", right(4,0));
    printf("right6: %f\n", right(5,0));*/
    //const Eigen::Matrix<float, 1, 6> A = A_T.transpose();
    /*left << A_T(0,0) * A_T(0,0), A_T(0,0) * A_T(1,0), A_T(0,0) * A_T(2,0), A_T(0,0) * A_T(3,0), A_T(0,0) * A_T(4,0), A_T(0,0) * A_T(5,0),
            A_T(1,0) * A_T(0,0), A_T(1,0) * A_T(1,0), A_T(1,0) * A_T(2,0), A_T(1,0) * A_T(3,0), A_T(1,0) * A_T(4,0), A_T(1,0) * A_T(5,0),
            A_T(2,0) * A_T(0,0), A_T(2,0) * A_T(1,0), A_T(2,0) * A_T(2,0), A_T(2,0) * A_T(3,0), A_T(2,0) * A_T(4,0), A_T(2,0) * A_T(5,0),
            A_T(3,0) * A_T(0,0), A_T(3,0) * A_T(1,0), A_T(3,0) * A_T(2,0), A_T(3,0) * A_T(3,0), A_T(3,0) * A_T(4,0), A_T(3,0) * A_T(5,0),
            A_T(4,0) * A_T(0,0), A_T(4,0) * A_T(1,0), A_T(4,0) * A_T(2,0), A_T(4,0) * A_T(3,0), A_T(4,0) * A_T(4,0), A_T(4,0) * A_T(5,0),
            A_T(5,0) * A_T(0,0), A_T(5,0) * A_T(1,0), A_T(5,0) * A_T(2,0), A_T(5,0) * A_T(3,0), A_T(5,0) * A_T(4,0), A_T(5,0) * A_T(5,0);
    *//*for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            float temp = A_T(j,0);
            left(i,j) = 1.f * A_T(i,0) * temp;
        } 
    }*/
    //left = A_T * A;

    for (int i = 0; i < 6; i++)
    {
        /*if (row == r && col == c)
        {   
            printf("right(%d,0): %f\n", i, right(i,0));
        }*/
        //right_buffer(i, row*cols + col) = right(i, 0);
        right_buffer(i, row*cols + col) = A_T(i, 0) * b;
    }

    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            /*if (row == r && col == c)
            {   
                printf("left(%d,%d): %f\n", i, j, A_T(i,0)*A_T(j,0));
            }*/
            left_buffer(i*6+j, row*cols + col) = A_T(i,0)*A_T(j,0);
            //left_buffer(i*6+j, row*cols + col) = left(i,j);
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
    
    cv::cuda::GpuMat sum_left {cv::cuda::createContinuous(36, 1, CV_32FC1)};
    cv::cuda::GpuMat sum_right {cv::cuda::createContinuous(6, 1, CV_32FC1)};
    
    cv::cuda::GpuMat left_buffer {cv::cuda::createContinuous(36, vertex_map.cols * vertex_map.rows, CV_32FC1)};
    cv::cuda::GpuMat right_buffer {cv::cuda::createContinuous(6, vertex_map.cols * vertex_map.rows, CV_32FC1)};
    
    left_buffer.setTo(0.f);
    right_buffer.setTo(0.f);

    kernel_pose_estimate<<< M , T >>>(
        vertex_map, normal_map, 
        pred_vertex_map, pred_normal_map, 
        valid_vertex_mask, 
        left_buffer, right_buffer,
        vertex_map.rows, vertex_map.cols,
        T_g_k, 
        threshold_dist, threshold_angle);
    
    cudaDeviceSynchronize();
    
    /*cv::Mat temp1, temp2;
    left_buffer.download(temp1);
    right_buffer.download(temp2);
    printf("temp1: %f\n", temp1.at<float>(0,0));
    printf("temp1: %f\n", temp1.at<float>(1,0));
    printf("temp1: %f\n", temp1.at<float>(2,2));
    printf("temp1: %f\n", temp1.at<float>(3,6));
    printf("temp1: %f\n", temp1.at<float>(4,7));
    printf("temp1: %f\n", temp1.at<float>(5,3));
    printf("temp2: %f\n", temp2.at<float>(0,0));
    printf("temp2: %f\n", temp2.at<float>(1,0));
    printf("temp2: %f\n", temp2.at<float>(2,2));
    printf("temp2: %f\n", temp2.at<float>(3,6));
    printf("temp2: %f\n", temp2.at<float>(4,7));
    printf("temp2: %f\n", temp2.at<float>(5,3));*/
    
    cv::cuda::reduce(left_buffer, sum_left, 1, cv::REDUCE_SUM);
    cv::cuda::reduce(right_buffer, sum_right, 1, cv::REDUCE_SUM);
    
    cv::Mat left_d;
    cv::Mat right_d;
    sum_left.download(left_d);
    sum_right.download(right_d);

    /*printf("left_d1: %f\n", left_d.at<float>(0,0));
    printf("left_d2: %f\n", left_d.at<float>(1,0));
    printf("left_d3: %f\n", left_d.at<float>(2,0));
    printf("left_d4: %f\n", left_d.at<float>(3,0));
    printf("left_d5: %f\n", left_d.at<float>(4,0));
    printf("left_d6: %f\n", left_d.at<float>(5,0));*/
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            left(i,j) = left_d.at<float>(i*6+j,0);
        }
    }
    for (int i = 0; i < 6; i++)
    {
        right(i,0) = right_d.at<float>(i,0);
    }
}