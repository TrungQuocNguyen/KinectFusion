#include <cuda/kernel_common.cuh>
#include <datatypes.hpp>

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

template<int SIZE>
static __device__ __forceinline__ void reduce(volatile double* buffer)
{
    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    double value = buffer[thread_id];

    if (SIZE >= 1024) 
    {
        if (thread_id < 512) buffer[thread_id] = value = value + buffer[thread_id + 512];
        __syncthreads();
    }
    if (SIZE >= 512) 
    {
        if (thread_id < 256) buffer[thread_id] = value = value + buffer[thread_id + 256];
        __syncthreads();
    }
    if (SIZE >= 256) 
    {
        if (thread_id < 128) buffer[thread_id] = value = value + buffer[thread_id + 128];
        __syncthreads();
    }
    if (SIZE >= 128)
    {
        if (thread_id < 64) buffer[thread_id] = value = value + buffer[thread_id + 64];
        __syncthreads();
    }

    if (thread_id < 32)
    {
        if (SIZE >= 64) buffer[thread_id] = value = value + buffer[thread_id + 32];
        if (SIZE >= 32) buffer[thread_id] = value = value + buffer[thread_id + 16];
        if (SIZE >= 16) buffer[thread_id] = value = value + buffer[thread_id + 8];
        if (SIZE >= 8) buffer[thread_id] = value = value + buffer[thread_id + 4];
        if (SIZE >= 4) buffer[thread_id] = value = value + buffer[thread_id + 2];
        if (SIZE >= 2) buffer[thread_id] = value = value + buffer[thread_id + 1];
    }
}


__global__ void kernel_pose_estimate(
    const PtrStep<float3> prev_vertex_map, const PtrStep<float3> prev_normal_map,
    const PtrStepSz<float3> curr_vertex_map, const PtrStep<float3> curr_normal_map,
    const Matrix3f_da prev_rotation, const Vector3f_da prev_translation,
    const Matrix3f_da curr_rotation, const Vector3f_da curr_translation,
    const CameraParameters cam,
    const float distance_threshold, const float angle_threshold,
    PtrStep<double> global_buffer
)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    float temp[7] {};
    if (x <= curr_vertex_map.cols - 1 && y <= curr_vertex_map.rows - 1)
    {
        const float3* curr_vertex = &curr_vertex_map.ptr(y)[x];
        if (curr_vertex->z > 1e-5)
        {
            const Vector3f_da curr_vertex_g = curr_rotation * Vector3f_da(curr_vertex->x, curr_vertex->y, curr_vertex->z) + curr_translation;
            const Vector3f_da curr_vertex_c = prev_rotation.transpose() * (curr_vertex_g - prev_translation);

            const float3* curr_normal = &curr_normal_map.ptr(y)[x];
            if (abs(curr_normal->x) + abs(curr_normal->y) + abs(curr_normal->z) > 1e-5)
            {
                const Vector3f_da curr_normal_g = curr_rotation * Vector3f_da(curr_normal->x, curr_normal->y, curr_normal->z);
                const Vector2i_da projected_pt(
                    __float2int_rd(curr_vertex_c[0] * cam.fx / curr_vertex_c[2] + cam.cx + 0.5f),
                    __float2int_rd(curr_vertex_c[1] * cam.fy / curr_vertex_c[2] + cam.cy + 0.5f)
                );

                if (projected_pt[0] >= 0 && projected_pt[0] <= curr_vertex_map.cols - 1 && projected_pt[1] >= 0 && projected_pt[1] <= curr_vertex_map.rows - 1)
                {
                    const float3* prev_vertex = &prev_vertex_map.ptr(projected_pt[1])[projected_pt[0]];
                    if (prev_vertex->z > 1e-5)
                    {
                        const Vector3f_da prev_vertex_g(prev_vertex->x, prev_vertex->y, prev_vertex->z);
                    
                        if ((curr_vertex_g - prev_vertex_g).norm() <= distance_threshold)
                        {
                            const float3* prev_normal = &prev_normal_map.ptr(projected_pt[1])[projected_pt[0]];
                            if (abs(prev_normal->x) + abs(prev_normal->y) + abs(prev_normal->z) > 1e-5)
                            {
                                const Vector3f_da prev_normal_g {prev_normal->x, prev_normal->y, prev_normal->z};
                                if (abs(curr_normal_g.dot(prev_normal_g)) >= angle_threshold)
                                {
                                    *(Vector3f_da*) &temp[0] = curr_vertex_g.cross(prev_normal_g);
                                    *(Vector3f_da*) &temp[3] = prev_normal_g;
                                    temp[6] = prev_normal_g.dot(prev_vertex_g - curr_vertex_g);
                                }
                            }                            
                        }
                    }
                }
            }            
        }
    }    
    
    __shared__ double shared_memory[BLOCK_SIZE_X * BLOCK_SIZE_Y];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int shift = 0;
    for (int i = 0; i < 6; ++i) 
    {
        for (int j = i; j < 7; ++j)
        {
            __syncthreads();
            shared_memory[tid] = temp[i] * temp[j];
            __syncthreads();

            reduce<BLOCK_SIZE_X * BLOCK_SIZE_Y>(shared_memory);

            if (tid == 0)
            {
                global_buffer.ptr(shift++)[gridDim.x * blockIdx.y + blockIdx.x] = shared_memory[0];
            }                
        }
    }
}


__global__ void estimate_kernel(
    const PtrStep<float3> prev_vertex_map, const PtrStep<float3> prev_normal_map,
    const PtrStepSz<float3> vertex_map, const PtrStep<float3> normal_map,
    const Matrix3f_da prev_rotation, const Vector3f_da prev_translation,
    const Matrix3f_da rotation, const Vector3f_da translation,
    const CameraParameters cam,
    const float distance_threshold, const float angle_threshold,
    PtrStep<double> global_buffer
)    
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    float temp[7];
    temp[0] = temp[1] = temp[2] = temp[3] = temp[4] = temp[5] = temp[6] = 0.f;
    Vector3f_da prev_normal, prev_vertex, curr_vertex;

    if (x < vertex_map.cols && y < vertex_map.rows)
    {
        Vector3f_da curr_normal;
        curr_normal.x() = normal_map.ptr(y)[x].x;

        if (!isnan(curr_normal.x())) 
        {
            Vector3f_da curr_vertex(
                vertex_map.ptr(y)[x].x,
                vertex_map.ptr(y)[x].y,
                vertex_map.ptr(y)[x].z
            );

            Vector3f_da curr_vertex_g = rotation * curr_vertex + translation;
            Vector3f_da curr_vertex_c = prev_rotation.transpose() * (curr_vertex_g - prev_translation);

            Eigen::Vector2i point(
                __float2int_rd(curr_vertex_c.x() * cam.fx / curr_vertex_c.z() + cam.cx + 0.5f),
                __float2int_rd(curr_vertex_c.y() * cam.fy / curr_vertex_c.z() + cam.cy + 0.5f)
            );

            if (point.x() >= 0 && point.y() >= 0 && point.x() < vertex_map.cols && point.y() < vertex_map.rows && curr_vertex_c.z() >= 0)
            {
                Vector3f_da prev_normal_g;
                prev_normal_g.x() = prev_normal_map.ptr(point.y())[point.x()].x;

                if (!isnan(prev_normal_g.x()))
                {
                    Vector3f_da prev_vertex_g(
                        prev_vertex_map.ptr(point.y())[point.x()].x,
                        prev_vertex_map.ptr(point.y())[point.x()].y,
                        prev_vertex_map.ptr(point.y())[point.x()].z
                    );

                    const float distance = (prev_vertex_g - curr_vertex_g).norm();
                    if (distance <= distance_threshold)
                    {
                        curr_normal.y() = normal_map.ptr(y)[x].y;
                        curr_normal.z() = normal_map.ptr(y)[x].z;
                        Vector3f_da curr_normal_g = rotation * curr_normal;

                        prev_normal_g.y() = prev_normal_map.ptr(point.y())[point.x()].y;
                        prev_normal_g.z() = prev_normal_map.ptr(point.y())[point.x()].z;

                        if (curr_normal_g.dot(prev_normal_g) >= angle_threshold)
                        {
                            prev_normal = prev_normal_g;
                            prev_vertex = prev_vertex_g;
                            curr_vertex = curr_vertex_g;
                            *(Vector3f_da*) &temp[0] = curr_vertex.cross(prev_normal);
                            *(Vector3f_da*) &temp[3] = prev_normal;
                            temp[6] = prev_normal.dot(prev_vertex - curr_vertex);
                        }
                    }
                }
            }
        }
    }   

    __shared__ double shared_memory[BLOCK_SIZE_X * BLOCK_SIZE_Y];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int shift = 0;
    for (int i = 0; i < 6; ++i)
    {
        for (int j = i; j < 7; ++j)
        {
            __syncthreads();
            shared_memory[tid] = temp[i] * temp[j];
            __syncthreads();
            reduce<BLOCK_SIZE_X * BLOCK_SIZE_Y>(shared_memory);

            if (tid == 0) global_buffer.ptr(shift++)[gridDim.x * blockIdx.y + blockIdx.x] = shared_memory[0];
        }
    }
}


__global__ void reduction_kernel(
    PtrStep<double> global_buffer, const int length, PtrStep<double> output
)
{
    double sum = 0.0;
    for (int t = threadIdx.x; t < length; t += 512)
    {
        sum += *(global_buffer.ptr(blockIdx.x) + t);
    }        

    __shared__ double shared_memory[512];
    shared_memory[threadIdx.x] = sum;
    __syncthreads();

    reduce<512>(shared_memory);

    if (threadIdx.x == 0) output.ptr(blockIdx.x)[0] = shared_memory[0];
};


void calculateAb(
    const GpuMat& prev_vertex_map, const GpuMat& prev_normal_map,
    const GpuMat& vertex_map, const GpuMat& normal_map,
    const Matrix3f_da& prev_rotation, const Vector3f_da& prev_translation,
    const Matrix3f_da& rotation, const Vector3f_da& translation,
    const CameraParameters& cam,
    const float& distance_threshold, const float& angle_threshold,
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& AtA, Eigen::Matrix<double, 6, 1>& Atb
)
{
    const dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    const dim3 blocks(
        static_cast<unsigned int>(std::ceil(vertex_map.cols / threads.x)),
        static_cast<unsigned int>(std::ceil(vertex_map.rows / threads.y))
    );

    GpuMat sum_buffer { cv::cuda::createContinuous(27, 1, CV_64FC1) };
    GpuMat global_buffer { cv::cuda::createContinuous(27, blocks.x * blocks.y, CV_64FC1) };
    kernel_pose_estimate<<<blocks, threads>>>(
        prev_vertex_map, prev_normal_map,
        vertex_map, normal_map,
        prev_rotation, prev_translation,
        rotation, translation,
        cam,
        distance_threshold, angle_threshold,
        global_buffer
    );
    
    reduction_kernel<<<27, 512>>>(global_buffer, blocks.x * blocks.y, sum_buffer);

    cv::Mat host_data { 27, 1, CV_64FC1 };
    sum_buffer.download(host_data);

    int shift = 0;
    for (int i = 0; i < 6; ++i)
    {
        for (int j = i; j < 7; ++j)
        {
            double value = host_data.ptr<double>(shift++)[0];
            if (j == 6)
            {
                Atb.data()[i] = value;
            }
            else
            {
                AtA.data()[j * 6 + i] = AtA.data()[i * 6 + j] = value;
            }
        }
    }
}
