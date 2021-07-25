#pragma once

#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

using cv::cuda::GpuMat;


struct Configuration
{
    Configuration(){}
    Configuration(const int num_layers, const int kernel_size, const float sigma_color, const float sigma_spatial, const float max_depth) :
        num_layers(num_layers), kernel_size(kernel_size), sigma_color(sigma_color), sigma_spatial(sigma_spatial), max_depth(max_depth) {}
    // sub-sampling: Number of pyramid layers for each data frame
    int num_layers {3};
    // bilateral filtering
    int kernel_size {5};        // values are for debugging need to be changed later
    float sigma_color {1.f};
    float sigma_spatial {1.f};
    // Cut off depth values that are further away (set depth to 0)
    float max_depth {1000.f};
};


struct CameraParameters
{
    int width, height;
    float fx, fy, cx, cy;  // focal lengths and center point
    float min_depth, max_depth;  // minimum and maximum depth of the depth sensor

    // Constructor
    CameraParameters() {}
    CameraParameters(
        const int &width, const int &height, 
        const float& fx, const float& fy, const float& cx, const float& cy,
        const float min_depth = 150.f, const float max_depth = 6000.f
    ) : width(width), height(height), fx(fx), fy(fy), cx(cx), cy(cy), min_depth(min_depth), max_depth(max_depth) {}

    // get camera parameters at certain pyramid level
    CameraParameters getCameraParameters(const int layer) const 
    {
        if (layer == 0) return *this;

        const int scale_factor = 1 << layer;
        return CameraParameters(
            width >> layer, height >> layer, 
            fx / scale_factor, fy / scale_factor, cx / scale_factor, cy / scale_factor,
            min_depth, max_depth
        );
    }
};


struct PreprocessedData
{
    GpuMat color_map;
    std::vector<GpuMat> depth_pyramid;
    std::vector<GpuMat> vertex_pyramid;
    std::vector<GpuMat> normal_pyramid;


    PreprocessedData(const int& num_layers, CameraParameters &cam) :
        depth_pyramid(num_layers), vertex_pyramid(num_layers), normal_pyramid(num_layers)
    {
        color_map = cv::cuda::createContinuous(cam.height, cam.width, CV_8UC3);
        for (int i = 0; i < num_layers; i++) 
        {
            auto scaled_cam = cam.getCameraParameters(i);
            const int width = scaled_cam.width;
            const int height = scaled_cam.height;
            depth_pyramid[i] = cv::cuda::createContinuous(height, width, CV_32FC1);
            vertex_pyramid[i] = cv::cuda::createContinuous(height, width, CV_32FC3);
            normal_pyramid[i] = cv::cuda::createContinuous(height, width, CV_32FC3);
        }
    }
};


struct ModelData 
{
    std::vector<GpuMat> vertex_pyramid;
    std::vector<GpuMat> normal_pyramid;

    ModelData(const size_t num_levels, const CameraParameters cam) :
            vertex_pyramid(num_levels), normal_pyramid(num_levels)
    {
        for (size_t level = 0; level < num_levels; ++level)
        {
            auto scaled_cam = cam.getCameraParameters(level);
            vertex_pyramid[level] = cv::cuda::createContinuous(scaled_cam.height, scaled_cam.width, CV_32FC3);
            normal_pyramid[level] = cv::cuda::createContinuous(scaled_cam.height, scaled_cam.width, CV_32FC3);
            vertex_pyramid[level].setTo(0);
            normal_pyramid[level].setTo(0);
        }
    }

    // No copying
    ModelData(const ModelData&) = delete;
    ModelData& operator=(const ModelData& data) = delete;

    ModelData(ModelData&& data) noexcept :
            vertex_pyramid(std::move(data.vertex_pyramid)),
            normal_pyramid(std::move(data.normal_pyramid))
    { }

    ModelData& operator=(ModelData&& data) noexcept
    {
        vertex_pyramid = std::move(data.vertex_pyramid);
        normal_pyramid = std::move(data.normal_pyramid);
        return *this;
    }
};


struct TSDFData
{
    // GpuMat only supports 2D
    // (volume_size * volume_size, volume_size) short2
    GpuMat tsdf;  // (F, W)

    int3 volume_size;
    float voxel_scale;

    TSDFData() {}
    
    TSDFData(const int3 _volume_size, const float _voxel_scale) :
        tsdf(cv::cuda::createContinuous(_volume_size.y * _volume_size.z, _volume_size.x, CV_16SC2)),
        volume_size(_volume_size), voxel_scale(_voxel_scale)
    {
        tsdf.setTo(0);
    }
};


struct PointCloud
{
    cv::Mat vertices, normals;
    int num_points;
};


struct CloudData
{
    GpuMat vertices, normals;

    int* point_num;
    int host_point_num;

    explicit CloudData(const int max_number) :
        vertices{cv::cuda::createContinuous(1, max_number, CV_32FC3)},
        normals{cv::cuda::createContinuous(1, max_number, CV_32FC3)},
        point_num{nullptr}, host_point_num{}
    {
        vertices.setTo(0.f);
        normals.setTo(0.f);

        cudaMalloc(&point_num, sizeof(int));
        cudaMemset(point_num, 0, sizeof(int));
    }

    // No copying
    CloudData(const CloudData&) = delete;
    CloudData& operator=(const CloudData& data) = delete;

    PointCloud download()
    {
        cv::Mat host_vertices, host_normals;
        vertices.download(host_vertices);
        normals.download(host_normals);

        cudaMemcpy(&host_point_num, point_num, sizeof(int), cudaMemcpyDeviceToHost);

        PointCloud pc;
        pc.vertices = host_vertices;
        pc.normals = host_normals;
        pc.num_points = host_point_num;
        return pc;
    }
};


struct SurfaceMesh
{
    cv::Mat triangles;
    int num_vertices;
    int num_triangles;
};

struct MeshData {
GpuMat occupied_voxel_ids_buffer;
GpuMat number_vertices_buffer;
GpuMat vertex_offsets_buffer;
GpuMat triangle_buffer;

GpuMat occupied_voxel_ids;
GpuMat number_vertices;
GpuMat vertex_offsets;

explicit MeshData(const int buffer_size):
        occupied_voxel_ids_buffer{cv::cuda::createContinuous(1, buffer_size, CV_32SC1)},
        number_vertices_buffer{cv::cuda::createContinuous(1, buffer_size, CV_32SC1)},
        vertex_offsets_buffer{cv::cuda::createContinuous(1, buffer_size, CV_32SC1)},
        triangle_buffer{cv::cuda::createContinuous(1, buffer_size * 3, CV_32FC3)},
        occupied_voxel_ids{}, number_vertices{}, vertex_offsets{}
{ }

void create_view(const int length)
{
    occupied_voxel_ids = GpuMat(1, length, CV_32SC1, occupied_voxel_ids_buffer.ptr<int>(0),
                                occupied_voxel_ids_buffer.step);
    number_vertices = GpuMat(1, length, CV_32SC1, number_vertices_buffer.ptr<int>(0),
                                number_vertices_buffer.step);
    vertex_offsets = GpuMat(1, length, CV_32SC1, vertex_offsets_buffer.ptr<int>(0),
                            vertex_offsets_buffer.step);
}
};