#pragma once
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include "utils.hpp"
#include "datatypes.hpp"
#include "surface_measurement.hpp"
#include "surface_prediction.hpp"
#include "pose_estimation.hpp"



void surface_reconstruction(
    const cv::cuda::GpuMat& depth, 
    const CameraParameters& cam,
    const Eigen::Matrix4f& T_c_w,
    const float& truncation_distance,
    TSDFData& volume
);


PointCloud extract_points(const TSDFData& volume, const int buffer_size);


void export_ply(const std::string& filename, const PointCloud& point_cloud)
{
    std::ofstream file_out { filename };
    if (!file_out.is_open()) return;

    file_out << "ply" << std::endl;
    file_out << "format ascii 1.0" << std::endl;
    file_out << "element vertex " << point_cloud.num_points << std::endl;
    file_out << "property float x" << std::endl;
    file_out << "property float y" << std::endl;
    file_out << "property float z" << std::endl;
    file_out << "property float nx" << std::endl;
    file_out << "property float ny" << std::endl;
    file_out << "property float nz" << std::endl;
    file_out << "end_header" << std::endl;

    for (int i = 0; i < point_cloud.num_points; ++i) {
        float3 vertex = point_cloud.vertices.ptr<float3>(0)[i];
        float3 normal = point_cloud.normals.ptr<float3>(0)[i];
        file_out << vertex.x << " " << vertex.y << " " << vertex.z << " " << normal.x << " " << normal.y << " "
                    << normal.z << std::endl;
    }
}


SurfaceMesh marching_cubes(const TSDFData& volume, const int buffer_size);


SurfaceMesh extract_mesh(const TSDFData& volume, const int buffer_size)
{
    SurfaceMesh surface_mesh = marching_cubes(volume, buffer_size);
    return surface_mesh;
}


void export_ply(const std::string& filename, const SurfaceMesh& surface_mesh)
{
    std::ofstream file_out { filename };
    if (!file_out.is_open())
        return;

    file_out << "ply" << std::endl;
    file_out << "format ascii 1.0" << std::endl;
    file_out << "element vertex " << surface_mesh.num_vertices << std::endl;
    file_out << "property float x" << std::endl;
    file_out << "property float y" << std::endl;
    file_out << "property float z" << std::endl;
    file_out << "element face " << surface_mesh.num_triangles << std::endl;
    file_out << "property list uchar int vertex_index" << std::endl;
    file_out << "end_header" << std::endl;

    for (int v_idx = 0; v_idx < surface_mesh.num_vertices; ++v_idx) {
        float3 vertex = surface_mesh.triangles.ptr<float3>(0)[v_idx];
        file_out << vertex.x << " " << vertex.y << " " << vertex.z  << std::endl;
    }

    for (int t_idx = 0; t_idx < surface_mesh.num_vertices; t_idx += 3) {
        file_out << 3 << " " << t_idx + 1 << " " << t_idx << " " << t_idx + 2 << std::endl;
    }
}