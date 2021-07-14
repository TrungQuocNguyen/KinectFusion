#include "config.hpp"
#include "dataset.hpp"
#include "datatypes.hpp"
#include "surface_measurement.hpp"

void surface_reconstruction(
    const cv::cuda::GpuMat& depth, 
    const CameraParameters& cam,
    const Eigen::Matrix4f T_c_w,
    const float truncation_distance,
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

int main()
{
    if (Config::setParameterFile("../data/kinfu_tumrgbd.yaml") == false) return -1;

    std::string dataset_dir = config::get<std::string>("dataset_dir");
    Dataset dataset = TUMRGBDDataset(dataset_dir, config::get<TUMRGBDDataset::TUMRGBD>("tumrgbd"));
    auto cam = dataset.getCameraParameters();

    int num_levels = config::get<int>("num_levels");
    int kernel_size = config::get<int>("bf_kernel_size");
    float sigma_color {1.f};
    float sigma_spatial {1.f};
    float truncation_distance {10.f};
    TSDFData tsdf_data(make_int3(1024, 1024, 512), 10.f);
    Eigen::Matrix4f current_pose = Eigen::Matrix4f::Identity();
    for (int index = 0; index < dataset.size(); ++index)
    {
        cv::Mat img, depth;
        dataset.getData(index, img, depth);
        depth *= 1000.f;  // m -> mm
        
        if (index != 0)
        {
            // get ground truth pose
            Eigen::Matrix4f rel_pose = dataset.getPose(index - 1).inverse() * dataset.getPose(index);
            rel_pose.block<3, 1>(0, 3) *= 1000.f;  // m -> mm
            current_pose = current_pose * rel_pose;
        }

        PreprocessedData data(num_levels);
        surface_measurement(data, depth, img, num_levels, kernel_size, sigma_color, sigma_spatial, cam, 4000.f);

        surface_reconstruction(data.depth_pyramid[0], cam, current_pose, truncation_distance, tsdf_data);
    }

    PointCloud pc = extract_points(tsdf_data, 3 * 1000000);
    export_ply(dataset_name + ".ply", pc);
    cv::Vec4f::ones();
}