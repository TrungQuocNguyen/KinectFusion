#include <opencv2/viz/viz3d.hpp>
#include "config.hpp"
#include "dataset.hpp"
#include "datatypes.hpp"
#include "surface_measurement.hpp"
#include "pose_estimation.hpp"


void surface_prediction(
    const TSDFData &volume,
    const CameraParameters &cam,
    const Eigen::Matrix4f T_c_w,
    const float trancation_distance,
    GpuMat &vertex_map, GpuMat &normal_map
);


void surface_reconstruction(
    const cv::cuda::GpuMat& depth, 
    const CameraParameters& cam,
    const Eigen::Matrix4f T_c_w,
    const float truncation_distance,
    TSDFData& volume
);


int main()
{
    if (Config::setParameterFile("../data/kinfu_tumrgbd.yaml") == false) return -1;

    std::string dataset_dir = Config::get<std::string>("dataset_dir");
    Dataset dataset = TUMRGBDDataset(dataset_dir, static_cast<TUMRGBDDataset::TUMRGBD>(Config::get<int>("tumrgbd")));

    auto cam = dataset.getCameraParameters();
    cam.max_depth = Config::get<float>("max_depth");
    cam.min_depth = Config::get<float>("min_depth");
    Eigen::Matrix4f current_pose = Eigen::Matrix4f::Identity();

    cv::viz::Viz3d my_window("Surface Prediction");
    my_window.setBackgroundColor(cv::viz::Color::black());
    // Show coordinate system
    my_window.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());
    cv::Matx33f K = cv::Matx33f::eye();
    K(0, 0) = cam.fx;
    K(1, 1) = cam.fy;
    K(0, 2) = cam.cx;
    K(1, 2) = cam.cy;
    cv::viz::WCameraPosition wcam(K, 1.0, cv::viz::Color::red());
    my_window.showWidget("cam", wcam);

    const int num_levels {Config::get<int>("num_levels")};

    const int kernel_size {Config::get<int>("bf_kernel_size")};
    const float sigma_color {Config::get<float>("bf_sigma_color")};
    const float sigma_spatial {Config::get<float>("bf_sigma_spatial")};
    const float distance_threshold {Config::get<float>("icp_distance_threshold")};
    const float angle_threshold {Config::get<float>("icp_angle_threshold")};
    std::vector<int> icp_iterations(num_levels);
    for (int i = 0; i < num_levels; ++i)
    {
        icp_iterations.push_back(Config::get<int>("icp_iteration_" + std::to_string(i)));
    }
    const float truncation_distance {Config::get<float>("truncation_distance")};
    const bool flag_use_gt_pose {Config::get<int>("flag_use_gt_pose") == 1};
    TSDFData tsdf_data(
        make_int3(Config::get<int>("tsdf_size_x"), Config::get<int>("tsdf_size_y"), Config::get<int>("tsdf_size_z")), 
        Config::get<int>("tsdf_scale")
    );
    PreprocessedData data(num_levels);
    ModelData model_data(num_levels, cam);
    for (int index = 0; index < dataset.size(); ++index)
    {
        cv::Mat img, depth;
        dataset.getData(index, img, depth);
        depth *= 1000.f;  // m -> mm

        surface_measurement(depth, img, num_levels, kernel_size, sigma_color, sigma_spatial, cam, data);

        if (index != 0)
        {
            if (flag_use_gt_pose)
            {
                Eigen::Matrix4f rel_pose = dataset.getPose(index - 1).inverse() * dataset.getPose(index);
                rel_pose.block<3, 1>(0, 3) *= 1000.f;  // m -> mm
                current_pose = current_pose * rel_pose;
            }
            else
            {
                bool icp_success = pose_estimation(
                    current_pose, data, model_data, cam,
                    num_levels, distance_threshold, angle_threshold,
                    icp_iterations
                );
            }            
        }

        surface_reconstruction(data.depth_pyramid[0], cam, current_pose, truncation_distance, tsdf_data);

        for (int level = 0; level < num_levels; ++level)
        {
            surface_prediction(
                tsdf_data, cam.getCameraParameters(level), current_pose,
                truncation_distance,
                model_data.vertex_pyramid[level], model_data.normal_pyramid[level]
            );
        }

        cv::Mat normals, vertices;
        model_data.normal_pyramid[0].download(normals);
        model_data.vertex_pyramid[0].download(vertices);
        cv::imshow("n", normals);
        cv::imshow("img", img);
        int k = cv::waitKey(1);
        if (k == 'q') break;  // press q to quit
        else if (k == ' ') cv::waitKey(0);  // press space to stop

        cv::Matx44f T = cv::Matx44f::eye();
        for (int y = 0; y < 4; ++y)
        {
            for (int x = 0; x < 4; ++x)
            {
                T(y, x) = current_pose(y, x);
            }
        }
        cv::viz::WCloud point_cloud(vertices, img);
        my_window.showWidget("points", point_cloud);
        cv::viz::WCloudNormals normal_cloud(vertices, normals, 64, 0.10, cv::viz::Color::red());
        my_window.showWidget("normals", normal_cloud);
        my_window.setWidgetPose("cam", cv::Affine3f(T));
        my_window.spinOnce(10);
    }
}
