#include <opencv2/viz/viz3d.hpp>
#include "utils.hpp"
#include "dataset.hpp"
#include "datatypes.hpp"
#include "surface_measurement.hpp"
#include "surface_prediction.hpp"


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
    Eigen::Matrix4f current_pose = Eigen::Matrix4f::Identity();

    // visualization to check normals and vertices
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

    int num_levels {Config::get<int>("num_levels")};
    int kernel_size {Config::get<int>("bf_kernel_size")};
    float sigma_color {Config::get<float>("bf_sigma_color")};
    float sigma_spatial {Config::get<float>("bf_sigma_spatial")};
    float truncation_distance {Config::get<float>("truncation_distance")};
    TSDFData tsdf_data(make_int3(Config::get<int>("tsdf_size_x"), Config::get<int>("tsdf_size_y"), Config::get<int>("tsdf_size_z")), Config::get<int>("tsdf_scale"));
    ModelData model_data(num_levels, cam);
    PreprocessedData data(num_levels, cam);
    double sum_t = 0.;
    for (int index = 0; index < dataset.size(); ++index)
    {
        cv::Mat img, depth;
        dataset.getData(index, img, depth);
        depth *= 1000.f;  // m -> mm
        
        Timer timer("Frame " + std::to_string(index));

        if (index != 0)
        {
            // get ground truth pose
            Eigen::Matrix4f rel_pose = dataset.getPose(index - 1).inverse() * dataset.getPose(index);
            rel_pose.block<3, 1>(0, 3) *= 1000.f;  // m -> mm
            current_pose = current_pose * rel_pose;
        }

        surface_measurement(depth, img, num_levels, kernel_size, sigma_color, sigma_spatial, cam, data);
        timer.print("Surface Measurement");
        
        surface_reconstruction(data.depth_pyramid[0], cam, current_pose, truncation_distance, tsdf_data);
        timer.print("Surface Reconstruction");

        surface_prediction(
            tsdf_data, cam, current_pose,
            truncation_distance, num_levels,
            model_data
        );
        timer.print("Surface Prediction");

        sum_t += timer.print();
        std::cout << "[ FPS ] : " << (index + 1) * 1000.f / sum_t << std::endl;

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