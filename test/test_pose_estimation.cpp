#include "dataset.hpp"
#include "surface_measurement.hpp"
#include "datatypes.hpp"


int main()
{
    if (Config::read("../data/tumrgbd.yaml") == false) return -1;

    std::string dataset_dir = Config::get<std::string>("dataset_dir");
    Dataset dataset = TUMRGBDDataset(dataset_dir, static_cast<TUMRGBDDataset::TUMRGBD>(Config::get<int>("tumrgbd")));

    auto cam = dataset.getCameraParameters();
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
    const float truncation_distance {Config::get<float>("truncation_distance")};
    for (int index = 0; index < dataset.size(); ++index)
    {
        cv::Mat img, depth;
        dataset.getData(index, img, depth);
        depth *= 1000.f;  // m -> mm
        
        FrameData data(num_levels);
        surfaceMeasurement(data, depth, img, num_levels, kernel_size, sigma_color, sigma_spatial, cam, 4000.f);

        bool icp_success {true};
        if (index > 0)
        {
            icp_success = poseEstimation(
                current_pose, frame_data, model_data, cam,
                configuration.num_levels,
                configuration.distance_threshold, configuration.angle_threshold,
                configuration.icp_iterations
            );
        }

        surfaceReconstruction(data.depth_pyramid[0], cam, current_pose, truncation_distance, tsdf_data);

        for (int level = 0; level < num_levels; ++level)
        {
            surfacePrediction(
                tsdf_data, cam.getCameraParameters(level), current_pose,
                truncation_distance,
                model_data.vertex_pyramid[level], model_data.normal_pyramid[level]
            );
        }

        std::cout << "frame : " << index << std::endl;

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
