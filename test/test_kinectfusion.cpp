#include <opencv2/viz/viz3d.hpp>
#include "kinectfusion.hpp"
#include "dataset.hpp"

int main()
{
    if (Config::setParameterFile("../data/tumrgbd.yaml") == false) return -1;

    std::string dataset_dir = Config::get<std::string>("tum_dataset_dir");
    Dataset dataset = TUMRGBDDataset(dataset_dir, static_cast<TUMRGBDDataset::TUMRGBD>(Config::get<int>("tumrgbd")));

    auto cam = dataset.getCameraParameters();
    cam.max_depth = Config::get<float>("max_depth");
    cam.min_depth = Config::get<float>("min_depth");
    Eigen::Matrix4f current_pose =  Eigen::Matrix4f::Identity();

    cv::viz::Viz3d my_window("Surface Prediction");
    my_window.setBackgroundColor(cv::viz::Color::black());
    // Show coordinate system
    my_window.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());
    cv::Matx33f K = cv::Matx33f::eye();
    K(0, 0) = cam.fx;
    K(1, 1) = cam.fy;
    K(0, 2) = cam.cx;
    K(1, 2) = cam.cy;
    cv::viz::WCameraPosition wcam(K, 100.0, cv::viz::Color::red());
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
        icp_iterations[i] = Config::get<int>("icp_iteration_" + std::to_string(i));
    }
    const float truncation_distance {Config::get<float>("truncation_distance")};
    const bool flag_use_gt_pose {Config::get<int>("flag_use_gt_pose") == 1};
    TSDFData tsdf_data(
        make_int3(Config::get<int>("tsdf_size_x"), Config::get<int>("tsdf_size_y"), Config::get<int>("tsdf_size_z")), 
        Config::get<int>("tsdf_scale")
    );
    FrameData data(num_levels, cam);
    ModelData model_data(num_levels, cam);
    double sum_time = 0.;
    Eigen::Matrix4f gt_init_pose = dataset.getPose(0);
    std::vector<Eigen::Matrix4f> estimated_poses;
    for (int index = 0; index < dataset.size(); ++index)
    {
        printf("index : %d\n", index);
        cv::Mat img, depth;
        dataset.getData(index, img, depth);
        depth *= 1000.f;  // m -> mm

        Timer timer("Frame " + std::to_string(index));

        surface_measurement(depth, img, num_levels, kernel_size, sigma_color, sigma_spatial, cam, data);
        timer.print("Surface Measurement");

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
                    model_data, data, cam,
                    num_levels, distance_threshold, angle_threshold,
                    icp_iterations,
                    current_pose
                );
                timer.print("Pose Estimation");

                estimated_poses.push_back(current_pose);
                if (!icp_success) continue;
            }
        }

        surface_reconstruction(data.depth_pyramid[0], cam, current_pose, truncation_distance, tsdf_data);
        timer.print("Surface Reconstruction");

        surface_prediction(tsdf_data, cam, current_pose, truncation_distance, num_levels, model_data);
        timer.print("Surface Prediction");

        sum_time += timer.print();
        printf("[ FPS ] : %f\n", (index + 1) * 1000.f / sum_time);

        cv::Mat measured_normals, normals, vertices;
        data.normal_pyramid[0].download(measured_normals);
        model_data.normal_pyramid[0].download(normals);
        model_data.vertex_pyramid[0].download(vertices);
        cv::imshow("measured normals", measured_normals);
        cv::imshow("predicted normals", normals);
        cv::imshow("img", img);
        int k = cv::waitKey(1);
        if (k == 'q') break;  // press q to quit
        else if (k == ' ') cv::waitKey(0);  // press space to stop

        cv::Matx44f T = cv::Matx44f::eye();
        for (int y = 0; y < 4; ++y)
        {
            for (int x = 0; x < 4; ++x)
            {
                T(y, x) = current_pose.matrix()(y, x);
            }
        }
        cv::viz::WCloud point_cloud(vertices, img);
        my_window.showWidget("points", point_cloud);
        cv::viz::WCloudNormals normal_cloud(vertices, normals, 64, 0.10, cv::viz::Color::red());
        my_window.showWidget("normals", normal_cloud);
        my_window.setWidgetPose("cam", cv::Affine3f(T));
        my_window.spinOnce(10);
    }
    
    int tmp = dataset_dir.rfind("/", dataset_dir.size() - 2);
    std::string dataset_name = dataset_dir.substr(tmp + 1, dataset_dir.size() - tmp - 2);

    // save as point cloud
    PointCloud pc = extract_pointcloud(tsdf_data, 3 * 1000000);
    export_ply(dataset_name + ".ply", pc);

    // save poses
    std::ofstream ofs(dataset_name + "_pose.txt");
    for (int index = 0; index < estimated_poses.size(); ++index)
    {
        Eigen::Matrix4f pose = estimated_poses[index];
        pose.block<3, 1>(0, 3) *= 0.001f;
        pose = pose * gt_init_pose;

        Eigen::Vector3f t = pose.block<3, 1>(0, 3);
        auto q = Eigen::Quaternionf(pose.block<3, 3>(0, 0));
        ofs << dataset.getTimestamp(index) << " " << 
            t.x() << " " << t.y() << " " << t.z() << " " << 
            q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }
    ofs.close();
}
