#include <opencv2/viz/viz3d.hpp>
#include "kinectfusion.hpp"
#include "dataset.hpp"


int main()
{
    if (Config::read("../data/tumrgbd.yaml") == false) return -1;

    std::string dataset_dir = Config::get<std::string>("dataset_dir");
    Dataset dataset = TUMRGBDDataset(dataset_dir, static_cast<TUMRGBDDataset::TUMRGBD>(Config::get<int>("tumrgbd")));

    auto cam = dataset.getCameraParameters();
    cam.max_depth = Config::get<float>("max_depth");
    cam.min_depth = Config::get<float>("min_depth");
    Eigen::Matrix4f current_pose = Eigen::Matrix4f::Identity();

    KinectFusion kinfu(cam);
    kinfu.readConfig();

    bool flag_use_gt_pose = Config::get<int>("flag_use_gt_pose") == 1;
    Eigen::Matrix4f gt_init_pose = dataset.getPose(0);

    // for visualization
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
    
    for (int index = 0; index < dataset.size(); ++index)
    {
        cv::Mat img, depth;
        dataset.getData(index, img, depth);
        depth *= 1000.f;  // m -> mm
        
        if (flag_use_gt_pose)
        {
            Eigen::Matrix4f rel_pose = dataset.getPose(index - 1).inverse() * dataset.getPose(index);
            rel_pose.block<3, 1>(0, 3) *= 1000.f;  // m -> mm
            current_pose = current_pose * rel_pose;
            kinfu.setPose(current_pose);
        }

        kinfu.addFrame(img, depth, dataset.getTimestamp(index));
        
        // Visualization
        // TODO: make another thread
        // TODO: visualize by gpu
        kinfu.showImages(img, depth);
        kinfu.visualize3D(my_window, img, true);

        int k = cv::waitKey(1);
        if (k == 'q') break;  // press q to quit
        else if (k == ' ') cv::waitKey(0);  // press space to stop
    }
    
    int tmp = dataset_dir.rfind("/", dataset_dir.size() - 2);
    std::string dataset_name = dataset_dir.substr(tmp + 1, dataset_dir.size() - tmp - 2);

    // save as point cloud
    kinfu.savePointcloud(dataset_name + ".ply", 3 * 1000000);

    // save poses
    kinfu.savePoses(dataset_name + "_pose.txt", gt_init_pose);
}
