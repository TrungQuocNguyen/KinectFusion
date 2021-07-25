#include <iostream>
#include <string>


#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/viz/viz3d.hpp>

#include "dataset.hpp"
#include "surface_measurement.hpp"
#include "datatypes.hpp"

#include <opencv2/surface_matching/ppf_helpers.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    // The first argument should be the path to the dataset directory
    // The second argument should be whether the dataset is freiburg 1, 2, or 3.
    TUMRGBDDataset tum_dataset("../data/TUMRGBD/rgbd_dataset_freiburg1_360/", TUMRGBDDataset::TUMRGBD::FREIBURG1);
    Configuration config;

    cv::Mat img, depth;
    
    CameraIntrinsics camera_params = tum_dataset.getCameraIntrinsics();

    size_t index = 10;   // index for choosing frame in dataset
    tum_dataset.getData(index, img, depth);
    
    depth *= 128.f; // this is only for debug depth image (0.1f for imshow; 128.f for imwrite)
    
    camera_params.img_height = depth.rows;
    camera_params.img_width = depth.cols;
    
    // declare data pyramids
    assert (config.num_layers > 0);
    PreprocessedData data(config.num_layers);

    // Compute surface measurement
    surface_measurement(data, depth, img, config.num_layers, config.kernel_size, config.sigma_color, config.sigma_spatial, camera_params, config.max_depth);
    
    cv::Mat vertex, normal;
    data.vertex_pyramid[0].download(vertex);
    data.normal_pyramid[0].download(normal);
    data.depth_pyramid[0].download(depth);
    std::cout << "depth: " << depth << std::endl;

    // safe images for comparison
    cv::imwrite("rgb.png", img);
    cv::imwrite("depth.png", depth);
    cv::imwrite("vertex.png", vertex);

    // Create a window
    cv::viz::Viz3d myWindow("Viz Demo");
    myWindow.setBackgroundColor(cv::viz::Color::black());

    // Show coordinate system
    myWindow.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());

    // Show point cloud
    //cv::viz::WCloud pointCloud(normal, cv::viz::Color::green());
    cv::viz::WCloud pointCloud(vertex/128.f, cv::viz::Color::green());
    myWindow.showWidget("points", pointCloud);

    // Start event loop (run until user terminates it by pressing e, E, q, Q)
    myWindow.spin();
    std::cout << "First event loop is over" << std::endl;
}
