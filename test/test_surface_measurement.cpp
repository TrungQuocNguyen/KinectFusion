#include <opencv2/viz/viz3d.hpp>
#include "utils.hpp"
#include "dataset.hpp"
#include "surface_measurement.hpp"
#include "datatypes.hpp"

using namespace cv;
using namespace std;

int main()
{
    // The first argument should be the path to the dataset directory
    // The second argument should be whether the dataset is freiburg 1, 2, or 3.
    TUMRGBDDataset dataset("../data/TUMRGBD/rgbd_dataset_freiburg1_360/", TUMRGBDDataset::TUMRGBD::FREIBURG1);
    auto cam = dataset.getCameraParameters();
    Configuration config;

    // Create a window
    cv::viz::Viz3d myWindow("Viz Demo");
    myWindow.setBackgroundColor(cv::viz::Color::black());
    // Show coordinate system
    myWindow.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());


    cv::Mat img, depth;

    size_t index = 10;   // index for choosing frame in dataset
    dataset.getData(index, img, depth);
    
    // declare data pyramids
    PreprocessedData data(config.num_layers, cam);

    // Compute surface measurement
    surface_measurement(depth, img, config.num_layers, config.kernel_size, config.sigma_color, config.sigma_spatial, cam, data);
    
    cv::Mat vertex, normal;
    data.vertex_pyramid[0].download(vertex);
    data.normal_pyramid[0].download(normal);
    data.depth_pyramid[0].download(depth);

    // safe images for comparison
    depth.convertTo(depth, CV_16U, 5000.f);
    cv::imwrite("rgb.png", img);
    cv::imwrite("depth.png", depth);
    cv::imwrite("vertex.png", vertex);

    // Show point cloud
    //cv::viz::WCloud pointCloud(normal, cv::viz::Color::green());
    cv::viz::WCloud pointCloud(vertex, cv::viz::Color::green());
    myWindow.showWidget("vertex", pointCloud);

    // Start event loop (run until user terminates it by pressing e, E, q, Q)
    myWindow.spin();
}
