#include <opencv2/viz/viz3d.hpp>
#include "utils.hpp"
#include "dataset.hpp"
#include "surface_measurement.hpp"
#include "datatypes.hpp"

using namespace cv;
using namespace std;

int main()
{

    if (Config::setParameterFile("../data/tumrgbd.yaml") == false) return -1;

    std::string dataset_dir = Config::get<std::string>("tum_dataset_dir");
    Dataset dataset = TUMRGBDDataset(dataset_dir, static_cast<TUMRGBDDataset::TUMRGBD>(Config::get<int>("tumrgbd")));

    auto cam = dataset.getCameraParameters();
    cam.max_depth = Config::get<float>("max_depth");
    cam.min_depth = Config::get<float>("min_depth");
    
    const int num_levels {Config::get<int>("num_levels")};

    const int kernel_size {Config::get<int>("bf_kernel_size")};
    const float sigma_color {Config::get<float>("bf_sigma_color")};
    const float sigma_spatial {Config::get<float>("bf_sigma_spatial")};

    PreprocessedData data(num_levels, cam);
    
    // Create a window
    cv::viz::Viz3d myWindow("Viz Demo");
    myWindow.setBackgroundColor(cv::viz::Color::black());
    // Show coordinate system
    myWindow.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());


    for (int index = 0; index < dataset.size(); ++index)
    {
        printf("index : %d\n", index);
        cv::Mat img, depth;
        dataset.getData(index, img, depth);
        depth *= 1000.f;  // m -> mm

        Timer timer("Frame " + std::to_string(index));

        surface_measurement(depth, img, num_levels, kernel_size, sigma_color, sigma_spatial, cam, data);
        timer.print("Surface Measurement");

        cv::Mat vertex, normal, flt_depth;
        data.vertex_pyramid[0].download(vertex);
        data.normal_pyramid[0].download(normal);
        data.depth_pyramid[0].download(flt_depth);

        depth.convertTo(depth, CV_16U, 5.f);
        flt_depth.convertTo(flt_depth, CV_16U, 5.f);

        cv::imshow("normal", normal);
        cv::imshow("measured depth", depth);
        cv::imshow("filtered depth", flt_depth);
        int key = cv::waitKey(0);
        if (key == ' ') break;
    }
    /*
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
    */
}
