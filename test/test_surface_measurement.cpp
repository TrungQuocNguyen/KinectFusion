#include <opencv2/viz/viz3d.hpp>
#include "utils.hpp"
#include "dataset.hpp"
#include "surface_measurement.hpp"
#include "datatypes.hpp"

using namespace cv;
using namespace std;

int main()
{

    if (Config::read("../data/tumrgbd.yaml") == false) return -1;

    std::string dataset_dir = Config::get<std::string>("dataset_dir");
    Dataset dataset = TUMRGBDDataset(dataset_dir, static_cast<TUMRGBDDataset::TUMRGBD>(Config::get<int>("tumrgbd")));

    auto cam = dataset.getCameraParameters();
    cam.max_depth = Config::get<float>("max_depth");
    cam.min_depth = Config::get<float>("min_depth");
    
    const int num_levels {Config::get<int>("num_levels")};

    const int kernel_size {Config::get<int>("bf_kernel_size")};
    const float sigma_color {Config::get<float>("bf_sigma_color")};
    const float sigma_spatial {Config::get<float>("bf_sigma_spatial")};

    FrameData data(num_levels, cam);
    
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

        surfaceMeasurement(depth, img, num_levels, kernel_size, sigma_color, sigma_spatial, cam, data);
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
        int key = cv::waitKey(1);
        if (key == 'q') break;
    }
}
