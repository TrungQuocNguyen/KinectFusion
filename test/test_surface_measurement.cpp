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

int main()
{
    Dataset dataset;
    dataset = TUMRGBDDataset("../data/TUMRGBD/rgbd_dataset_freiburg1_desk/", TUMRGBDDataset::TUMRGBD::FREIBURG1);

    CameraIntrinsics cam = dataset.getCameraIntrinsics();
    
    // visualization to check normals and vertices
    cv::viz::Viz3d my_window0("Surface Measurement0");
    cv::viz::Viz3d my_window1("Surface Measurement1");
    cv::viz::Viz3d my_window2("Surface Measurement2");
    my_window0.setBackgroundColor(cv::viz::Color::black());
    my_window1.setBackgroundColor(cv::viz::Color::black());
    my_window2.setBackgroundColor(cv::viz::Color::black());
    // Show coordinate system
    my_window0.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());
    my_window1.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());
    my_window2.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());
    
    int num_levels {3};
    int kernel_size {9};
    float sigma_color {1.f};
    float sigma_spatial {1.f};
    for (int index = 0; index < dataset.size(); ++index)
    {
        cv::Mat img0, img1, img2, depth;
        dataset.getData(index, img0, depth);
        depth *= 1000.f;  // m -> mm

        PreprocessedData data(num_levels);
        surface_measurement(data, depth, img0, num_levels, kernel_size, sigma_color, sigma_spatial, cam, 100000.f);

        std::cout << "frame : " << index << std::endl;

        cv::Mat normals0, vertices0;
        cv::Mat normals1, vertices1;
        cv::Mat normals2, vertices2;
        data.normal_pyramid[0].download(normals0);
        data.vertex_pyramid[0].download(vertices0);
        data.normal_pyramid[1].download(normals1);
        data.vertex_pyramid[1].download(vertices1);
        data.normal_pyramid[2].download(normals2);
        data.vertex_pyramid[2].download(vertices2);
        data.color_pyramid[1].download(img1);
        data.color_pyramid[2].download(img2);
        cv::imshow("n0", normals0);
        cv::imshow("img0", img0);
        cv::imshow("n1", normals1);
        cv::imshow("img1", img1);
        cv::imshow("n2", normals2);
        cv::imshow("img2", img2);
        int k = cv::waitKey(1);
        if (k == 'q') break;  // press q to quit
        else if (k == ' ') cv::waitKey(0);  // press space to stop

        cv::viz::WCloud point_cloud0(vertices0, img0);
        my_window0.showWidget("points0", point_cloud0);
        cv::viz::WCloudNormals normal_cloud0(vertices0, normals0, 64, 0.10, cv::viz::Color::red());
        my_window0.showWidget("normals0", normal_cloud0);
        my_window0.spinOnce(10);
        
        cv::viz::WCloud point_cloud1(vertices1, img1);
        my_window1.showWidget("points1", point_cloud1);
        cv::viz::WCloudNormals normal_cloud1(vertices1, normals1, 64, 0.10, cv::viz::Color::red());
        my_window1.showWidget("normals1", normal_cloud1);
        my_window1.spinOnce(10);

        cv::viz::WCloud point_cloud2(vertices2, img2);
        my_window2.showWidget("points2", point_cloud2);
        cv::viz::WCloudNormals normal_cloud2(vertices2, normals2, 64, 0.10, cv::viz::Color::red());
        my_window2.showWidget("normals2", normal_cloud2);
        my_window2.spinOnce(10);
    }
}
