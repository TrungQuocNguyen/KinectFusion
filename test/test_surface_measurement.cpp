#include <iostream>
#include <string>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

#include "dataset.hpp"
#include "surface_measurement.hpp"


int main(int argc, char **argv)
{
    // The first argument should be the path to the dataset directory
    // The second argument should be whether the dataset is freiburg 1, 2, or 3.
    TUMRGBDDataset tum_dataset("../data/TUMRGBD/rgbd_dataset_freiburg1_360/", TUMRGBDDataset::TUMRGBD::FREIBURG1);

    cv::Mat img, depth_0, depth_1, depth_2, depth_filtered_0, depth_filtered_1, depth_filtered_2;
    size_t num_levels = 3;
    size_t kernel_size = 5;        // values are for debugging
    size_t sigma_color = 1.f;
    size_t sigma_spatial = 1.f;

    size_t index = 4;   // index for choosing frame in dataset
    tum_dataset.getData(index, img, depth_0);
    
    depth_0 *= 128.f; // this is only for debug depth image (0.1f looks good for imshow; 128.f looks good for imsave)
    
    // Compute and show smoothed depth image
    PreprocessedData data = surface_measurement(depth_0, num_levels, kernel_size, sigma_color, sigma_spatial);
    
    // safe images for comparison
    data.depth_pyramid[0].download(depth_0);
    data.depth_pyramid[1].download(depth_1);
    data.depth_pyramid[2].download(depth_2);
    data.filtered_depth_pyramid[0].download(depth_filtered_0);
    data.filtered_depth_pyramid[1].download(depth_filtered_1);
    data.filtered_depth_pyramid[2].download(depth_filtered_2);

    cv::imwrite("rgb.png",img);
    cv::imwrite("depth0.png",depth_0);
    cv::imwrite("depth1.png",depth_1);
    cv::imwrite("depth2.png",depth_2);
    cv::imwrite("depth_filtered0.png",depth_filtered_0);
    cv::imwrite("depth_filtered1.png",depth_filtered_1);
    cv::imwrite("depth_filtered2.png",depth_filtered_2);
}
