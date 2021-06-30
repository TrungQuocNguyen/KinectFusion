#include <iostream>
#include <string>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

#include "dataset.hpp"
#include "surface_measurement.hpp"
#include "datatypes.hpp"

#include <opencv2/surface_matching/ppf_helpers.hpp>

using Vec3fda = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;

int main(int argc, char **argv)
{
    // The first argument should be the path to the dataset directory
    // The second argument should be whether the dataset is freiburg 1, 2, or 3.
    TUMRGBDDataset tum_dataset("../data/TUMRGBD/rgbd_dataset_freiburg1_360/", TUMRGBDDataset::TUMRGBD::FREIBURG1);
    Configuration config;

    cv::Mat img, depth;
    
    // camera paramerters
    CameraIntrinsics camera_params;
    tum_dataset.getCameraParameters(camera_params.fx, camera_params.fy, camera_params.cx, camera_params.cy);

    size_t index = 230;   // index for choosing frame in dataset
    tum_dataset.getData(index, img, depth);
    
    depth *= 128.f; // this is only for debug depth image (0.1f for imshow; 128.f for imwrite)
    
    camera_params.img_height = depth.rows;
    camera_params.img_width = depth.cols;

    // declare data pyramids
    assert (config.num_layers > 0);
    PreprocessedData data(config.num_layers);
    
    // Allocate GPU memory
    data.color_map = cv::cuda::createContinuous(camera_params.img_height, camera_params.img_width, CV_8UC3);
    data.color_map.upload(img);
    for (int i = 0; i < config.num_layers; i++) {
        const int width = camera_params.getCameraIntrinsics(i).img_width;
        const int height = camera_params.getCameraIntrinsics(i).img_height;
        data.depth_pyramid[i] = cv::cuda::createContinuous(height, width, CV_32FC1);
        data.filtered_depth_pyramid[i] = cv::cuda::createContinuous(height, width, CV_32FC1);
        data.vertex_pyramid[i] = cv::cuda::createContinuous(height, width, CV_32FC3);
        data.normal_pyramid[i] = cv::cuda::createContinuous(height, width, CV_32FC3);
    }
    
    data.depth_pyramid[0].upload(depth);

    // Compute surface measurement
    surface_measurement(data, config.num_layers, config.kernel_size, config.sigma_color, config.sigma_spatial, camera_params, config.max_depth);
    
    // safe images for comparison
    cv::Mat vertex;
    //std::cout << "Size ( data.depth_pyramid[0]): " << data.vertex_pyramid[0].size() << std::endl;
    data.depth_pyramid[0].download(depth);
    data.vertex_pyramid[0].download(vertex);
    cv::imwrite("depth.png", depth);
    cv::imwrite("vertex.png", vertex);
    //std::cout << "Size (vertex): " << vertex.size() << std::endl;
    
    //cv::ppf_match_3d::writePLY(vertex,"vertex.ply");	
}
