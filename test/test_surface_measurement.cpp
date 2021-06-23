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

    for (int index=0; index < tum_dataset.size(); ++index)
    {
        //std::cout << std::to_string(tum_dataset.size());
        // You can get RGB image and depth image by getData()
        // The depth is read by cv::IMREAD_UNCHANGED
        cv::Mat img, depth, depth_filtered;
        
        tum_dataset.getData(index, img, depth);
        
        // Show rgb image and depth image
        cv::imshow("rgb", img);
        depth *= 128.f; // this is only for debug depth image (0.1f looks good for imshow; 128.f looks good for imsave)
        cv::imshow("depth", depth);
        
        // Compute and show smoothed depth image
        PreprocessedData data = surface_measurement(depth);
        data.filtered_depth_pyramid.download(depth_filtered);
        cv::imshow("depth_filtered", depth_filtered);
        
        // safe images for comparison
        if (index == 4){
            cv::imwrite("rgb.png",img);
            cv::imwrite("depth.png",depth);
            cv::imwrite("depth_filtered.png",depth_filtered);
        }

        const int key = cv::waitKey(10);
        if (key == 'q')
        {
            break;
        }
    }
}
