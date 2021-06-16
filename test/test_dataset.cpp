#include <iostream>
#include <opencv2/opencv.hpp>

#include "dataset.hpp"


int main(int argc, char **argv)
{
    // The first argument should be the path to the dataset directory
    // The second argument should be whether the dataset is freiburg 1, 2, or 3.
    TUMRGBDDataset tum_dataset("data/TUMRGBD/rgbd_dataset_freiburg1_360/", TUMRGBDDataset::TUMRGBD::FREIBURG1);
    
    for (int index=0; index < tum_dataset.size(); ++index)
    {
        // You can get RGB image and depth image by getData()
        // The depth is read by cv::IMREAD_UNCHANGED
        cv::Mat img, depth;
        tum_dataset.getData(index, img, depth);

        // Show rgb image and depth image
        cv::imshow("rgb", img);
        depth *= 0.1f; // this is only for debug depth image
        cv::imshow("depth", depth);
        const int key = cv::waitKey(10);
        if (key == 'q')
        {
            break;
        }
    }
}