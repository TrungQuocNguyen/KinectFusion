#include <iostream>
#include <opencv2/opencv.hpp>

#include "dataset.hpp"
#include "utils.hpp"



int main()
{
    if (Config::setParameterFile("../data/tumrgbd.yaml") == false) return -1;
    std::string dataset_dir = Config::get<std::string>("dataset_dir");

    // The first argument should be the path to the dataset directory
    // The second argument should be whether the dataset is freiburg 1, 2, or 3.
    TUMRGBDDataset dataset(dataset_dir, static_cast<TUMRGBDDataset::TUMRGBD>(Config::get<int>("tumrgbd")));
    
    for (int index=0; index < dataset.size(); ++index)
    {
        // You can get RGB image and depth image by getData()
        // The depth is read by cv::IMREAD_UNCHANGED
        cv::Mat img, depth;
        dataset.getData(index, img, depth);

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