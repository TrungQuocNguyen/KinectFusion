#include "dataset.hpp"
#include "datatypes.hpp"
#include "surface_measurement.hpp"
#include "cuda/surface_reconstruct.cuh"

int main()
{
    TUMRGBDDataset dataset("../data/TUMRGBD/rgbd_dataset_freiburg1_360/", TUMRGBDDataset::TUMRGBD::FREIBURG1);
    CameraIntrinsics cam_intrinsics = dataset.getCameraIntrinsics();
    size_t num_levels = 3;
    size_t kernel_size = 5;        // values are for debugging
    size_t sigma_color = 1.f;
    size_t sigma_spatial = 1.f;

    for (int index = 0; index < dataset.size(); ++index)
    {
        cv::Mat img, depth;
        dataset.getData(index, img, depth);
        Eigen::Matrix4f T_c_w = dataset.getPose(index);  // world -> cam

        PreprocessedData data = surface_measurement(depth, num_levels, kernel_size, sigma_color, sigma_spatial);


    }
}