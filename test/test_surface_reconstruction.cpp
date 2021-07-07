#include "dataset.hpp"
#include "datatypes.hpp"
#include "surface_measurement.hpp"

void surface_reconstruction(
    const cv::cuda::GpuMat& depth, 
    const CameraIntrinsics& cam_params, const float truncation_distance,
    const Eigen::Matrix4f& T_c_w,
    VolumeData& volume
);

int main()
{
    TUMRGBDDataset dataset("../data/TUMRGBD/rgbd_dataset_freiburg1_360/", TUMRGBDDataset::TUMRGBD::FREIBURG1);
    CameraIntrinsics cam_intrinsics = dataset.getCameraIntrinsics();
    size_t num_levels {3};
    size_t kernel_size {5};
    float sigma_color {1.f};
    float sigma_spatial {1.f};
    float truncation_distance {0.25f};
    VolumeData volume;

    for (int index = 0; index < 1 /*dataset.size()*/; ++index)
    {
        cv::Mat img, depth;
        dataset.getData(index, img, depth);
        Eigen::Matrix4f T_c_w = dataset.getPose(index);  // world -> cam

        PreprocessedData data(num_levels);
        data.depth_pyramid[0].upload(depth);
        surface_measurement(data, num_levels, kernel_size, sigma_color, sigma_spatial, cam_intrinsics, 8.f);
        surface_reconstruction(data.depth_pyramid[0], cam_intrinsics, truncation_distance, T_c_w, volume);
    }
}