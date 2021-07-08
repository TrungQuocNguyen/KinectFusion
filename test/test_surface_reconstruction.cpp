#include "dataset.hpp"
#include "datatypes.hpp"
#include "surface_measurement.hpp"

void surface_reconstruction(
    const cv::cuda::GpuMat& depth, 
    const CameraIntrinsics& cam_params,
    const Eigen::Matrix4f& T_c_w,
    const float truncation_distance,
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
    TSDFData tsdf_data;

    for (int index = 0; index < 4 /*dataset.size()*/; ++index)
    {
        cv::Mat img, depth;
        dataset.getData(index, img, depth);
        Eigen::Matrix4f T_c_w = dataset.getPose(index);  // world -> cam

        PreprocessedData data(num_levels);
        surface_measurement(data, depth, img, num_levels, kernel_size, sigma_color, sigma_spatial, cam_intrinsics, 8.f);
        surface_reconstruction(data.depth_pyramid[0], cam_intrinsics, T_c_w, truncation_distance, tsdf_data);
    }
}