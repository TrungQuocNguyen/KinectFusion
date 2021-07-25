#include <opencv2/viz/viz3d.hpp>
#include "dataset.hpp"
#include "datatypes.hpp"
#include "surface_measurement.hpp"


void surface_prediction(
    const TSDFData &volume,
    const CameraIntrinsics &cam,
    const Eigen::Matrix4f T_c_w,
    const float trancation_distance,
    GpuMat &vertex_map, GpuMat &normal_map
);


void surface_reconstruction(
    const cv::cuda::GpuMat& depth, 
    const CameraIntrinsics& cam_params,
    const Eigen::Matrix4f T_c_w,
    const float truncation_distance,
    TSDFData& volume
);


struct ModelData {
    std::vector<GpuMat> vertex_pyramid;
    std::vector<GpuMat> normal_pyramid;

    ModelData(const size_t num_levels, const CameraIntrinsics cam) :
            vertex_pyramid(num_levels), normal_pyramid(num_levels)
    {
        for (size_t level = 0; level < num_levels; ++level)
        {
            auto scale_cam = cam.getCameraIntrinsics(level);
            vertex_pyramid[level] = cv::cuda::createContinuous(scale_cam.img_height, scale_cam.img_width, CV_32FC3);
            normal_pyramid[level] = cv::cuda::createContinuous(scale_cam.img_height, scale_cam.img_width, CV_32FC3);
            vertex_pyramid[level].setTo(0);
            normal_pyramid[level].setTo(0);
        }
    }

    // No copying
    ModelData(const ModelData&) = delete;
    ModelData& operator=(const ModelData& data) = delete;

    ModelData(ModelData&& data) noexcept :
            vertex_pyramid(std::move(data.vertex_pyramid)),
            normal_pyramid(std::move(data.normal_pyramid))
    { }

    ModelData& operator=(ModelData&& data) noexcept
    {
        vertex_pyramid = std::move(data.vertex_pyramid);
        normal_pyramid = std::move(data.normal_pyramid);
        return *this;
    }
};


int main()
{
    Dataset dataset;
    dataset = TUMRGBDDataset("../data/TUMRGBD/rgbd_dataset_freiburg1_desk/", TUMRGBDDataset::TUMRGBD::FREIBURG1);

    CameraIntrinsics cam = dataset.getCameraIntrinsics();
    Eigen::Matrix4f current_pose = Eigen::Matrix4f::Identity();

    // visualization to check normals and vertices
    cv::viz::Viz3d my_window("Surface Prediction");
    my_window.setBackgroundColor(cv::viz::Color::black());
    // Show coordinate system
    my_window.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());
    cv::Matx33f K = cv::Matx33f::eye();
    K(0, 0) = cam.fx;
    K(1, 1) = cam.fy;
    K(0, 2) = cam.cx;
    K(1, 2) = cam.cy;
    cv::viz::WCameraPosition wcam(K, 1.0, cv::viz::Color::red());
    my_window.showWidget("cam", wcam);

    int num_levels {3};
    int kernel_size {9};
    float sigma_color {1.f};
    float sigma_spatial {1.f};
    float truncation_distance {100000.f};
    TSDFData tsdf_data(make_int3(1024, 1024, 512), 10.f);
    ModelData model_data(num_levels, cam);
    for (int index = 0; index < dataset.size(); ++index)
    {
        cv::Mat img, depth;
        dataset.getData(index, img, depth);
        depth *= 1000.f;  // m -> mm
        
        if (index != 0)
        {
            // get ground truth pose
            Eigen::Matrix4f rel_pose = dataset.getPose(index - 1).inverse() * dataset.getPose(index);
            rel_pose.block<3, 1>(0, 3) *= 1000.f;  // m -> mm
            current_pose = current_pose * rel_pose;
        }

        PreprocessedData data(num_levels);
        surface_measurement(data, depth, img, num_levels, kernel_size, sigma_color, sigma_spatial, cam, 4000.f);

        surface_reconstruction(data.depth_pyramid[0], cam, current_pose, truncation_distance, tsdf_data);
        cv::Mat tsdf;
        tsdf_data.tsdf.download(tsdf);
        std::cout << "tsdf_data : " << tsdf << std::endl;
        for (int level = 0; level < num_levels; ++level)
        {
            surface_prediction(
                tsdf_data, cam.getCameraIntrinsics(level), current_pose,
                truncation_distance,
                model_data.vertex_pyramid[level], model_data.normal_pyramid[level]
            );
        }

        std::cout << "frame : " << index << std::endl;

        cv::Mat normals, vertices;
        model_data.normal_pyramid[0].download(normals);
        model_data.vertex_pyramid[0].download(vertices);
        cv::imshow("n", normals);
        cv::imshow("img", img);
        int k = cv::waitKey(1);
        if (k == 'q') break;  // press q to quit
        else if (k == ' ') cv::waitKey(0);  // press space to stop

        cv::Matx44f T = cv::Matx44f::eye();
        for (int y = 0; y < 4; ++y)
        {
            for (int x = 0; x < 4; ++x)
            {
                T(y, x) = current_pose(y, x);
            }
        }
        cv::viz::WCloud point_cloud(vertices, img);
        my_window.showWidget("points", point_cloud);
        cv::viz::WCloudNormals normal_cloud(vertices, normals, 64, 0.10, cv::viz::Color::red());
        my_window.showWidget("normals", normal_cloud);

        my_window.setWidgetPose("cam", cv::Affine3f(T));

        my_window.spinOnce(10);
    }
}