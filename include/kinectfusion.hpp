#pragma once
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include "utils.hpp"
#include "datatypes.hpp"
#include "surface_measurement.hpp"
#include "surface_prediction.hpp"
#include "pose_estimation.hpp"


void surfaceReconstruction(
    const cv::cuda::GpuMat& depth, 
    const CameraParameters& cam, const Eigen::Matrix4f& T_c_w,
    TSDFData& volume
);


PointCloud extractPointCloud(const TSDFData& volume, const int buffer_size);


void exportPly(const std::string& filename, const PointCloud& point_cloud)
{
    std::ofstream file_out { filename };
    if (!file_out.is_open()) return;

    file_out << "ply" << std::endl;
    file_out << "format ascii 1.0" << std::endl;
    file_out << "element vertex " << point_cloud.num_points << std::endl;
    file_out << "property float x" << std::endl;
    file_out << "property float y" << std::endl;
    file_out << "property float z" << std::endl;
    file_out << "property float nx" << std::endl;
    file_out << "property float ny" << std::endl;
    file_out << "property float nz" << std::endl;
    file_out << "end_header" << std::endl;

    for (int i = 0; i < point_cloud.num_points; ++i) {
        float3 vertex = point_cloud.vertices.ptr<float3>(0)[i];
        float3 normal = point_cloud.normals.ptr<float3>(0)[i];
        file_out << vertex.x << " " << vertex.y << " " << vertex.z << " " << normal.x << " " << normal.y << " "
                    << normal.z << std::endl;
    }
}


class KinectFusion
{
public:
    KinectFusion(CameraParameters cam) : cam_(cam) {}

    void readConfig()
    {
        num_levels_ = Config::get<int>("num_levels");
        
        // birateral filter
        kernel_size_ = Config::get<int>("bf_kernel_size");
        sigma_color_  = Config::get<float>("bf_sigma_color");
        sigma_spatial_ = Config::get<float>("bf_sigma_spatial");

        // icp
        distance_threshold_ = Config::get<float>("icp_distance_threshold");
        angle_threshold_ = Config::get<float>("icp_angle_threshold");
        for (int i = 0; i < num_levels_; ++i)
        {
            icp_iterations_.push_back(Config::get<int>("icp_iteration_" + std::to_string(i)));
        }
        flag_use_gt_pose_ = Config::get<int>("flag_use_gt_pose") == 1;

        // tsdf
        tsdf_data_ = TSDFData(
            make_int3(Config::get<int>("tsdf_size_x"), Config::get<int>("tsdf_size_y"), Config::get<int>("tsdf_size_z")), 
            Config::get<int>("tsdf_scale"), Config::get<float>("truncation_distance")
        );

        model_data_ = ModelData(num_levels_, cam_);
        current_frame_ = FrameData(num_levels_, cam_);
    }

    void setPose(Eigen::Matrix4f &pose)
    {
        T_g_k_ = pose;
    }

    bool addFrame(cv::Mat &img, cv::Mat &depth, std::string time_stamp)
    {
        static int frame_factory_id = 0;
        static float sum_time = 0.;
        Timer timer("Frame " + std::to_string(frame_factory_id));
        
        // FrameData frame(num_levels_, cam_);
        time_stamps_.push_back(time_stamp);

        surfaceMeasurement(depth, img, num_levels_, kernel_size_, sigma_color_, sigma_spatial_, cam_, current_frame_);
        timer.print("Surface Measurement");

        if (frame_factory_id == 0)
        {
            T_g_k_ = Eigen::Matrix4f::Identity();
            estimated_poses_.push_back(T_g_k_);
        }
        else
        {
            if (!flag_use_gt_pose_)
            {
                bool icp_success = poseEstimation(
                    model_data_, current_frame_, cam_, num_levels_, distance_threshold_, angle_threshold_,
                    icp_iterations_, T_g_k_
                );
                timer.print("Pose Estimation");

                estimated_poses_.push_back(T_g_k_);
                if (!icp_success) return false;
            }
        }

        surfaceReconstruction(current_frame_.depth_pyramid[0], cam_, T_g_k_, tsdf_data_);
        timer.print("Surface Reconstruction");

        surfacePrediction(tsdf_data_, cam_, T_g_k_, num_levels_, model_data_);
        timer.print("Surface Prediction");

        sum_time += timer.print();
        printf("[ FPS ] : %f\n", (frame_factory_id + 1) * 1000.f / sum_time);

        frame_factory_id++;
        return true;
    }

    void savePoses(std::string filename, Eigen::Matrix4f init_pose = Eigen::Matrix4f::Identity())
    {
        std::ofstream ofs(filename);
        for (int index = 0; index < estimated_poses_.size(); ++index)
        {
            Eigen::Matrix4f pose = estimated_poses_[index];
            pose.block<3, 1>(0, 3) *= 0.001f;  // save in meter
            pose = pose * init_pose;

            Eigen::Vector3f t = pose.block<3, 1>(0, 3);
            auto q = Eigen::Quaternionf(pose.block<3, 3>(0, 0));
            ofs << time_stamps_[index] << " " << 
                t.x() << " " << t.y() << " " << t.z() << " " << 
                q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }
        ofs.close();
    }

    void savePointcloud(const std::string filename, const int buffer_size)
    {
        PointCloud pc = extractPointCloud(tsdf_data_, buffer_size);
        exportPly(filename, pc);
    }

    void showImages(cv::Mat &img, cv::Mat &depth)
    {
        cv::Mat measured_normals, measured_depth, pred_normals;
        model_data_.normal_pyramid[0].download(pred_normals);
        current_frame_.normal_pyramid[0].download(measured_normals);
        current_frame_.depth_pyramid[0].download(measured_depth);
        cv::imshow("raw img", img);
        cv::imshow("raw depth", depth / 1000.f);
        cv::imshow("measured normals", measured_normals);
        cv::imshow("measured depth", measured_depth / 1000.f);
        cv::imshow("predicted normal", pred_normals);
        cv::waitKey(1);        
    }

    void visualize3D(cv::viz::Viz3d &window, cv::Mat &img, bool flag_show_all_points = false)
    {
        cv::Mat pred_normals, pred_vertices;
        model_data_.normal_pyramid[0].download(pred_normals);
        model_data_.vertex_pyramid[0].download(pred_vertices);

        cv::Matx44f T = cv::Matx44f::eye();
        for (int y = 0; y < 4; ++y)
        {
            for (int x = 0; x < 4; ++x)
            {
                T(y, x) = T_g_k_.matrix()(y, x);
            }
        }
        cv::viz::WCloud pred_point_cloud(pred_vertices, img);
        window.showWidget("predicted vertices", pred_point_cloud);

        cv::viz::WCloudNormals pred_normal_cloud(pred_vertices, pred_normals, 64, 1, cv::viz::Color::red());
        window.showWidget("predicted normals", pred_normal_cloud);

        if (flag_show_all_points)
        {
            PointCloud pc = extractPointCloud(tsdf_data_, 3 * 1000000);
            cv::viz::WCloud point_cloud(pc.vertices);
            window.showWidget("points", point_cloud);
        }

        window.setWidgetPose("cam", cv::Affine3f(T));
        window.spinOnce(1);
    }
    
private:
    CameraParameters cam_;
    int num_levels_ = 3;
    
    // bilateral filter
    int kernel_size_ = 7;
    float sigma_color_ = 40.f;
    float sigma_spatial_ = 4.5f;
    
    // icp
    bool flag_use_gt_pose_ = false;
    float distance_threshold_ = 40.f;  // mm
    float angle_threshold_ = 30.f;  // degree
    std::vector<int> icp_iterations_;

    // tsdf
    TSDFData tsdf_data_;

    std::vector<FrameData> frames_data_;  
    FrameData current_frame_;
    ModelData model_data_;

    Eigen::Matrix4f T_g_k_;  // current pose
    std::vector<Eigen::Matrix4f> estimated_poses_;
    std::vector<std::string> time_stamps_;  // to save tum trajectory
};
