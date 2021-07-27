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


void surface_reconstruction(
    const cv::cuda::GpuMat& depth, 
    const CameraParameters& cam,
    const Eigen::Matrix4f& T_c_w,
    const float& truncation_distance,
    TSDFData& volume
);


PointCloud extract_pointcloud(const TSDFData& volume, const int buffer_size);


void export_ply(const std::string& filename, const PointCloud& point_cloud)
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
    KinectFusion(CameraParameters cam) : cam_(cam) {}

    void read_config()
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
        truncation_distance_ = Config::get<float>("truncation_distance");
        tsdf_data_ = TSDFData(
            make_int3(Config::get<int>("tsdf_size_x"), Config::get<int>("tsdf_size_y"), Config::get<int>("tsdf_size_z")), 
            Config::get<int>("tsdf_scale")
        );

        model_data_ = ModelData(num_levels_, cam_);
    }

    void set_pose(Eigen::Matrix4f &pose, std::string time_stamp)
    {
        T_g_k_ = pose;
    }

    bool add_frame(cv::Mat &img, cv::Mat &depth)
    {
        static int frame_factory_id = 0;
        static float sum_time = 0.;
        Timer timer("Frame " + std::to_string(frame_factory_id));
        
        FrameData frame(num_levels_, cam_);

        surface_measurement(depth, img, num_levels_, kernel_size_, sigma_color_, sigma_spatial_, cam_, frame);
        timer.print("Surface Measurement");

        if (frame_factory_id != 0)
        {
            if (!flag_use_gt_pose_)
            {
                bool icp_success = pose_estimation(
                    model_data_, frame, cam_, num_levels_, distance_threshold_, angle_threshold_,
                    icp_iterations_, T_g_k_
                );
                timer.print("Pose Estimation");

                estimated_poses_.push_back(T_g_k_);
                if (!icp_success) return false;
            }
        }

        surface_reconstruction(frame.depth_pyramid[0], cam_, T_g_k_, truncation_distance_, tsdf_data_);
        timer.print("Surface Reconstruction");

        surface_prediction(tsdf_data_, cam_, T_g_k_, truncation_distance_, num_levels_, model_data_);
        timer.print("Surface Prediction");

        sum_time += timer.print();
        printf("[ FPS ] : %f\n", (frame_factory_id + 1) * 1000.f / sum_time);

        frame_factory_id++;
        return true;
    }

    void save_poses(std::string file_name, Eigen::Matrix4f init_pose = Eigen::Matrix4f::Identity())
    {
        std::ofstream ofs(file_name + "_pose.txt");
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
    float truncation_distance_;  // mm
    TSDFData tsdf_data_;

    std::vector<FrameData> frames_data_;  
    ModelData model_data_;

    Eigen::Matrix4f T_g_k_;  // current pose
    std::vector<Eigen::Matrix4f> estimated_poses_;
    std::vector<std::string> time_stamps_;  // to save tum trajectory
};
