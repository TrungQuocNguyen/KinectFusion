#include <opencv2/viz/viz3d.hpp>
#include "dataset.hpp"
#include "datatypes.hpp"
#include "surface_measurement.hpp"
#include <string>

#include "Eigen/Core"
#include "opencv2/core/eigen.hpp"

//Forward declarations
void compute_pose_estimate(
    const GpuMat& vertex_map,
    const GpuMat& normal_map, 
    const GpuMat& pred_vertex_map,
    const GpuMat& pred_normal_map,
    const GpuMat& valid_vertex_mask,
    const Eigen::Matrix4f& T_g_k, 
    Eigen::Matrix<float, 6, 6, Eigen::RowMajor>& left,
    Eigen::Matrix<float, 6, 1>& right,
    const float& threshold_dist, 
    const float& threshold_angle
);

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
using namespace std;
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int main()
{
    Dataset dataset;
    dataset = TUMRGBDDataset("../data/TUMRGBD/rgbd_dataset_freiburg1_desk/", TUMRGBDDataset::TUMRGBD::FREIBURG1);

    CameraIntrinsics cam = dataset.getCameraIntrinsics();
    Eigen::Matrix4f current_pose = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f T_inc;


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
    int kernel_size {5};
    float sigma_color {1.f};
    float sigma_spatial {1.f};
    PreprocessedData data(num_levels);
    PreprocessedData data2(num_levels);
    
    //std::vector<int> iterations {4, 5, 10}; // iterations for icp in pose estimation (level 3, level 2, level 1)
    std::vector<int> iterations {2, 3, 7}; // iterations for icp in pose estimation (level 3, level 2, level 1)
    float threshold_dist {10.f};
    float threshold_angle {20.f};
    
    //for (int index = 0; index < dataset.size()-1; ++index)
    for (int index = 0; index < 200; ++index)
    {
        cv::Mat img, depth;
        dataset.getData(index+1, img, depth);
        cv::Mat img2, depth2;
        dataset.getData(index, img2, depth2);
        depth *= 1000.f;  // m -> mm
        depth2 *= 1000.f;  // m -> mm

        surface_measurement(data, depth, img, num_levels, kernel_size, sigma_color, sigma_spatial, cam, 4000.f);
        surface_measurement(data2, depth2, img2, num_levels, kernel_size, sigma_color, sigma_spatial, cam, 4000.f);

        current_pose = Eigen::Matrix4f::Identity();
        if (index != 0){
            for (int i = num_levels-1; i > -1; i--)     // from coarse to fine
            {
                for (int j = 0; j < iterations[i]; j++)
                {
                    Eigen::Matrix<float, 6, 1> right;
                    Eigen::Matrix<float, 6, 6, Eigen::RowMajor> left;
                    right.setZero();
                    left.setZero();

                    compute_pose_estimate(
                                data.vertex_pyramid[i], data.normal_pyramid[i], 
                                data2.vertex_pyramid[i], data2.normal_pyramid[i], 
                                data.valid_vertex_mask[i], 
                                current_pose, 
                                left, right, 
                                threshold_dist, threshold_angle);

                    //Eigen::Matrix<float, 6, 1> x = left.llt().solve(right);
                    Eigen::Matrix<float, 6, 1> x = left.fullPivLu().solve(right).cast<float>();

                    T_inc <<    1,     x[2],  -x[1], x[3],
                                -x[2], 1,     x[0],  x[4],
                                x[1],  -x[0], 1,     x[5],
                                0,     0,     0,     1;
                    std::cout << "T_inc: " << T_inc << std::endl;            
                    std::cout << "left: " << left << std::endl;
                    std::cout << "right: " << right << std::endl;
                    current_pose = T_inc * current_pose;
                    //current_pose = identity;
                }
            }
        }

        std::cout << "frame : " << index << std::endl;

        cv::Mat normals, vertices;
        data.normal_pyramid[0].download(normals);
        data.vertex_pyramid[0].download(vertices);
        cv::Mat normals2, vertices2;
        data2.normal_pyramid[0].download(normals2);
        data2.vertex_pyramid[0].download(vertices2);


        Eigen::Matrix3f R_g_k = current_pose.block<3, 3>(0, 0);
        Eigen::Vector3f t_g_k = current_pose.block<3, 1>(0, 3);

        cv::Mat R(3,3,CV_32FC1);
        cv::Mat t(3,1,CV_32FC1);
        cv::eigen2cv(R_g_k, R);
        cv::eigen2cv(t_g_k, t);
        
        cv::Mat vert(vertices.rows, vertices.cols, CV_32FC3); 
        cv::Mat norma(vertices.rows, vertices.cols, CV_32FC3); 

        for (int i = 0; i < vertices.rows; i++)
        {
            for (int j = 0; j < vertices.cols; j++)
            {
                vert.at<float3>(i,j) = make_float3(
                    R.at<float>(0,0)*vertices.at<float3>(i,j).x + R.at<float>(0,1)*vertices.at<float3>(i,j).y + R.at<float>(0,2)*vertices.at<float3>(i,j).z + t.at<float>(0),
                    R.at<float>(1,0)*vertices.at<float3>(i,j).x + R.at<float>(1,1)*vertices.at<float3>(i,j).y + R.at<float>(1,2)*vertices.at<float3>(i,j).z + t.at<float>(1),
                    R.at<float>(2,0)*vertices.at<float3>(i,j).x + R.at<float>(2,1)*vertices.at<float3>(i,j).y + R.at<float>(2,2)*vertices.at<float3>(i,j).z + t.at<float>(2)
                );
                norma.at<float3>(i,j) = make_float3(
                    R.at<float>(0,0)*normals.at<float3>(i,j).x + R.at<float>(0,1)*normals.at<float3>(i,j).y + R.at<float>(0,2)*normals.at<float3>(i,j).z,
                    R.at<float>(1,0)*normals.at<float3>(i,j).x + R.at<float>(1,1)*normals.at<float3>(i,j).y + R.at<float>(1,2)*normals.at<float3>(i,j).z,
                    R.at<float>(2,0)*normals.at<float3>(i,j).x + R.at<float>(2,1)*normals.at<float3>(i,j).y + R.at<float>(2,2)*normals.at<float3>(i,j).z
                );
            }
            
        }

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
        //cv::viz::WCloud point_cloud(vert, img);
        cv::viz::WCloud point_cloud(vert, cv::viz::Color::red());
        my_window.showWidget("points", point_cloud);
        //cv::viz::WCloudNormals normal_cloud(vert, normals, 64, 0.10, cv::viz::Color::red());
        //my_window.showWidget("normals", normal_cloud);

        //cv::viz::WCloud point_cloud2(vertices2, img2);
        
        cv::viz::WCloud point_cloud2(vertices2, cv::viz::Color::green());
        my_window.showWidget("points2", point_cloud2);
        /*cv::viz::WCloudNormals normal_cloud2(vertices2, normals2, 64, 0.10, cv::viz::Color::green());
        my_window.showWidget("normals2", normal_cloud2);*/

        my_window.setWidgetPose("cam", cv::Affine3f(T));

        my_window.spinOnce(200);
    }
}