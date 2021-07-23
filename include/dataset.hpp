#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <datatypes.hpp>


class Dataset 
{
public:
	Dataset() {}
	int size() { return size_; }

	CameraParameters getCameraParameters()
	{
		return cam_;
	}

	inline void getData(const int index, cv::Mat& img, cv::Mat& depth)
	{
		assert(0 <= index && index < size_);
		img = cv::imread(imgs_filenames_[index], cv::IMREAD_COLOR);
		depth = cv::imread(depths_filenames_[index], cv::IMREAD_UNCHANGED);
		depth.convertTo(depth, CV_32F, 1.f / 5000.f);
	}

	inline Eigen::Matrix4f getPose(const int index)
	{
		assert(0 <= index && index < size_);
		return gt_poses_[index];		
	}

protected:
	CameraParameters cam_;
	float distortions_[5] = { 0 };

	int size_;  // the number of images

	std::vector<std::string> imgs_filenames_, depths_filenames_;
	std::vector<Eigen::Matrix4f> gt_poses_;
	std::vector<std::string> time_stamps_;

	inline std::string getZeropadStr(int num, int len)
	{
		// For example: getNumOfZeropadString(1234, 6) return "001234"
		std::ostringstream oss;
		oss << std::setw(len) << std::setfill('0') << num;
		return oss.str();
	}
};


/// <summary>
/// https://vision.in.tum.de/data/datasets/rgbd-dataset
/// https://www.mrpt.org/Collection_of_Kinect_RGBD_datasets_with_ground_truth_CVPR_TUM_2011
/// </summary>
class TUMRGBDDataset : public Dataset
{
public:
	enum class TUMRGBD { FREIBURG1 = 1, FREIBURG2 = 2, FREIBURG3 = 3 };

	TUMRGBDDataset(const std::string &dataset_dir, TUMRGBD &tumrgbd)
	{
		setCameraIntrinsics(tumrgbd);

		std::ifstream ifs(dataset_dir + "associations.txt");
		if (ifs.fail())
		{
			std::cerr << "Failed to load " + dataset_dir + "associations.txt!!!!!!!!\n";
			throw std::exception();
		}

		std::string line;
		while (std::getline(ifs, line))
		{
			std::stringstream ss(line);
			Eigen::Matrix<float, 7, 1> tmp;
			for (int i = 0; i < 12; ++i)
			{
				std::string s;
				ss >> s;
				if (i == 1)
				{
					imgs_filenames_.push_back(dataset_dir + s);
				}
				else if (i == 3)
				{
					depths_filenames_.push_back(dataset_dir + s);
				}
				else if (i >= 5)
				{
					tmp(i - 5) = std::stof(s);
				}
			}
			Eigen::Quaternionf q(tmp(6), tmp(3), tmp(4), tmp(5));
			Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
			T.block<3, 3>(0, 0) = q.toRotationMatrix();
			T.block<3, 1>(0, 3) = tmp.head(3);
			gt_poses_.push_back(T);
		}
		ifs.close();
		size_ = gt_poses_.size();
	}

private:
	void setCameraIntrinsics(TUMRGBD &tumrgbd)
	{
		switch (tumrgbd)
		{
		case TUMRGBD::FREIBURG1:
			cam_ = CameraParameters(640, 480, 517.3f, 516.5f, 318.6f, 255.3f);
			distortions_[0] = 0.2624f;
			distortions_[1] = -0.9531f;
			distortions_[2] = -0.0054f;
			distortions_[3] = 0.0026f;
			distortions_[4] = 1.1633f;
			break;
		case TUMRGBD::FREIBURG2:
			cam_ = CameraParameters(640, 480, 520.9f, 521.0f, 325.1f, 249.7f);
			distortions_[0] = 0.2312f;
			distortions_[1] = -0.7849f;
			distortions_[2] = -0.0033f;
			distortions_[3] = 0.0001f;
			distortions_[4] = 0.9172f;
			break;
		case TUMRGBD::FREIBURG3:
			cam_ = CameraParameters(640, 480, 535.4f, 539.2f, 320.1f, 247.6f);
			break;
		default:
			throw std::exception();
		}
	}
};


/// <summary>
/// https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html
/// </summary>
class ICLNUIMDataset : public Dataset
{
public:
	enum class ICLNUIM
	{
		LR_KT0,  // living room
		LR_KT1,
		LR_KT2,
		LR_KT3,
		OF_KT0,  // office room
		OF_KT1,
		OF_KT2,
		OF_KT3,
	};

	ICLNUIMDataset(std::string &dataset_dir, ICLNUIM &iclnuim)
	{
		cam_ = CameraParameters(640, 480, 481.2f, -480.f, 319.5f, 239.5f);

		std::ifstream ifs(dataset_dir + getPoseFilename(iclnuim));
		if (ifs.fail())
		{
			std::cerr << "Failed to load " + dataset_dir + "associations.txt!!!!!!!!\n";
			throw std::exception();
		}

		std::string line;
		int count = 0;
		while (std::getline(ifs, line))
		{
			std::stringstream ss(line);
			for (int i = 0; i < 8; ++i)
			{
				std::string s;
				Eigen::Matrix<float, 7, 1> tmp;
				ss >> s;
				if (i == 0)
				{
					time_stamps_.push_back(s);
					imgs_filenames_.push_back(dataset_dir + "rgb/" + s + ".png");
					depths_filenames_.push_back(dataset_dir + "depth/" + s + ".png");
				}
				else
				{
					tmp(i - 1) = std::stof(s);
				}
				Eigen::Quaternionf q(tmp(6), tmp(3), tmp(4), tmp(5));
				Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
				T.block<3, 3>(0, 0) = q.toRotationMatrix();
				T.block<3, 1>(0, 3) = tmp.head(3);
				gt_poses_.push_back(T);
			}
			++count;
		}
		ifs.close();
		size_ = count;
	}

private:
	std::string getPoseFilename(ICLNUIM &iclnuim)
	{
		switch (iclnuim)
		{
		case ICLNUIM::LR_KT0:
			return "livingRoom0.gt.freiburg";
		case ICLNUIM::LR_KT1:
			return "livingRoom1.gt.freiburg";
		case ICLNUIM::LR_KT2:
			return "livingRoom2.gt.freiburg";
		case ICLNUIM::LR_KT3:
			return "livingRoom3.gt.freiburg";
		case ICLNUIM::OF_KT0:
			return "traj0.gt.freiburg";
		case ICLNUIM::OF_KT1:
			return "traj1.gt.freiburg";
		case ICLNUIM::OF_KT2:
			return "traj2.gt.freiburg";
		case ICLNUIM::OF_KT3:
			return "traj3.gt.freiburg";
		default:
			throw std::exception();
		}
	}
};