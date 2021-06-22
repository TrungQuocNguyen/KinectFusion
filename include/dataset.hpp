#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>


// TODO: get camera matrix
class Dataset {
public:
	int size() { return size_; }
	void getCameraParameters(float &fx, float &fy, float &cx, float &cy)
	{
		fx = fx_;
		fy = fy_;
		cx = cx_;
		cy = cy_;
	}
	void getCameraDistortions(float *distortions)
	{
		distortions = distortions_;
	}

protected:
	Dataset() {}

	int width_, height_;
	float fx_, fy_, cx_, cy_;
	float distortions_[5] = { 0 };

	int size_;  // the number of images

	std::vector<std::string> imgs_filenames_, depths_filenames_;
	
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
	enum class TUMRGBD { FREIBURG1, FREIBURG2, FREIBURG3 };

	TUMRGBDDataset(const std::string dataset_dir, TUMRGBD tumrgbd)
	{
		width_ = 640;
		height_ = 480;
		setCalibrationParameters(tumrgbd);

		std::ifstream ifs(dataset_dir + "associations.txt");
		int count = 0;
		std::string line;
		while (std::getline(ifs, line))
		{
			std::stringstream ss(line);
			for (int i = 0; i < 4; ++i)
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
			}
			++count;
		}
		ifs.close();
		size_ = count;
	}

	void getData(int index, cv::Mat& img, cv::Mat& depth, bool flag_rgb = true)
	{
		assert(0 <= index && index < size_);
		img = cv::imread(imgs_filenames_[index], flag_rgb);
		depth = cv::imread(depths_filenames_[index], cv::IMREAD_UNCHANGED);
		depth.convertTo(depth, CV_32F, 1.f / 5000.f);
	}

private:
	void setCalibrationParameters(TUMRGBD tumrgbd)
	{
		switch (tumrgbd)
		{
		case TUMRGBD::FREIBURG1:
			fx_ = 517.3f;
			fy_ = 516.5f;
			cx_ = 318.6f;
			cy_ = 255.3f;
			distortions_[0] = 0.2624f;
			distortions_[1] = -0.9531f;
			distortions_[2] = -0.0054f;
			distortions_[3] = 0.0026f;
			distortions_[4] = 1.1633f;
			break;
		case TUMRGBD::FREIBURG2:
			fx_ = 520.9f;
			fy_ = 521.0f;
			cx_ = 325.1f;
			cy_ = 249.7f;
			distortions_[0] = 0.2312f;
			distortions_[1] = -0.7849f;
			distortions_[2] = -0.0033f;
			distortions_[3] = 0.0001f;
			distortions_[4] = 0.9172f;
			break;
		case TUMRGBD::FREIBURG3:
			fx_ = 535.4f;
			fy_ = 539.2f;
			cx_ = 320.1f;
			cy_ = 247.6f;
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

	ICLNUIMDataset(std::string dataset_dir, ICLNUIM iclnuim)
	{
		fx_ = 481.2f;
		fy_ = -480.f;
		cx_ = 319.5f;
		cy_ = 239.5f;

		std::ifstream ifs(dataset_dir + getPoseFilename(iclnuim));
		std::string line;
		int count = 0;
		while (std::getline(ifs, line))
		{
			std::stringstream ss(line);
			for (int i = 0; i < 8; ++i)
			{
				std::string s;
				ss >> s;
				if (i == 0)
				{
					time_stamps_.push_back(s);
					imgs_filenames_.push_back(dataset_dir + "rgb/" + s + ".png");
					depths_filenames_.push_back(dataset_dir + "depth/" + s + ".png");
				}
			}
			++count;
		}
		ifs.close();
		size_ = count;
	}
	
	void getData(int index, cv::Mat& img, cv::Mat& depth, bool flag_rgb=true)
	{
		assert(0 <= index && index < size_);
		img = cv::imread(imgs_filenames_[index], flag_rgb);
		depth = cv::imread(depths_filenames_[index], cv::IMREAD_UNCHANGED);
		depth.convertTo(depth, CV_32F, 1.f / 5000.f);
	}

private:
	std::string getPoseFilename(ICLNUIM iclnuim)
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