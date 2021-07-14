#pragma once
#include <iostream>

class Config
{
private:
    static std::shared_ptr<Config> config_;
    cv::FileStorage file_;
    Config() {}

public:
    ~Config()
    {
        if (file_.isOpened())
        {
            file_.release();
        }
    }

    static bool setParameterFile(const std::string &filename)
    {
        if (config_ == nullptr)
        {
            config_ = std::shared_ptr<Config>(new Config);
        }
        
        config_->file_.open(filename.c_str(), cv::FileStorage::READ);
        if (config_->file_.isOpened() == false)
        {
            std::cout << filename << " does not exist!!!\n";
            config_->file_.release();
            return false;
        }
        return true;
    }

    template <typename T>
    static T get(const std::string &key)
    {
        return T(Config::config_->file_[key]);
    }
};

std::shared_ptr<Config> Config::config_ = nullptr;
