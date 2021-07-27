#pragma once
#include <iostream>
#include <chrono>
#include <memory>
#include <opencv2/core.hpp>


class Timer
{
public:
    inline Timer(std::string name) : name_(name)
    {
        start();
    }

    inline double print(std::string s="")
    {
        double duration = end();
        if (s == "")
        {
            printf("[TIMER] : [ END ] %s %f[ms]\n", name_.c_str(), duration);
        }
        else
        {
            printf("[TIMER] : [ --- ] --- %s %f[ms]\n", s.c_str(), duration);
        }        
        return duration;
    }

private:
    inline void start()
    {
        printf("[TIMER] : [START] %s\n", name_.c_str());
        start_ = std::chrono::system_clock::now();
    }

    inline double end()
    {
        // return [ms]
        end_ = std::chrono::system_clock::now();
        return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count() / 1000.0);
    }

    std::chrono::system_clock::time_point start_, end_;
    std::string name_;
};


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

    static bool read(const std::string &filename)
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
