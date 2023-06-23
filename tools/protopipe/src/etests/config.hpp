#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/gapi/util/variant.hpp>  // variant

struct Network {
    // NB: *.xml & *.bin, *.onnx, *.pdpd
    struct ModelPath {
        std::string model;
        std::string bin;
    };
    struct BlobPath {
        std::string path;
    };

    using Path = cv::util::variant<ModelPath, BlobPath>;
    Path path;

    std::string stem;
    std::map<std::string, std::string> config;
    std::string priority = "NORMAL";
    int ip = -1;
    int op = -1;

    using FileNameT = cv::util::variant<cv::util::monostate, std::map<std::string, std::string>, std::string>;
    FileNameT input_filename;
    FileNameT output_filename;
};

struct Stream {
    std::vector<Network> networks;
    uint32_t target_fps;
    uint64_t delay_in_us;
    uint64_t after_stream_delay;
    size_t exec_time_in_secs;
    size_t iteration_count;
};
using Streams = std::vector<Stream>;

struct ETestsConfig {
    static ETestsConfig parseFromYAML(const std::string& filepath);

    std::string device = "VPU";
    std::string compiler_type = "DRIVER";
    std::string model_dir = ".";
    std::string blob_dir = ".";
    bool save_per_iteration_output_data = false;
    std::vector<Streams> multistreams;
};
