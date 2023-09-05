//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ovhelper.hpp"

#include <map>
#include <string>
#include <vector>

#include <opencv2/gapi/util/variant.hpp>  // variant

struct Network {
    Path path;

    std::string stem;
    std::map<std::string, std::string> config;
    std::string priority = "NORMAL";

    LayerVariantAttr<int> input_precision;
    LayerVariantAttr<int> output_precision;
    LayerVariantAttr<std::string> input_layout;
    LayerVariantAttr<std::string> output_layout;
    LayerVariantAttr<std::string> input_model_layout;
    LayerVariantAttr<std::string> output_model_layout;
    LayerVariantAttr<std::string> input_data;
    LayerVariantAttr<std::string> output_data;
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

struct Config {
    static Config parseFromYAML(const std::string& filepath);

    std::string device = "NPU";
    std::string compiler_type = "DRIVER";
    std::string model_dir = ".";
    std::string blob_dir = ".";
    bool save_per_iteration_output_data = false;
    std::vector<Streams> multistreams;
};
