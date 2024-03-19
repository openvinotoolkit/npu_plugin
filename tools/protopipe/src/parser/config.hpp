// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "scenario/accuracy_metrics.hpp"
#include "scenario/ovhelper.hpp"
#include "utils/data_providers.hpp"

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
    LayerVariantAttr<IRandomGenerator::Ptr> random_generators;
    LayerVariantAttr<IAccuracyMetric::Ptr> accuracy_metrics;

    size_t nireq = 1;
    std::string device;
};

struct Stream {
    std::string name;
    std::vector<std::vector<Network>> networks;
    uint32_t target_fps;
    uint64_t delay_in_us;
    uint64_t after_stream_delay;
    cv::util::optional<uint32_t> target_latency;
    cv::util::optional<uint64_t> exec_time_in_secs;
    cv::util::optional<uint64_t> iteration_count;
};

struct MultiInference {
    std::string name;
    std::vector<Stream> streams;
};

struct ScenarioConfig {
    static ScenarioConfig parseFromYAML(const std::string& filepath);

    std::string log_level = "NONE";
    std::string compiler_type = "DRIVER";
    std::string model_dir = ".";
    std::string blob_dir = ".";
    std::vector<MultiInference> multi_inferences;

    IRandomGenerator::Ptr global_random_generator;
    IAccuracyMetric::Ptr global_accuracy_metric;

    cv::util::optional<std::filesystem::path> per_iter_outputs_path;
};
