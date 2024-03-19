// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "parser/config.hpp"

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>  // depth

#include "utils/error.hpp"

namespace fs = std::filesystem;

static int toDepth(const std::string& prec) {
    if (prec == "FP32")
        return CV_32F;
    if (prec == "FP16")
        return CV_16F;
    if (prec == "U8")
        return CV_8U;
    throw std::logic_error("Unsupported precision type: " + prec);
}

static AttrMap<int> toDepth(const AttrMap<std::string>& attrmap) {
    AttrMap<int> depthmap;
    for (const auto& [name, str_depth] : attrmap) {
        depthmap.emplace(name, toDepth(str_depth));
    }
    return depthmap;
}

static LayerVariantAttr<int> toDepth(const LayerVariantAttr<std::string>& attr) {
    LayerVariantAttr<int> depthattr;
    if (cv::util::holds_alternative<std::string>(attr)) {
        depthattr = toDepth(cv::util::get<std::string>(attr));
    } else {
        depthattr = toDepth(cv::util::get<AttrMap<std::string>>(attr));
    }
    return depthattr;
}

namespace YAML {

template <typename K, typename V>
struct convert<std::map<K, V>> {
    static bool decode(const Node& node, std::map<K, V>& map) {
        if (!node.IsMap()) {
            return false;
        }
        for (const auto& itr : node) {
            map.emplace(itr.first.as<K>(), itr.second.as<V>());
        }
        return true;
    }
};

template <typename T>
struct convert<LayerVariantAttr<T>> {
    static bool decode(const Node& node, LayerVariantAttr<T>& layer_attr) {
        if (node.IsMap()) {
            layer_attr = node.as<std::map<std::string, T>>();
        } else {
            layer_attr = node.as<T>();
        }
        return true;
    }
};

template <>
struct convert<LayerVariantAttr<IRandomGenerator::Ptr>> {
    static bool decode(const Node& node, LayerVariantAttr<IRandomGenerator::Ptr>& layer_attr) {
        if (!node.IsMap()) {
            return false;
        }
        if (node.begin()->second.IsMap()) {
            layer_attr = node.as<std::map<std::string, IRandomGenerator::Ptr>>();
        } else {
            layer_attr = node.as<IRandomGenerator::Ptr>();
        }
        return true;
    }
};

template <>
struct convert<UniformGenerator::Ptr> {
    static bool decode(const Node& node, UniformGenerator::Ptr& generator) {
        if (!node["low"]) {
            THROW_ERROR("Uniform distribution must have \"low\" attribute");
        }
        if (!node["high"]) {
            THROW_ERROR("Uniform distribution must have \"high\" attribute");
        }
        generator = std::make_shared<UniformGenerator>(node["low"].as<double>(), node["high"].as<double>());
        return true;
    }
};

template <>
struct convert<IRandomGenerator::Ptr> {
    static bool decode(const Node& node, IRandomGenerator::Ptr& generator) {
        if (!node["dist"]) {
            THROW_ERROR("\"random\" must have \"dist\" attribute!");
        }
        const auto dist = node["dist"].as<std::string>();
        if (dist == "uniform") {
            generator = node.as<UniformGenerator::Ptr>();
        } else {
            THROW_ERROR("Unsupported random distribution: \"" << dist << "\"");
        }
        return true;
    }
};

template <>
struct convert<LayerVariantAttr<IAccuracyMetric::Ptr>> {
    static bool decode(const Node& node, LayerVariantAttr<IAccuracyMetric::Ptr>& layer_attr) {
        if (!node.IsMap()) {
            return false;
        }
        if (node.begin()->second.IsMap()) {
            layer_attr = node.as<std::map<std::string, IAccuracyMetric::Ptr>>();
        } else {
            layer_attr = node.as<IAccuracyMetric::Ptr>();
        }
        return true;
    }
};

template <>
struct convert<Norm::Ptr> {
    static bool decode(const Node& node, Norm::Ptr& metric) {
        // NB: If bigger than tolerance - fail.
        if (!node["tolerance"]) {
            THROW_ERROR("Metric \"norm\" must have \"tolerance\" attribute!");
        }
        const auto tolerance = node["tolerance"].as<double>();
        metric = std::make_shared<Norm>(tolerance);
        return true;
    }
};

template <>
struct convert<Cosine::Ptr> {
    static bool decode(const Node& node, Cosine::Ptr& metric) {
        // NB: If lower than threshold - fail.
        if (!node["threshold"]) {
            THROW_ERROR("Metric \"cosine\" must have \"threshold\" attribute!");
        }
        const auto threshold = node["threshold"].as<double>();
        metric = std::make_shared<Cosine>(threshold);
        return true;
    }
};

template <>
struct convert<IAccuracyMetric::Ptr> {
    static bool decode(const Node& node, IAccuracyMetric::Ptr& metric) {
        const auto type = node["name"].as<std::string>();
        if (type == "norm") {
            metric = node.as<Norm::Ptr>();
        } else if (type == "cosine") {
            metric = node.as<Cosine::Ptr>();
        } else {
            THROW_ERROR("Unsupported metric type: " << type);
        }
        return true;
    }
};

template <>
struct convert<Network> {
    static bool decode(const Node& node, Network& network) {
        const std::string name = node["name"].as<std::string>();
        fs::path path{name};
        network.stem = path.stem().string();

        if (path.extension() == ".xml") {
            auto bin_path{path};
            bin_path.replace_extension(".bin");
            network.path = ModelPath{path.string(), bin_path.string()};
        } else if (path.extension() == ".blob") {
            network.path = BlobPath{path.string()};
        } else if (path.extension() == ".onnx" || path.extension() == ".pdpd") {
            network.path = ModelPath{path.string(), ""};
        } else {
            throw std::logic_error("Unsupported model format: " + name);
        }

        if (node["ip"]) {
            network.input_precision = toDepth(node["ip"].as<LayerVariantAttr<std::string>>());
        }

        if (node["op"]) {
            network.output_precision = toDepth(node["op"].as<LayerVariantAttr<std::string>>());
        }

        if (node["il"]) {
            network.input_layout = node["il"].as<LayerVariantAttr<std::string>>();
        }

        if (node["ol"]) {
            network.output_layout = node["ol"].as<LayerVariantAttr<std::string>>();
        }

        if (node["iml"]) {
            network.input_model_layout = node["iml"].as<LayerVariantAttr<std::string>>();
        }

        if (node["oml"]) {
            network.output_model_layout = node["oml"].as<LayerVariantAttr<std::string>>();
        }

        if (node["priority"]) {
            network.priority = node["priority"].as<std::string>();
        }

        if (node["config"]) {
            network.config = node["config"].as<std::map<std::string, std::string>>();
        }

        if (node["input_data"]) {
            network.input_data = node["input_data"].as<LayerVariantAttr<std::string>>();
        }

        if (node["output_data"]) {
            network.output_data = node["output_data"].as<LayerVariantAttr<std::string>>();
        }

        if (node["nireq"]) {
            network.nireq = node["nireq"].as<size_t>();
        }

        if (node["device"]) {
            network.device = node["device"].as<std::string>();
        }

        if (node["random"]) {
            network.random_generators = node["random"].as<LayerVariantAttr<IRandomGenerator::Ptr>>();
        }

        if (node["metric"]) {
            network.accuracy_metrics = node["metric"].as<LayerVariantAttr<IAccuracyMetric::Ptr>>();
        }

        return true;
    }
};

template <>
struct convert<Stream> {
    static bool decode(const Node& node, Stream& stream) {
        if (node["name"]) {
            stream.name = node["name"].as<std::string>();
        }

        for (auto network : node["network"]) {
            if (network.IsSequence()) {
                stream.networks.push_back(network.as<std::vector<Network>>());
            } else {
                stream.networks.push_back(std::vector<Network>{network.as<Network>()});
            }
        }

        stream.target_fps = node["target_fps"] ? node["target_fps"].as<uint32_t>() : 0;
        if (node["target_latency"]) {
            stream.target_latency = cv::util::make_optional(node["target_latency"].as<uint32_t>());
        }
        stream.delay_in_us = node["delay_in_us"] ? node["delay_in_us"].as<uint64_t>() : 0;
        stream.after_stream_delay = node["after_stream_delay"] ? node["after_stream_delay"].as<uint64_t>() : 0;

        if (node["iteration_count"]) {
            stream.iteration_count = cv::util::make_optional(node["iteration_count"].as<uint64_t>());
        }

        if (node["exec_time_in_secs"]) {
            stream.exec_time_in_secs = cv::util::make_optional(node["exec_time_in_secs"].as<uint64_t>());
        }

        return true;
    }
};

template <>
struct convert<MultiInference> {
    static bool decode(const Node& node, MultiInference& multi_inference) {
        if (node["name"]) {
            multi_inference.name = node["name"].as<std::string>();
        }
        multi_inference.streams = node["input_stream_list"].as<std::vector<Stream>>();
        return true;
    }
};

template <typename T>
struct convert<std::vector<T>> {
    static bool decode(const Node& node, std::vector<T>& vec) {
        if (!node.IsSequence()) {
            return false;
        }

        for (auto& child : node) {
            vec.push_back(child.as<T>());
        }

        return true;
    }
};

}  // namespace YAML

static void clarifyNetworkPath(Path& path, const std::string& blob_dir, const std::string& model_dir) {
    if (cv::util::holds_alternative<ModelPath>(path)) {
        auto& model_path = cv::util::get<ModelPath>(path);
        fs::path model_file_path{model_path.model};
        fs::path bin_file_path{model_path.bin};

        if (model_file_path.is_relative()) {
            model_path.model = (model_dir / model_file_path).string();
        }
        if (!model_path.bin.empty() && bin_file_path.is_relative()) {
            model_path.bin = (model_dir / bin_file_path).string();
        }
    } else {
        ASSERT(cv::util::holds_alternative<BlobPath>(path));
        auto& blob = cv::util::get<BlobPath>(path);
        fs::path blob_path{blob.path};

        if (blob_path.is_relative()) {
            blob.path = (blob_dir / blob_path).string();
        }
    }
}

static void clarifyPathForAllNetworks(ScenarioConfig& config) {
    for (auto& multi_inference : config.multi_inferences) {
        for (auto& stream : multi_inference.streams) {
            for (auto& network_list : stream.networks) {
                for (auto& network : network_list) {
                    clarifyNetworkPath(network.path, config.blob_dir, config.model_dir);
                }
            }
        }
    }
}

static void clarifyDeviceForAllNetworks(std::vector<MultiInference>& multi_inferences,
                                        const std::string& global_device) {
    for (auto& multi_inference : multi_inferences) {
        for (auto& stream : multi_inference.streams) {
            for (auto& network_list : stream.networks) {
                for (auto& network : network_list) {
                    if (network.device.empty()) {
                        network.device = global_device;
                    }
                }
            }
        }
    }
}

ScenarioConfig ScenarioConfig::parseFromYAML(const std::string& filepath) {
    auto node = YAML::LoadFile(filepath);

    ScenarioConfig cfg;
    if (node["npu_compiler_type"]) {
        cfg.compiler_type = node["npu_compiler_type"].as<std::string>();
    }

    if (node["model_dir"]) {
        if (!node["model_dir"]["local"]) {
            // TODO: Wrap this logic.
            std::stringstream ss;
            ss << "Failed to parse: " << filepath << "\nmodel_dir must contain \"local\" key";
            throw std::logic_error(ss.str());
        }
        cfg.model_dir = node["model_dir"]["local"].as<std::string>();
    }

    if (node["blob_dir"]) {
        if (!node["blob_dir"]["local"]) {
            std::stringstream ss;
            ss << "Failed to parse: " << filepath << "\nblob_dir must contain \"local\" key";
            throw std::logic_error(ss.str());
        }
        cfg.blob_dir = node["blob_dir"]["local"].as<std::string>();
    }

    if (node["metric"]) {
        cfg.global_accuracy_metric = node["metric"].as<IAccuracyMetric::Ptr>();
    } else {
        cfg.global_accuracy_metric = std::make_shared<Norm>(0.f);
    }

    if (node["random"]) {
        cfg.global_random_generator = node["random"].as<IRandomGenerator::Ptr>();
    } else {
        cfg.global_random_generator = std::make_shared<UniformGenerator>(0.0, 255.0);
    }

    if (node["multi_inference"]) {
        cfg.multi_inferences = node["multi_inference"].as<std::vector<MultiInference>>();
    } else {
        std::stringstream ss;
        ss << "Failed to parse: " << filepath << "\nConfig must contain \"multi_inference\" key";
        throw std::logic_error(ss.str());
    }

    if (node["log_level"]) {
        cfg.log_level = node["log_level"].as<std::string>();
    }

    if (node["save_validation_outputs"]) {
        const auto path = node["save_validation_outputs"].as<std::string>();
        cfg.per_iter_outputs_path = cv::util::make_optional(std::filesystem::path{path});
    }

    // FIXME: Information about blob_dir & model_dir isn't avaialble
    // from convert<Network>::decode function.
    clarifyPathForAllNetworks(cfg);

    std::string global_device = "NPU";
    if (node["device_name"]) {
        global_device = node["device_name"].as<std::string>();
    }
    clarifyDeviceForAllNetworks(cfg.multi_inferences, global_device);
    return cfg;
}
