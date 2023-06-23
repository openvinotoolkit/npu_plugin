#include "config.hpp"

#include <yaml-cpp/yaml.h>

#include <opencv2/gapi/own/assert.hpp>  // GAPI_Assert
#include <opencv2/opencv.hpp>           // depth

#include <filesystem>
#include <map>
#include <string>
#include <vector>

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

template <>
struct convert<Network::FileNameT> {
    static bool decode(const Node& node, Network::FileNameT& filename) {
        if (node.IsMap()) {
            filename = node.as<std::map<std::string, std::string>>();
        } else {
            filename = node.as<std::string>();
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
            network.path = Network::ModelPath{path.string(), bin_path.string()};
        } else if (path.extension() == ".blob") {
            network.path = Network::BlobPath{path.string()};
        } else if (path.extension() == ".onnx" || path.extension() == ".pdpd") {
            network.path = Network::ModelPath{path.string(), ""};
        } else {
            throw std::logic_error("Unsupported model format: " + name);
        }

        if (node["priority"]) {
            network.priority = node["priority"].as<std::string>();
        }

        if (node["config"]) {
            network.config = node["config"].as<std::map<std::string, std::string>>();
        }

        if (node["ip"]) {
            network.ip = toDepth(node["ip"].as<std::string>());
        }

        if (node["op"]) {
            network.op = toDepth(node["op"].as<std::string>());
        }

        if (node["input_data"]) {
            network.input_filename = node["input_data"].as<Network::FileNameT>();
        }

        if (node["output_data"]) {
            network.output_filename = node["output_data"].as<Network::FileNameT>();
        }

        return true;
    }
};

template <>
struct convert<Stream> {
    static bool decode(const Node& node, Stream& stream) {
        stream.networks = node["network"].as<std::vector<Network>>();
        stream.target_fps = node["target_fps"] ? node["target_fps"].as<uint32_t>() : 0;
        stream.delay_in_us = node["delay_in_us"] ? node["delay_in_us"].as<uint64_t>() : 0;

        stream.after_stream_delay = node["after_stream_delay"] ? node["after_stream_delay"].as<uint64_t>() : 0;
        stream.iteration_count = node["iteration_count"] ? node["iteration_count"].as<size_t>() : 0;
        stream.exec_time_in_secs = node["exec_time_in_secs"] ? node["exec_time_in_secs"].as<size_t>() : 0;

        // Set default exec_time to 60 seconds if both iteration count and exec time are not set
        if (stream.iteration_count == 0 && stream.exec_time_in_secs == 0) {
            stream.exec_time_in_secs = 60;
        }

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

static void clarifyNetworkPath(Network::Path& path, const std::string& blob_dir, const std::string& model_dir) {
    if (cv::util::holds_alternative<Network::ModelPath>(path)) {
        auto& model_path = cv::util::get<Network::ModelPath>(path);
        fs::path model_file_path{model_path.model};
        fs::path bin_file_path{model_path.bin};

        if (model_file_path.is_relative()) {
            model_path.model = (model_dir / model_file_path).string();
        }
        if (!model_path.bin.empty() && bin_file_path.is_relative()) {
            model_path.bin = (model_dir / bin_file_path).string();
        }
    } else {
        GAPI_Assert(cv::util::holds_alternative<Network::BlobPath>(path));
        auto& blob = cv::util::get<Network::BlobPath>(path);
        fs::path blob_path{blob.path};

        if (blob_path.is_relative()) {
            blob.path = (blob_dir / blob_path).string();
        }
    }
}

static void clarifyPathForAllNetworks(ETestsConfig& config) {
    for (auto& multi_stream : config.multistreams) {
        for (auto& stream : multi_stream) {
            for (auto& network : stream.networks) {
                clarifyNetworkPath(network.path, config.blob_dir, config.model_dir);
            }
        }
    }
}

ETestsConfig ETestsConfig::parseFromYAML(const std::string& filepath) {
    auto node = YAML::LoadFile(filepath);

    ETestsConfig cfg;
    if (node["device_name"]) {
        cfg.device = node["device_name"].as<std::string>();
    }
    if (cfg.device == "VPU") {
        cfg.device = "VPUX";
    }

    if (node["vpux_compiler_type"]) {
        cfg.compiler_type = node["vpux_compiler_type"].as<std::string>();
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

    if (!node["multi_inference"]) {
        std::stringstream ss;
        ss << "Failed to parse: " << filepath << "\nConfig must contain \"multi_inference\" key";
        throw std::logic_error(ss.str());
    }

    if (node["save_per_iteration_output_data"]) {
        cfg.save_per_iteration_output_data = node["save_per_iteration_output_data"].as<bool>();
    }

    for (auto it : node["multi_inference"]) {
        cfg.multistreams.push_back(it["input_stream_list"].as<std::vector<Stream>>());
    }
    // FIXME: Information about blob_dir & model_dir isn't avaialble
    // from convert<Network>::decode function.
    clarifyPathForAllNetworks(cfg);

    return cfg;
}
