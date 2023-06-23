//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "etests_provider.hpp"
#include "etests_simulation.hpp"

#include "dummy_source.hpp"
#include "inference_builder.hpp"
#include "operations.hpp"
#include "ovhelper.hpp"

#include <opencv2/gapi/infer/ie.hpp>  // ie::Params
#include <opencv2/gapi/infer/ov.hpp>  // ov::Params

#include <filesystem>
#include <string>
#include <vector>

static std::string toPriority(const std::string& priority) {
    if (priority == "LOW")
        return "MODEL_PRIORITY_LOW";
    if (priority == "NORMAL")
        return "MODEL_PRIORITY_MED";
    if (priority == "HIGH")
        return "MODEL_PRIORITY_HIGH";
    throw std::logic_error("Unsupported model priority: " + priority);
}

ETestsProvider::ETestsProvider(const std::string& filepath, const bool use_ov_old_api)
        : m_use_ov_old_api(use_ov_old_api), m_config(ETestsConfig::parseFromYAML(filepath)) {
}

cv::gapi::GNetPackage ETestsProvider::createIEParams(const std::string& tag, const Network& network) {
    // NB: No default ctor;
    using P = cv::gapi::ie::Params<cv::gapi::Generic>;
    std::unique_ptr<P> params;

    if (cv::util::holds_alternative<Network::ModelPath>(network.path)) {
        const auto& model_path = cv::util::get<Network::ModelPath>(network.path);
        params = std::make_unique<P>(tag, model_path.model, model_path.bin, m_config.device);
    } else {
        GAPI_Assert(cv::util::holds_alternative<Network::BlobPath>(network.path));
        const auto& blob = cv::util::get<Network::BlobPath>(network.path);
        params = std::make_unique<P>(tag, blob.path, m_config.device);
    }

    auto plugin_config = network.config;
    if (m_config.device == "VPUX") {
        plugin_config["MODEL_PRIORITY"] = toPriority(network.priority);
        plugin_config["VPUX_COMPILER_TYPE"] = m_config.compiler_type;
    }
    params->pluginConfig(plugin_config);
    // NB: For model in blob format ip/op have been already specified.
    params->cfgInferMode(cv::gapi::ie::InferMode::Sync);
    params->cfgNumRequests(1);
    // NB: Output precision can be configured only for Model case.
    if (network.op != -1 && cv::util::holds_alternative<Network::ModelPath>(network.path)) {
        params->cfgOutputPrecision(network.op);
    }
    return cv::gapi::networks(*params);
}

cv::gapi::GNetPackage ETestsProvider::createOVParams(const std::string& tag, const Network& network) {
    // NB: No default ctor;
    using P = cv::gapi::ov::Params<cv::gapi::Generic>;
    std::unique_ptr<P> params;

    if (cv::util::holds_alternative<Network::ModelPath>(network.path)) {
        const auto& model_path = cv::util::get<Network::ModelPath>(network.path);
        params = std::make_unique<P>(tag, model_path.model, model_path.bin, m_config.device);
    } else {
        GAPI_Assert(cv::util::holds_alternative<Network::BlobPath>(network.path));
        const auto& blob = cv::util::get<Network::BlobPath>(network.path);
        params = std::make_unique<P>(tag, blob.path, m_config.device);
    }

    auto plugin_config = network.config;
    if (m_config.device == "VPUX") {
        plugin_config["MODEL_PRIORITY"] = toPriority(network.priority);
        plugin_config["VPUX_COMPILER_TYPE"] = m_config.compiler_type;
    }
    params->cfgPluginConfig(plugin_config);
    // NB: For model in blob format ip/op have been already specified.
    params->cfgNumRequests(1);
    // NB: Output precision can be configured only for Model case.
    if (network.op != -1 && cv::util::holds_alternative<Network::ModelPath>(network.path)) {
        params->cfgOutputTensorPrecision(network.op);
    }
    return cv::gapi::networks(*params);
}

cv::gapi::GNetPackage ETestsProvider::createInferenceParams(const std::string& tag, const Network& network) {
    return m_use_ov_old_api ? createIEParams(tag, network) : createOVParams(tag, network);
}

std::vector<std::string> extractLayerNames(const std::vector<LayerInfo>& layers) {
    std::vector<std::string> names;
    std::transform(layers.begin(), layers.end(), std::back_inserter(names), [](const auto& layer) {
        return layer.name;
    });
    return names;
}

static cv::Mat createFromBinFile(const std::string& filename, const std::vector<int>& dims, const int prec) {
    cv::Mat mat;
    utils::createNDMat(mat, dims, prec);
    utils::readFromBinFile(filename, mat);
    return mat;
}

static InOutLayers readInOutLayers(ILayersReader::Ptr reader, const Network::Path& path, const std::string& device) {
    InOutLayers layers;
    if (cv::util::holds_alternative<Network::ModelPath>(path)) {
        const auto& model_path = cv::util::get<Network::ModelPath>(path);
        layers = reader->readFromModel(model_path.model, model_path.bin);
    } else {
        GAPI_Assert(cv::util::holds_alternative<Network::BlobPath>(path));
        const auto& blob = cv::util::get<Network::BlobPath>(path);
        layers = reader->readFromBlob(blob.path, device);
    }
    return layers;
}

static std::map<std::string, std::string> mapLayerToFileNames(const Network::FileNameT& filename,
                                                              const std::vector<std::string>& layer_names) {
    using M = std::map<std::string, std::string>;
    switch (filename.index()) {
    case Network::FileNameT::index_of<cv::util::monostate>(): {
        return {};
    }
    case Network::FileNameT::index_of<std::string>(): {
        GAPI_Assert(layer_names.size() == 1u);
        return {{layer_names.front(), cv::util::get<std::string>(filename)}};
    }
    case Network::FileNameT::index_of<M>(): {
        // TODO: Check that all layers exist.
        std::unordered_map<std::string, bool> checked_names;
        for (const auto& layer_name : layer_names) {
            checked_names.emplace(layer_name, false);
        }
        const auto& filename_map = cv::util::get<M>(filename);
        for (const auto& [layer_name, filename] : filename_map) {
            const auto it = checked_names.find(layer_name);
            if (it == checked_names.end()) {
                throw std::logic_error("Layer: " + layer_name + " for file: " + filename + " is not found!");
            } else if (it->second) {
                throw std::logic_error("Layer: " + layer_name + " for file: " + filename + " is set twice!");
            } else {
                it->second = true;
            }
        }
        return filename_map;
    }
    default:
        GAPI_Assert(false);
    }
    // Unreachable code...
    GAPI_Assert(false);
}

static std::map<std::string, cv::Mat> collectLayersData(const Network::FileNameT& filename,
                                                        const std::vector<LayerInfo>& layers) {
    std::map<std::string, cv::Mat> layers_data;
    auto mapping = mapLayerToFileNames(filename, extractLayerNames(layers));
    for (const auto& layer : layers) {
        const auto it = mapping.find(layer.name);
        if (it != mapping.end()) {
            layers_data.emplace(layer.name, createFromBinFile(it->second, layer.dims, layer.prec));
        }
    }
    return layers_data;
}

static std::map<std::string, cv::Mat> collectInputLayersData(const Network::FileNameT& filename,
                                                             const std::vector<LayerInfo>& layers) {
    auto inputs_data = collectLayersData(filename, layers);
    for (const auto& layer : layers) {
        const auto it = inputs_data.find(layer.name);
        if (it == inputs_data.end()) {
            cv::Mat random_data;
            utils::createNDMat(random_data, layer.dims, layer.prec);
            utils::generateRandom(random_data);
            inputs_data.emplace(layer.name, random_data);
        }
    }
    return inputs_data;
}

static std::map<std::string, cv::optional<cv::Mat>> collectReferenceData(const Network::FileNameT& filename,
                                                                         const std::vector<LayerInfo>& layers) {
    std::map<std::string, cv::optional<cv::Mat>> ref_data;
    const auto outputs_data = collectLayersData(filename, layers);
    for (const auto& layer : layers) {
        const auto it = outputs_data.find(layer.name);
        cv::optional<cv::Mat> opt_mat;
        if (it != outputs_data.end()) {
            opt_mat = cv::util::make_optional(it->second);
        }
        ref_data.emplace(layer.name, opt_mat);
    }
    return ref_data;
}

// FIXME: Decompose this method !!!
// Separate graph construction logic from parameters and input data.
std::vector<Scenario> ETestsProvider::createScenarios() {
    auto reader = ILayersReader::create(m_use_ov_old_api);

    std::vector<Scenario> use_cases;
    for (auto&& streams : m_config.multistreams) {
        Scenario scenario;
        size_t stream_id = 0;
        for (auto&& stream : streams) {
            auto pipeline_inputs = cv::gin();

            TagsManager tg_mngr;
            cv::gapi::GNetPackage networks;

            InferenceBuilder builder;
            using LayerID = InferenceBuilder::LayerID;
            // NB: Holds output layer from the previous network in the sequence.
            // For the first network it will be its first layer (see below).
            LayerID prev_id;

            size_t network_idx = 0;
            size_t num_outputs = 0;
            ValidationMap validation_map;
            for (auto&& network : stream.networks) {
                auto [in_layers, out_layers] = readInOutLayers(reader, network.path, m_config.device);
                if (network.ip != -1) {
                    for (auto& layer : in_layers) {
                        layer.prec = network.ip;
                    }
                }

                if (network.op != -1) {
                    for (auto& layer : out_layers) {
                        layer.prec = network.op;
                    }
                }

                // NB: Use stem as network unique "tag".
                std::string tag = tg_mngr.add(network.stem);
                builder.addInference(tag, extractLayerNames(in_layers), extractLayerNames(out_layers));

                auto inputs_data = collectInputLayersData(network.input_filename, in_layers);
                for (int layer_idx = 0; layer_idx < in_layers.size(); ++layer_idx) {
                    const auto& layer = in_layers[layer_idx];
                    auto layer_data = inputs_data.at(layer.name);
                    LayerID curr_id{tag, layer.name};

                    if (layer_idx == 0) {
                        // NB: First layer of the first model connects directly to source.
                        if (network_idx == 0) {
                            // NB: 0 is special value means no limit fps for source.
                            const auto latency_in_us =
                                    stream.target_fps != 0 ? static_cast<uint32_t>(1000 * 1000 / stream.target_fps) : 0;
                            using S = cv::gapi::wip::IStreamSource::Ptr;
                            S src = std::make_shared<DummySource>(std::chrono::microseconds{latency_in_us}, layer_data);
                            builder.addGraphInput(curr_id);
                            pipeline_inputs += cv::gin(src);
                        } else {
                            // NB: Hang the delay only on the first (any) layer of the model.
                            InferenceBuilder::PP pp =
                                    std::bind(GDummy::on, std::placeholders::_1, stream.delay_in_us, layer_data);
                            builder.addConnection(prev_id, curr_id, std::move(pp));
                        }
                    } else {
                        // NB: Otherwise makes layer as constant input.
                        builder.addGraphInput(curr_id);
                        pipeline_inputs += cv::gin(layer_data);
                    }
                }

                // NB: Update prev_id.
                prev_id = LayerID{tag, out_layers.front().name};

                auto ref_data = collectReferenceData(network.output_filename, out_layers);
                for (int layer_idx = 0; layer_idx < out_layers.size(); ++layer_idx) {
                    const auto& layer = out_layers[layer_idx];
                    // NB: Map output idx to its reference.
                    validation_map.emplace(num_outputs, ReferenceInfo{ref_data.at(layer.name), tag, layer.name});
                    builder.addGraphOutput(LayerID{tag, layer.name});
                    ++num_outputs;
                }

                if (stream.after_stream_delay && network_idx == stream.networks.size() - 1) {
                    // NB: Hang the post work delay on any models's output.
                    LayerID layer_id{tag, out_layers.front().name};
                    ;
                    InferenceBuilder::PP pp = std::bind(GDummy::on, std::placeholders::_1, stream.after_stream_delay,
                                                        // NB: Dummy output won't be
                                                        // consumsed by any other operation
                                                        // and used only for delay simulation so
                                                        // it doesn't matter what it returns
                                                        cv::Mat{1, 1, CV_8U});
                    builder.addGraphOutput(LayerID{tag, out_layers.front().name}, std::move(pp));
                    ++num_outputs;
                }

                networks += createInferenceParams(tag, network);
                ++network_idx;
            }

            ITermCriterion::Ptr criterion;
            if (stream.exec_time_in_secs != 0) {
                criterion = std::make_shared<TimeOut>(stream.exec_time_in_secs * 1000000);
            } else {
                // NB: One iteration is used for warmup.
                criterion = std::make_shared<Iterations>(stream.iteration_count - 1);
            }

            ValidationInfo validation_info{m_config.save_per_iteration_output_data, stream_id,
                                           std::move(validation_map)};
            auto simulation =
                    std::make_shared<ETestsSimulation>(builder.build(), num_outputs, std::move(validation_info));
            std::shared_ptr<ExecutionProtocol> protocol(new ExecutionProtocol{
                    std::move(simulation), std::move(pipeline_inputs),
                    cv::compile_args(networks, cv::gapi::kernels<GCPUDummy>()), std::move(criterion)});

            scenario.protocols.push_back(std::move(protocol));
            ++stream_id;
        }
        use_cases.push_back(std::move(scenario));
    }

    return use_cases;
}
