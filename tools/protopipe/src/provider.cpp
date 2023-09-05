//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "provider.hpp"
#include "stream_simulation.hpp"

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

ScenarioProvider::ScenarioProvider(const std::string& filepath, const bool use_ov_old_api)
        : m_use_ov_old_api(use_ov_old_api), m_config(Config::parseFromYAML(filepath)) {
}

template <typename Params>
Params createParams(const std::string& tag, const Path& path, const std::string& device) {
    // NB: No default ctor;
    std::unique_ptr<Params> params;
    if (cv::util::holds_alternative<ModelPath>(path)) {
        const auto& model_path = cv::util::get<ModelPath>(path);
        params = std::make_unique<Params>(tag, model_path.model, model_path.bin, device);
    } else {
        GAPI_Assert(cv::util::holds_alternative<BlobPath>(path));
        const auto& blob = cv::util::get<BlobPath>(path);
        params = std::make_unique<Params>(tag, blob.path, device);
    }
    return *params;
}

cv::gapi::GNetPackage ScenarioProvider::createIEParams(const std::string& tag, const Network& network) {
    using P = cv::gapi::ie::Params<cv::gapi::Generic>;
    auto params = createParams<P>(tag, network.path, m_config.device);

    auto plugin_config = network.config;
    if (m_config.device == "NPU") {
        plugin_config["MODEL_PRIORITY"] = toPriority(network.priority);
        plugin_config["NPU_COMPILER_TYPE"] = m_config.compiler_type;
    }
    params.pluginConfig(plugin_config);
    // NB: For model in blob format ip/op have been already specified.
    params.cfgInferMode(cv::gapi::ie::InferMode::Sync);
    params.cfgNumRequests(1);
    // NB: Pre/Post processing can be configured only for Model case.
    if (cv::util::holds_alternative<ModelPath>(network.path)) {
        if (cv::util::holds_alternative<int>(network.output_precision)) {
            params.cfgOutputPrecision(cv::util::get<int>(network.output_precision));
        } else if (cv::util::holds_alternative<AttrMap<int>>(network.output_precision)) {
            const auto& map = cv::util::get<AttrMap<int>>(network.output_precision);
            params.cfgOutputPrecision({map.begin(), map.end()});
        }

        if (cv::util::holds_alternative<std::string>(network.input_layout)) {
            params.cfgInputLayout(cv::util::get<std::string>(network.input_layout));
        } else if (cv::util::holds_alternative<AttrMap<std::string>>(network.input_layout)) {
            params.cfgInputLayout(cv::util::get<AttrMap<std::string>>(network.input_layout));
        }

        if (cv::util::holds_alternative<std::string>(network.output_layout)) {
            params.cfgOutputLayout(cv::util::get<std::string>(network.output_layout));
        } else if (cv::util::holds_alternative<AttrMap<std::string>>(network.output_layout)) {
            params.cfgOutputLayout(cv::util::get<AttrMap<std::string>>(network.output_layout));
        }
    }
    return cv::gapi::networks(params);
}

cv::gapi::GNetPackage ScenarioProvider::createOVParams(const std::string& tag, const Network& network) {
    using P = cv::gapi::ov::Params<cv::gapi::Generic>;
    auto params = createParams<P>(tag, network.path, m_config.device);

    auto plugin_config = network.config;
    if (m_config.device == "NPU") {
        plugin_config["MODEL_PRIORITY"] = toPriority(network.priority);
        plugin_config["NPU_COMPILER_TYPE"] = m_config.compiler_type;
    }
    params.cfgPluginConfig(plugin_config);
    // NB: For model in blob format ip/op have been already specified.
    params.cfgNumRequests(1);

    // NB: Pre/Post processing can be configured only for Model case.
    if (cv::util::holds_alternative<ModelPath>(network.path)) {
        if (cv::util::holds_alternative<int>(network.output_precision)) {
            params.cfgOutputTensorPrecision(cv::util::get<int>(network.output_precision));
        } else if (cv::util::holds_alternative<AttrMap<int>>(network.output_precision)) {
            params.cfgOutputTensorPrecision(cv::util::get<AttrMap<int>>(network.output_precision));
        }

        if (cv::util::holds_alternative<std::string>(network.input_layout)) {
            params.cfgInputTensorLayout(cv::util::get<std::string>(network.input_layout));
        } else if (cv::util::holds_alternative<AttrMap<std::string>>(network.input_layout)) {
            params.cfgInputTensorLayout(cv::util::get<AttrMap<std::string>>(network.input_layout));
        }

        if (cv::util::holds_alternative<std::string>(network.output_layout)) {
            params.cfgOutputTensorLayout(cv::util::get<std::string>(network.output_layout));
        } else if (cv::util::holds_alternative<AttrMap<std::string>>(network.output_layout)) {
            params.cfgOutputTensorLayout(cv::util::get<AttrMap<std::string>>(network.output_layout));
        }

        if (cv::util::holds_alternative<std::string>(network.input_model_layout)) {
            params.cfgInputModelLayout(cv::util::get<std::string>(network.input_model_layout));
        } else if (cv::util::holds_alternative<AttrMap<std::string>>(network.input_model_layout)) {
            params.cfgInputModelLayout(cv::util::get<AttrMap<std::string>>(network.input_model_layout));
        }

        if (cv::util::holds_alternative<std::string>(network.output_model_layout)) {
            params.cfgOutputModelLayout(cv::util::get<std::string>(network.output_model_layout));
        } else if (cv::util::holds_alternative<AttrMap<std::string>>(network.output_model_layout)) {
            params.cfgOutputModelLayout(cv::util::get<AttrMap<std::string>>(network.output_model_layout));
        }
    }
    return cv::gapi::networks(params);
}

cv::gapi::GNetPackage ScenarioProvider::createInferenceParams(const std::string& tag, const Network& network) {
    return m_use_ov_old_api ? createIEParams(tag, network) : createOVParams(tag, network);
}

static cv::Mat createFromBinFile(const std::string& filename, const std::vector<int>& dims, const int prec) {
    cv::Mat mat;
    utils::createNDMat(mat, dims, prec);
    utils::readFromBinFile(filename, mat);
    return mat;
}

static PrePostProccesingInfo extractPrePostProcessingInfo(const Network& network) {
    return {network.input_precision, network.output_precision,   network.input_layout,
            network.output_layout,   network.input_model_layout, network.output_model_layout};
}

static InOutLayers readInOutLayers(ILayersReader::Ptr reader, const Network& network, const std::string& device) {
    InOutLayers layers;
    if (cv::util::holds_alternative<ModelPath>(network.path)) {
        const auto& model_path = cv::util::get<ModelPath>(network.path);
        layers = reader->readFromModel(model_path.model, model_path.bin, extractPrePostProcessingInfo(network));
    } else {
        GAPI_Assert(cv::util::holds_alternative<BlobPath>(network.path));
        const auto& blob = cv::util::get<BlobPath>(network.path);
        layers = reader->readFromBlob(blob.path, device, network.config);
    }
    return layers;
}

static std::map<std::string, cv::Mat> collectLayersData(const LayerVariantAttr<std::string>& filename,
                                                        const std::vector<LayerInfo>& layers,
                                                        const std::string& attrname) {
    std::map<std::string, cv::Mat> layers_data;
    auto filename_map = unpackLayerAttr(filename, extractLayerNames(layers), attrname);
    for (const auto& layer : layers) {
        const auto it = filename_map.find(layer.name);
        if (it != filename_map.end()) {
            layers_data.emplace(layer.name, createFromBinFile(it->second, layer.dims, layer.prec));
        }
    }
    return layers_data;
}

static std::map<std::string, cv::Mat> collectInputLayersData(const LayerVariantAttr<std::string>& filename,
                                                             const std::vector<LayerInfo>& layers) {
    auto inputs_data = collectLayersData(filename, layers, "input data");
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

static std::map<std::string, cv::optional<cv::Mat>> collectReferenceData(const LayerVariantAttr<std::string>& filename,
                                                                         const std::vector<LayerInfo>& layers) {
    std::map<std::string, cv::optional<cv::Mat>> ref_data;
    const auto outputs_data = collectLayersData(filename, layers, "output data");
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
std::vector<Scenario> ScenarioProvider::createScenarios() {
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
                auto [in_layers, out_layers] = readInOutLayers(reader, network, m_config.device);
                // NB: Use stem as network unique "tag".
                std::string tag = tg_mngr.add(network.stem);
                builder.addInference(tag, extractLayerNames(in_layers), extractLayerNames(out_layers));

                auto inputs_data = collectInputLayersData(network.input_data, in_layers);
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

                auto ref_data = collectReferenceData(network.output_data, out_layers);
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
                    std::make_shared<StreamSimulation>(builder.build(), num_outputs, std::move(validation_info));
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
