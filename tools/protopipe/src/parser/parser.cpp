// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "parser/parser.hpp"

#include <filesystem>
#include <stack>
#include <string>
#include <vector>

// FIXME: Ideally parser shouldn't know about G-API Params.
#include <opencv2/gapi/infer/ie.hpp>  // ie::Params
#include <opencv2/gapi/infer/ov.hpp>  // ov::Params

#include "scenario/accuracy_metrics.hpp"
#include "scenario/criterion.hpp"
#include "scenario/ovhelper.hpp"

#include "utils/error.hpp"
#include "utils/logger.hpp"

// NB: Handles duplicating tags.
class TagsManager {
public:
    std::string add(const std::string& tag);

private:
    std::unordered_multiset<std::string> m_tags;
};

std::string TagsManager::add(const std::string& tag) {
    std::string t = tag;
    m_tags.insert(t);
    const auto c = m_tags.count(t);
    if (c > 1) {
        t += "-" + std::to_string(c);
    }
    return t;
}

static std::string toPriority(const std::string& priority) {
    if (priority == "LOW") {
        return "LOW";
    }
    if (priority == "NORMAL") {
        return "MEDIUM";
    }
    if (priority == "HIGH") {
        return "HIGH";
    }
    throw std::logic_error("Unsupported model priority: " + priority);
}

static LogLevel toLogLevel(const std::string& lvl) {
    if (lvl == "NONE")
        return LogLevel::None;
    if (lvl == "INFO")
        return LogLevel::Info;
    if (lvl == "DEBUG")
        return LogLevel::Debug;
    THROW_ERROR("Unsupported log level: " << lvl);
}

ScenarioParser::ScenarioParser(const std::string& filepath, const bool use_ov_old_api)
        : m_use_ov_old_api(use_ov_old_api), m_config(ScenarioConfig::parseFromYAML(filepath)) {
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

cv::gapi::GNetPackage ScenarioParser::createIEParams(const std::string& tag, const Network& network) {
    using P = cv::gapi::ie::Params<cv::gapi::Generic>;
    auto params = createParams<P>(tag, network.path, network.device);

    params.pluginConfig(network.config);
    if (network.nireq == 1u) {
        params.cfgInferMode(cv::gapi::ie::InferMode::Sync);
    } else {
        params.cfgNumRequests(network.nireq);
    }
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

cv::gapi::GNetPackage ScenarioParser::createOVParams(const std::string& tag, const Network& network) {
    using P = cv::gapi::ov::Params<cv::gapi::Generic>;
    auto params = createParams<P>(tag, network.path, network.device);

    params.cfgPluginConfig(network.config);
    params.cfgNumRequests(network.nireq);

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

cv::gapi::GNetPackage ScenarioParser::createInferenceParams(const std::string& tag, const Network& network) {
    return m_use_ov_old_api ? createIEParams(tag, network) : createOVParams(tag, network);
}

static PrePostProccesingInfo extractPrePostProcessingInfo(const Network& network) {
    return {network.input_precision, network.output_precision,   network.input_layout,
            network.output_layout,   network.input_model_layout, network.output_model_layout};
}

static InOutLayers readInOutLayers(ILayersReader::Ptr reader, const Network& network) {
    InOutLayers layers;
    if (cv::util::holds_alternative<ModelPath>(network.path)) {
        const auto& model_path = cv::util::get<ModelPath>(network.path);
        layers = reader->readFromModel(model_path.model, model_path.bin, extractPrePostProcessingInfo(network));
    } else {
        GAPI_Assert(cv::util::holds_alternative<BlobPath>(network.path));
        const auto& blob = cv::util::get<BlobPath>(network.path);
        layers = reader->readFromBlob(blob.path, network.device, network.config);
    }
    return layers;
}

template <typename T>
std::map<std::string, T> unpackWithDefault(LayerVariantAttr<T>&& attr, const std::vector<std::string>& layer_names,
                                           T def_value) {
    std::map<std::string, T> result;
    if (cv::util::holds_alternative<cv::util::monostate>(attr)) {
        for (const auto& layer_name : layer_names) {
            result.emplace(layer_name, def_value);
        }
    } else if (cv::util::holds_alternative<T>(attr)) {
        auto val = cv::util::get<T>(attr);
        for (const auto& layer_name : layer_names) {
            result.emplace(layer_name, val);
        }
    } else {
        auto map = cv::util::get<AttrMap<T>>(attr);
        for (const auto& layer_name : layer_names) {
            if (auto it = map.find(layer_name); it != map.end()) {
                result.emplace(layer_name, it->second);
            } else {
                result.emplace(layer_name, def_value);
            }
        }
    }
    return result;
}

std::vector<ScenarioDesc> ScenarioParser::parseScenarios() {
    Logger::global_lvl = toLogLevel(m_config.log_level);
    auto reader = ILayersReader::create(m_use_ov_old_api);
    std::vector<ScenarioDesc> scenarios;
    for (size_t multi_inference_idx = 0; multi_inference_idx < m_config.multi_inferences.size();
         ++multi_inference_idx) {
        auto& multi_inference = m_config.multi_inferences[multi_inference_idx];
        const auto multi_inference_name = multi_inference.name.empty()
                                                  ? "multi_inference_" + std::to_string(multi_inference_idx)
                                                  : multi_inference.name;

        std::vector<StreamDesc> stream_desc;
        for (size_t stream_idx = 0; stream_idx < multi_inference.streams.size(); ++stream_idx) {
            auto& stream = multi_inference.streams[stream_idx];

            TagsManager tg_mngr;
            cv::gapi::GNetPackage networks;

            ScenarioGraph graph;
            auto src = graph.makeSource(stream.target_fps);
            std::vector<DataNode> prev_list = {src};
            for (size_t idx = 0; idx < stream.networks.size(); ++idx) {
                std::vector<DataNode> curr_outs;
                // NB: delay_in_us - applied as preproc delay for all models except the first one.
                if (idx != 0u && stream.delay_in_us != 0u) {
                    auto delay = graph.makeDelay(stream.delay_in_us);
                    for (auto prev : prev_list) {
                        graph.link(prev, delay);
                    }
                    prev_list = {delay.out()};
                }
                for (auto& network : stream.networks[idx]) {
                    if (network.device == "NPU") {
                        network.config["MODEL_PRIORITY"] = toPriority(network.priority);
                        network.config["NPU_COMPILER_TYPE"] = m_config.compiler_type;
                    }
                    std::string tag = tg_mngr.add(network.stem);
                    LOG_INFO() << "Read i/o layers for model: " << tag << std::endl;
                    auto [in_layers, out_layers] = readInOutLayers(reader, network);
                    LOG_INFO() << "Model: " << tag << " has " << in_layers.size() << " input layers and "
                               << out_layers.size() << " output layers" << std::endl;
                    networks += createInferenceParams(tag, network);

                    auto generators = unpackWithDefault(std::move(network.random_generators),
                                                        extractLayerNames(in_layers), m_config.global_random_generator);
                    auto metrics = unpackWithDefault(std::move(network.accuracy_metrics), extractLayerNames(out_layers),
                                                     m_config.global_accuracy_metric);
                    auto infer = graph.makeInfer(InferDesc{tag, in_layers, out_layers, network.input_data,
                                                           network.output_data, generators, metrics});
                    for (auto prev : prev_list) {
                        graph.link(prev, infer);
                    }
                    curr_outs.push_back(infer.out());
                }
                prev_list = std::move(curr_outs);
            }
            // NB: delay_after_stream - applied as postproc delay for the last model in stream.
            if (stream.after_stream_delay != 0u) {
                auto delay = graph.makeDelay(stream.after_stream_delay);
                for (auto prev : prev_list) {
                    graph.link(prev, delay);
                }
            }

            // NB: Take stream index as name unless name provided (perhaps not the best idea...)
            auto stream_name = stream.name.empty() ? std::to_string(stream_idx) : stream.name;

            if (stream.exec_time_in_secs.has_value() && stream.iteration_count.has_value()) {
                // TODO: In fact, it makes sense to support these two together...
                THROW_ERROR("Stream: " << stream_name
                                       << " has two termination criterions specified but only one is supported!");
            }

            ITermCriterion::Ptr criterion;
            if (stream.exec_time_in_secs.has_value()) {
                const auto exec_time_in_secs_val = stream.exec_time_in_secs.value();
                LOG_INFO() << "Stream " << stream_name << ": termination criterion is " << exec_time_in_secs_val
                           << " second(s)" << std::endl;
                criterion = std::make_shared<TimeOut>(exec_time_in_secs_val * 1'000'000);
            } else if (stream.iteration_count.has_value()) {
                const auto iteration_count_val = stream.iteration_count.value();
                LOG_INFO() << "Stream " << stream_name << ": termination criterion is " << iteration_count_val
                           << " iteration(s)" << std::endl;
                criterion = std::make_shared<Iterations>(stream.iteration_count.value());
            } else {
                LOG_INFO() << "Stream " << stream_name
                           << ": doesn't have termination criterion specified in config file" << std::endl;
            }

            if (stream.target_latency.has_value()) {
                LOG_INFO() << "Stream " << stream_name << ": has target latency " << stream.target_latency.value()
                           << "ms specified" << std::endl;
            }

            cv::util::optional<std::filesystem::path> per_iter_outputs_path;
            if (m_config.per_iter_outputs_path.has_value()) {
                std::filesystem::path root_path = m_config.per_iter_outputs_path.value();
                std::string stream_dir = "stream_" + stream_name;
                auto full_path = root_path / multi_inference_name / stream_dir;
                LOG_INFO() << "Stream " << stream_name << ": actual outputs will be dumped to " << full_path
                           << std::endl;
                per_iter_outputs_path = cv::util::make_optional(std::move(full_path));
            }

            // FIXME: GNetPackage is the only entity that leaked into parser level.
            auto compile_args = cv::compile_args(std::move(networks));
            stream_desc.push_back(StreamDesc{std::move(stream_name), std::move(graph), std::move(criterion),
                                             std::move(compile_args), std::move(stream.target_latency),
                                             std::move(per_iter_outputs_path)});
        }

        scenarios.push_back(ScenarioDesc{std::move(multi_inference_name), std::move(stream_desc)});
    }
    return scenarios;
}
