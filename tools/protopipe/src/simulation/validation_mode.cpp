//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "simulation/validation_mode.hpp"
#include "layers_data.hpp"
#include "scenario/accuracy_metrics.hpp"
#include "simulation/computation.hpp"
#include "simulation/executor.hpp"
#include "utils/logger.hpp"
#include "utils/utils.hpp"

class LayerValidator {
public:
    LayerValidator(const std::string& tag, const std::string& layer_name, IAccuracyMetric::Ptr metric);
    Result operator()(const cv::Mat& lhs, const cv::Mat& rhs);

private:
    std::string m_tag;
    std::string m_layer_name;
    IAccuracyMetric::Ptr m_metric;
};

LayerValidator::LayerValidator(const std::string& tag, const std::string& layer_name, IAccuracyMetric::Ptr metric)
        : m_tag(tag), m_layer_name(layer_name), m_metric(metric) {
}

Result LayerValidator::operator()(const cv::Mat& lhs, const cv::Mat& rhs) {
    auto result = m_metric->compare(lhs, rhs);
    if (!result) {
        std::stringstream ss;
        ss << "Model: " << m_tag << ", Layer: " << m_layer_name << ", Metric: " << m_metric->str()
           << ", Reason: " << result.str() << ";";
        return Error{ss.str()};
    }
    return Success{"Passed"};
}

namespace {

class ValidationStrategy : public BuildStrategy {
public:
    explicit ValidationStrategy(const cv::util::optional<std::filesystem::path>& _per_iter_outputs_path)
            : per_iter_outputs_path(_per_iter_outputs_path) {
    }

    BuildStrategy::ConstructInfo build(const InferDesc& infer) override {
        const auto& input_data = infer.input_data;
        std::vector<IDataProvider::Ptr> providers;
        switch (input_data.index()) {
        case LayerVariantAttr<std::string>::index_of<std::string>(): {
            std::filesystem::path path = cv::util::get<std::string>(input_data);
            LOG_INFO() << "Input data path: " << path << " for model: " << infer.tag
                       << " exists - data will be uploaded" << std::endl;
            auto layers_data = uploadData(path, infer.tag, infer.input_layers, LayersType::INPUT);
            providers = createConstantProviders(std::move(layers_data), extractLayerNames(infer.input_layers));
            break;
        }
        // TODO: In fact it's not mandatory, extend in future.
        default:
            THROW_ERROR("Validation mode requires input data path to be provided"
                        " in form of either directory or single file!");
            break;
        }

        const auto& output_data = infer.output_data;
        std::vector<Meta> output_meta(infer.output_layers.size(), Meta{});
        switch (output_data.index()) {
        case LayerVariantAttr<std::string>::index_of<std::string>(): {
            std::filesystem::path path = cv::util::get<std::string>(output_data);
            LOG_INFO() << "Reference output data path: " << path << " for model: " << infer.tag
                       << " exists - data will be uploaded" << std::endl;
            auto layers_data = uploadData(path, infer.tag, infer.output_layers, LayersType::OUTPUT);
            for (uint32_t i = 0; i < infer.output_layers.size(); ++i) {
                const auto& layer = infer.output_layers[i];
                LayerValidator validator{infer.tag, layer.name, infer.metrics.at(layer.name)};
                output_meta[i].set(Validate{std::move(validator), layers_data.at(layer.name)});
            }
            break;
        }
        // TODO: In fact it's not mandatory, extend in future.
        default:
            THROW_ERROR("Validation mode requires reference output data path to be provided"
                        " in form of either directory or single file!");
            break;
        }

        if (per_iter_outputs_path.has_value()) {
            auto model_dir = per_iter_outputs_path.value() / infer.tag;
            // NB: Remove the data from the previous run if such exist
            LOG_INFO() << "Actual output data for model: " << infer.tag
                       << " will be dumped and replaced at path: " << model_dir << std::endl;
            std::filesystem::remove_all(model_dir);
            auto dump_path_vec = createDirectoryLayout(model_dir, extractLayerNames(infer.output_layers));
            for (uint32_t i = 0; i < infer.output_layers.size(); ++i) {
                output_meta[i].set(Dump{dump_path_vec[i]});
            }
        }

        // NB: No special input meta for this mode.
        std::vector<Meta> input_meta(infer.input_layers.size(), Meta{});
        return {std::move(providers), std::move(input_meta), std::move(output_meta)};
    }

    cv::util::optional<std::filesystem::path> per_iter_outputs_path;
};

struct FailedIter {
    size_t iter_idx;
    std::vector<std::string> reasons;
};

static Result reportValidationResult(const std::vector<FailedIter>& failed_iters, const size_t total_iters) {
    std::stringstream ss;
    if (!failed_iters.empty()) {
        const auto kItersToShow = 10u;
        const auto kLimit = failed_iters.size() < kItersToShow ? failed_iters.size() : kItersToShow;
        ss << "Accuraccy check failed on " << failed_iters.size() << " iteration(s)"
           << " (first " << kLimit << "):";
        ss << "\n";
        for (uint32_t i = 0; i < kLimit; ++i) {
            ss << "Iteration " << failed_iters[i].iter_idx << ":\n";
            for (const auto& reason : failed_iters[i].reasons) {
                ss << "  " << reason << "\n";
            }
        }
        return Error{ss.str()};
    }
    ss << "Validation has passed for " << total_iters << " iteration(s)";
    return Success{ss.str()};
}

static std::vector<std::string> validateOutputs(const std::vector<cv::Mat>& out_mats, const std::vector<Meta>& out_meta,
                                                const size_t iter_idx) {
    std::vector<std::string> failed_list;
    for (size_t i = 0; i < out_mats.size(); ++i) {
        if (out_meta[i].has<Validate>()) {
            const auto& val = out_meta[i].get<Validate>();
            const auto& refvec = val.reference;
            ASSERT(!refvec.empty());
            const auto& refmat = refvec[iter_idx % refvec.size()];
            auto result = val.validator(refmat, out_mats[i]);
            if (!result) {
                failed_list.push_back(std::move(result.str()));
            }
        }
    }
    return failed_list;
}

static void dumpOutputs(const std::vector<cv::Mat>& out_mats, const std::vector<Meta>& out_meta,
                        const size_t iter_idx) {
    for (size_t i = 0; i < out_mats.size(); ++i) {
        if (out_meta[i].has<Dump>()) {
            std::stringstream ss;
            ss << "iter_" << iter_idx << ".bin";
            auto dump_path = out_meta[i].get<Dump>().path / ss.str();
            utils::writeToBinFile(dump_path.string(), out_mats[i]);
        }
    }
}

class SyncSimulation : public SyncCompiled {
public:
    SyncSimulation(cv::GCompiled&& compiled, std::vector<DummySource::Ptr>&& sources, std::vector<Meta>&& out_meta);

    Result run(ITermCriterion::Ptr criterion) override;

private:
    bool process(cv::GCompiled& pipeline);

    SyncExecutor m_exec;
    std::vector<DummySource::Ptr> m_sources;
    std::vector<Meta> m_out_meta;
    std::vector<cv::Mat> m_out_mats;
    size_t m_iter_idx;
    std::vector<FailedIter> m_failed_iters;
};

class PipelinedSimulation : public PipelinedCompiled {
public:
    PipelinedSimulation(cv::GStreamingCompiled&& compiled, std::vector<DummySource::Ptr>&& sources,
                        std::vector<Meta>&& out_meta);

    Result run(ITermCriterion::Ptr criterion) override;

private:
    bool process(cv::GStreamingCompiled& pipeline);

    PipelinedExecutor m_exec;
    std::vector<DummySource::Ptr> m_sources;
    std::vector<Meta> m_out_meta;
    std::vector<cv::optional<cv::Mat>> m_opt_mats;
    size_t m_iter_idx;
    std::vector<FailedIter> m_failed_iters;
};

//////////////////////////////// SyncSimulation ///////////////////////////////
SyncSimulation::SyncSimulation(cv::GCompiled&& compiled, std::vector<DummySource::Ptr>&& sources,
                               std::vector<Meta>&& out_meta)
        : m_exec(std::move(compiled)),
          m_sources(std::move(sources)),
          m_out_meta(std::move(out_meta)),
          m_out_mats(m_out_meta.size()),
          m_iter_idx(0u) {
}

Result SyncSimulation::run(ITermCriterion::Ptr criterion) {
    for (auto src : m_sources) {
        src->reset();
    }
    using namespace std::placeholders;
    auto cb = std::bind(&SyncSimulation::process, this, _1);
    m_exec.runLoop(cb, criterion);
    return reportValidationResult(m_failed_iters, m_iter_idx);
};

bool SyncSimulation::process(cv::GCompiled& pipeline) {
    auto pipeline_outputs = cv::gout();
    // NB: Reference is mandatory there since copying empty
    // Mat may lead to weird side effects.
    for (auto& out_mat : m_out_mats) {
        pipeline_outputs += cv::gout(out_mat);
    }
    cv::GRunArgs pipeline_inputs;
    pipeline_inputs.reserve(m_sources.size());
    for (auto src : m_sources) {
        cv::gapi::wip::Data data;
        src->pull(data);
        pipeline_inputs.push_back(std::move(data));
    }
    pipeline(std::move(pipeline_inputs), std::move(pipeline_outputs));

    dumpOutputs(m_out_mats, m_out_meta, m_iter_idx);
    auto failed_list = validateOutputs(m_out_mats, m_out_meta, m_iter_idx);
    if (!failed_list.empty()) {
        m_failed_iters.push_back(FailedIter{m_iter_idx, std::move(failed_list)});
    }
    ++m_iter_idx;
    return true;
}

//////////////////////////////// PipelinedSimulation ///////////////////////////////
PipelinedSimulation::PipelinedSimulation(cv::GStreamingCompiled&& compiled, std::vector<DummySource::Ptr>&& sources,
                                         std::vector<Meta>&& out_meta)
        : m_exec(std::move(compiled)),
          m_sources(std::move(sources)),
          m_out_meta(std::move(out_meta)),
          m_opt_mats(m_out_meta.size()),
          m_iter_idx(0u) {
}

Result PipelinedSimulation::run(ITermCriterion::Ptr criterion) {
    auto pipeline_inputs = cv::gin();
    for (auto source : m_sources) {
        pipeline_inputs += cv::gin(static_cast<cv::gapi::wip::IStreamSource::Ptr>(source));
    }
    using namespace std::placeholders;
    auto cb = std::bind(&PipelinedSimulation::process, this, _1);
    m_exec.runLoop(std::move(pipeline_inputs), cb, criterion);
    return reportValidationResult(m_failed_iters, m_iter_idx);
};

bool PipelinedSimulation::process(cv::GStreamingCompiled& pipeline) {
    cv::GOptRunArgsP pipeline_outputs;
    for (auto& opt_mat : m_opt_mats) {
        pipeline_outputs.emplace_back(cv::gout(opt_mat)[0]);
    }
    const bool has_data = pipeline.pull(std::move(pipeline_outputs));
    std::vector<cv::Mat> out_mats;
    out_mats.reserve(m_opt_mats.size());
    for (auto opt_mat : m_opt_mats) {
        ASSERT(opt_mat.has_value());
        out_mats.push_back(opt_mat.value());
    }

    dumpOutputs(out_mats, m_out_meta, m_iter_idx);
    auto failed_list = validateOutputs(out_mats, m_out_meta, m_iter_idx);
    if (!failed_list.empty()) {
        m_failed_iters.push_back(FailedIter{m_iter_idx, std::move(failed_list)});
    }
    ++m_iter_idx;
    return has_data;
}

}  // anonymous namespace

ValSimulation::ValSimulation(ScenarioGraph&& graph, const ValSimulation::Options& opts)
        : m_opts(opts),
          m_comp(Computation::build(std::move(graph), std::make_shared<ValidationStrategy>(opts.per_iter_outputs_path),
                                    false)) {
}

std::shared_ptr<PipelinedCompiled> ValSimulation::compilePipelined(const bool drop_frames,
                                                                   cv::GCompileArgs&& compile_args) {
    auto info = m_comp.compilePipelined(drop_frames, std::move(compile_args));
    return std::make_shared<PipelinedSimulation>(std::move(info.compiled), std::move(info.sources),
                                                 std::move(info.out_meta));
}

std::shared_ptr<SyncCompiled> ValSimulation::compileSync(const bool drop_frames, cv::GCompileArgs&& compile_args) {
    auto info = m_comp.compileSync(drop_frames, std::move(compile_args));
    return std::make_shared<SyncSimulation>(std::move(info.compiled), std::move(info.sources),
                                            std::move(info.out_meta));
}
