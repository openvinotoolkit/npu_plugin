//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "reference_mode.hpp"

#include <fstream>

#include "simulation/computation.hpp"
#include "simulation/executor.hpp"
#include "simulation/layers_data.hpp"
#include "utils/logger.hpp"
#include "utils/utils.hpp"

BuildStrategy::ConstructInfo ReferenceStrategy::build(const InferDesc& infer) {
    // NB: Relevant when input reference data already exists and need to
    // generate exactly the same amount of output data.
    // Note that this value must be the same for all models within stream.
    cv::util::optional<uint64_t> model_required_iterations;

    // NB: Entities below must correspond to layers amount.
    std::vector<IDataProvider::Ptr> providers(infer.input_layers.size());
    std::vector<Meta> in_meta_vec(infer.input_layers.size());
    std::vector<Meta> out_meta_vec(infer.output_layers.size());

    switch (infer.input_data.index()) {
    case LayerVariantAttr<std::string>::index_of<std::string>(): {
        // NB: Single path provided - either single file or directory.
        std::filesystem::path path = cv::util::get<std::string>(infer.input_data);
        if (std::filesystem::exists(path)) {
            // NB: Provided path exists - upload input data from there.
            LOG_INFO() << "Input data path: " << path << " for model: " << infer.tag
                       << " exists - data will be uploaded" << std::endl;
            auto layers_data = uploadData(path, infer.tag, infer.input_layers, LayersType::INPUT);
            // NB: The Number of iterations for every layer is ALWAYS the same.
            model_required_iterations = cv::util::make_optional(layers_data.begin()->second.size());
            providers = createConstantProviders(std::move(layers_data), extractLayerNames(infer.input_layers));
        } else {
            // NB: Provided path doesn't exist - generate data and dump.
            LOG_INFO() << "Input data path: " << path << " for model: " << infer.tag
                       << " doesn't exist - input data will be generated and dumped" << std::endl;
            std::vector<std::filesystem::path> dump_path_vec;
            if (isDirectory(path)) {
                // NB: When the directory is provided, the number of input iterations to be generated aren't
                // bounded so the "random" providers will generate input data on every iteration that will
                // be dumped on the disk afterwards.
                dump_path_vec = createDirectoryLayout(path, extractLayerNames(infer.input_layers));
            } else {
                // NB: When the single file is provided, the execution must be limited to perform
                // only 1 iteration.
                model_required_iterations = cv::util::optional<uint64_t>(1ul);
                if (infer.input_layers.size() > 1) {
                    THROW_ERROR("Model: " << infer.tag
                                          << " must have exactly one input layer in order to dump input data to file: "
                                          << path);
                }
                // NB: In case directories in that path don't exist.
                std::filesystem::create_directories(path.parent_path());
                dump_path_vec = {path};
            }
            providers = createRandomProviders(infer.input_layers, infer.generators);
            for (uint32_t i = 0; i < infer.input_layers.size(); ++i) {
                in_meta_vec[i].set(Dump{dump_path_vec[i]});
            }
        }
        break;
    }
    case LayerVariantAttr<std::string>::index_of<cv::util::monostate>(): {
        THROW_ERROR("Reference mode requires input data path to be provided"
                    " in form of either directory or single file!");
    }
    }  // switch

    switch (infer.output_data.index()) {
    case LayerVariantAttr<std::string>::index_of<std::string>(): {
        // NB: Single path provided - either single file or directory.
        std::filesystem::path path = cv::util::get<std::string>(infer.output_data);
        // NB: It doesn't matter if path exist or not - regenerate and dump outputs anyway.
        std::vector<std::filesystem::path> dump_path_vec;
        if (isDirectory(path)) {
            dump_path_vec = createDirectoryLayout(path, extractLayerNames(infer.output_layers));
        } else {
            if (infer.output_layers.size() > 1) {
                THROW_ERROR("Model: " << infer.tag
                                      << " must have exactly one output layer in order to dump output data to file: "
                                      << path);
            }
            dump_path_vec = {path};
        }
        for (uint32_t i = 0; i < infer.output_layers.size(); ++i) {
            const auto& layer = infer.output_layers[i];
            out_meta_vec[i].set(Dump{dump_path_vec[i]});
        }
        break;
    }
    case LayerVariantAttr<std::string>::index_of<cv::util::monostate>(): {
        THROW_ERROR("Reference mode requires output data path to be provided"
                    " in form of either directory or single file!");
    }
    }  // switch

    // NB: Check if there is required number iterations for current model
    // and fail if it's different comparing to other models in stream.
    if (model_required_iterations.has_value()) {
        const uint64_t required_iters_value = model_required_iterations.value();
        LOG_INFO() << "Model: " << infer.tag << " will perform at most " << required_iters_value << " iteration(s)"
                   << std::endl;
        if (!required_num_iterations.has_value()) {
            required_num_iterations = model_required_iterations;
        } else {
            if (required_iters_value != required_num_iterations.value()) {
                THROW_ERROR("All models in stream are required to have the same number of iterations!");
            }
        }
    }

    return {std::move(providers), std::move(in_meta_vec), std::move(out_meta_vec)};
}

namespace {

void updateCriterion(ITermCriterion::Ptr* criterion, cv::util::optional<uint64_t> required_num_iterations) {
    if (required_num_iterations.has_value()) {
        if (*criterion) {
            // NB: Limit user's termination criterion to perfom at most m_required_num_iterations
            *criterion = std::make_shared<CombinedCriterion>(
                    *criterion, std::make_shared<Iterations>(required_num_iterations.value()));
        } else {
            *criterion = std::make_shared<Iterations>(required_num_iterations.value());
        }
    }
}

void dumpIterOutput(const cv::Mat& mat, const Dump& dump, const size_t iter) {
    auto dump_path = dump.path;
    if (isDirectory(dump.path)) {
        std::stringstream ss;
        ss << "iter_" << iter << ".bin";
        dump_path = dump_path / ss.str();
    }
    utils::writeToBinFile(dump_path.string(), mat);
};

class SyncSimulation : public SyncCompiled {
public:
    SyncSimulation(cv::GCompiled&& compiled, std::vector<DummySource::Ptr>&& sources, std::vector<Meta>&& out_meta,
                   cv::util::optional<uint64_t> required_num_iterations);

    Result run(ITermCriterion::Ptr criterion) override;

private:
    bool process(cv::GCompiled& pipeline);

    SyncExecutor m_exec;
    std::vector<DummySource::Ptr> m_sources;
    std::vector<Meta> m_out_meta;
    std::vector<cv::Mat> m_out_mats;
    size_t m_iter_idx;
    cv::optional<uint64_t> m_required_num_iterations;
};

class PipelinedSimulation : public PipelinedCompiled {
public:
    PipelinedSimulation(cv::GStreamingCompiled&& compiled, std::vector<DummySource::Ptr>&& sources,
                        std::vector<Meta>&& out_meta, cv::util::optional<uint64_t> required_num_iterations);

    Result run(ITermCriterion::Ptr criterion) override;

private:
    bool process(cv::GStreamingCompiled& pipeline);

    PipelinedExecutor m_exec;
    std::vector<DummySource::Ptr> m_sources;
    std::vector<Meta> m_out_meta;
    std::vector<cv::optional<cv::Mat>> m_opt_mats;
    size_t m_iter_idx;
    cv::optional<uint64_t> m_required_num_iterations;
};

//////////////////////////////// SyncSimulation ///////////////////////////////
SyncSimulation::SyncSimulation(cv::GCompiled&& compiled, std::vector<DummySource::Ptr>&& sources,
                               std::vector<Meta>&& out_meta, cv::util::optional<uint64_t> required_num_iterations)
        : m_exec(std::move(compiled)),
          m_sources(std::move(sources)),
          m_out_meta(std::move(out_meta)),
          m_out_mats(m_out_meta.size()),
          m_iter_idx(0u),
          m_required_num_iterations(required_num_iterations) {
}

Result SyncSimulation::run(ITermCriterion::Ptr criterion) {
    for (auto src : m_sources) {
        src->reset();
    }
    using namespace std::placeholders;
    auto cb = std::bind(&SyncSimulation::process, this, _1);
    updateCriterion(&criterion, m_required_num_iterations);
    m_exec.runLoop(cb, criterion);
    std::stringstream ss;
    ss << "Reference data has been generated for " << m_iter_idx << " iteration(s)";
    return Success{ss.str()};
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
    for (size_t i = 0; i < m_out_mats.size(); ++i) {
        if (m_out_meta[i].has<Dump>()) {
            const auto& dump = m_out_meta[i].get<Dump>();
            dumpIterOutput(m_out_mats[i], dump, m_iter_idx);
        }
    }
    ++m_iter_idx;
    return true;
}

//////////////////////////////// PipelinedSimulation ///////////////////////////////
PipelinedSimulation::PipelinedSimulation(cv::GStreamingCompiled&& compiled, std::vector<DummySource::Ptr>&& sources,
                                         std::vector<Meta>&& out_meta,
                                         cv::util::optional<uint64_t> required_num_iterations)
        : m_exec(std::move(compiled)),
          m_sources(std::move(sources)),
          m_out_meta(std::move(out_meta)),
          m_opt_mats(m_out_meta.size()),
          m_iter_idx(0u),
          m_required_num_iterations(required_num_iterations) {
}

Result PipelinedSimulation::run(ITermCriterion::Ptr criterion) {
    auto pipeline_inputs = cv::gin();
    for (auto source : m_sources) {
        pipeline_inputs += cv::gin(static_cast<cv::gapi::wip::IStreamSource::Ptr>(source));
    }
    using namespace std::placeholders;
    auto cb = std::bind(&PipelinedSimulation::process, this, _1);
    updateCriterion(&criterion, m_required_num_iterations);
    m_exec.runLoop(std::move(pipeline_inputs), cb, criterion);
    std::stringstream ss;
    ss << "Reference data has been generated for " << m_iter_idx << " iteration(s)";
    return Success{ss.str()};
};

bool PipelinedSimulation::process(cv::GStreamingCompiled& pipeline) {
    cv::GOptRunArgsP pipeline_outputs;
    for (auto& opt_mat : m_opt_mats) {
        pipeline_outputs.emplace_back(cv::gout(opt_mat)[0]);
    }
    const bool has_data = pipeline.pull(std::move(pipeline_outputs));
    for (size_t i = 0; i < m_out_meta.size(); ++i) {
        if (m_out_meta[i].has<Dump>()) {
            const auto& dump = m_out_meta[i].get<Dump>();
            ASSERT(m_opt_mats[i].has_value());
            dumpIterOutput(m_opt_mats[i].value(), dump, m_iter_idx);
        }
    }
    ++m_iter_idx;
    return has_data;
}

}  // anonymous namespace

CalcRefSimulation::CalcRefSimulation(ScenarioGraph&& graph)
        : m_strategy(std::make_shared<ReferenceStrategy>()),
          m_comp(Computation::build(std::move(graph), m_strategy, false /* add perf meta */)) {
}

std::shared_ptr<PipelinedCompiled> CalcRefSimulation::compilePipelined(const bool drop_frames,
                                                                       cv::GCompileArgs&& compile_args) {
    auto info = m_comp.compilePipelined(drop_frames, std::move(compile_args));
    return std::make_shared<PipelinedSimulation>(std::move(info.compiled), std::move(info.sources),
                                                 std::move(info.out_meta), m_strategy->required_num_iterations);
}

std::shared_ptr<SyncCompiled> CalcRefSimulation::compileSync(const bool drop_frames, cv::GCompileArgs&& compile_args) {
    auto info = m_comp.compileSync(drop_frames, std::move(compile_args));
    return std::make_shared<SyncSimulation>(std::move(info.compiled), std::move(info.sources), std::move(info.out_meta),
                                            m_strategy->required_num_iterations);
}
