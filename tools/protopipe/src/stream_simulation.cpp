//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "stream_simulation.hpp"

#include <filesystem>

using OutVec = std::vector<cv::Mat>;
namespace fs = std::filesystem;

class SyncStream : public SyncExecutor {
public:
    SyncStream(cv::GCompiled&& compiled, const size_t num_outputs, const ValidationInfo& validation_info);

    virtual void validate() override;

protected:
    cv::GRunArgsP outputs() override;

private:
    size_t m_num_outputs;
    ValidationInfo m_validation_info;
    std::vector<OutVec> m_per_iter_outputs;
};

SyncStream::SyncStream(cv::GCompiled&& compiled, const size_t num_outputs, const ValidationInfo& validation_info)
        : SyncExecutor(std::move(compiled)), m_num_outputs(num_outputs), m_validation_info(validation_info) {
}

cv::GRunArgsP SyncStream::outputs() {
    m_per_iter_outputs.push_back(OutVec{m_num_outputs});
    auto& iter_outputs = m_per_iter_outputs.back();

    cv::GRunArgsP outs;
    for (auto&& m : iter_outputs) {
        outs += cv::gout(m);
    }
    return outs;
}

static bool compare(const cv::Mat& lhs, const cv::Mat& rhs) {
    size_t lhs_byte_size = lhs.total() * lhs.elemSize();
    size_t rhs_byte_size = rhs.total() * rhs.elemSize();

    return lhs_byte_size == rhs_byte_size && std::equal(lhs.begin<uint8_t>(), lhs.end<uint8_t>(), rhs.begin<uint8_t>());
}

static std::string normalizeLayerName(const std::string& layer_name) {
    // NB: Layer name might contain "/" which is prohibited for filename characters.
    // FIXME: In fact there much more prohibited chars for filenames e.g:
    // \, /, :, *, ?, ", <, >, |
    std::string normalized = layer_name;
    std::replace(normalized.begin(), normalized.end(), '/', '_');
    return normalized;
};

static void runValidation(const ValidationInfo& validation_info, const std::vector<OutVec>& per_iteration_outputs) {
    const auto& validation_map = validation_info.map;
    GAPI_Assert(!per_iteration_outputs.empty());
    const auto num_iters = per_iteration_outputs.size();
    const auto num_outputs = per_iteration_outputs.front().size();

    // FIXME: Apparanetly it should be moved somewhere else...
    if (validation_info.save_per_iter_outputs) {
        // FIXME: Dump only outputs that are configured to be validated.
        // Simulation can contain "service" outputs that
        // shouldn't be either dumped or validated.
        fs::path root_dir{"stream_" + std::to_string(validation_info.stream_idx)};
        for (const auto& [output_idx, refinfo] : validation_map) {
            for (size_t iter_idx = 0; iter_idx < num_iters; ++iter_idx) {
                auto curr_dir = root_dir / ("iteration_" + std::to_string(iter_idx));
                fs::create_directories(curr_dir);
                auto out_file_name = refinfo.model_name + "_" + normalizeLayerName(refinfo.layer_name) + ".bin";

                fs::path output_path = curr_dir / out_file_name;
                const auto& output_vec = per_iteration_outputs[iter_idx];
                utils::writeToBinFile(output_path.string(), output_vec[output_idx]);
            }
        }
    }

    // NB: Collect reference only for outputs that
    // are configured to be validated.
    std::map<size_t, cv::Mat> ref_map;
    for (const auto& [output_idx, info] : validation_map) {
        if (info.ref_data) {
            ref_map.emplace(output_idx, *info.ref_data);
        } else {
            ref_map.emplace(output_idx, per_iteration_outputs[0][output_idx]);
        }
    }

    std::vector<size_t> failed_iterations;
    std::vector<std::vector<ReferenceInfo>> failed_outname_per_iteration(num_iters);

    for (size_t iter_idx = 0; iter_idx < num_iters; ++iter_idx) {
        const auto& output_vec = per_iteration_outputs[iter_idx];
        // NB: If any of iterations output failed, iteration is treated as failed.
        bool is_failed = false;
        for (const auto& [output_idx, refinfo] : validation_map) {
            if (!compare(output_vec[output_idx], ref_map.at(output_idx))) {
                failed_outname_per_iteration[iter_idx].push_back(refinfo);
                if (!is_failed) {
                    failed_iterations.push_back(iter_idx);
                    is_failed = true;
                }
            }
        }
    }

    // NB: Throw with error if validation failed.
    constexpr size_t kItersToShow = 10u;
    if (failed_iterations.size() > 0) {
        std::stringstream ss;
        ss << "accuracy check failed on " << failed_iterations.size() << " iterations (first 10):";
        for (size_t i = 0; i < failed_iterations.size(); ++i) {
            ss << " " << failed_iterations[i];
            if (i == kItersToShow - 1) {
                break;
            }
        }
        // FIXME: Additional information what exactly outputs failed accuracy check
        // Apparently it shouldn't be here...
        ss << "\n";
        for (size_t i = 0; i < failed_iterations.size(); ++i) {
            const auto failed_idx = failed_iterations[i];
            ss << "Iteration " << failed_idx << " - accuracy check failed: " << std::endl;
            for (const auto& refinfo : failed_outname_per_iteration[failed_idx]) {
                ss << "  "
                   << "Model: " << refinfo.model_name << ", Layer: " << refinfo.layer_name << std::endl;
            }
            if (i == kItersToShow - 1) {
                break;
            }
        }
        throw std::logic_error(ss.str());
    }
}

void SyncStream::validate() {
    runValidation(m_validation_info, m_per_iter_outputs);
}

class PipelinedStream : public PipelinedExecutor {
public:
    PipelinedStream(cv::GStreamingCompiled&& compiled, const size_t num_outputs, const ValidationInfo& validation_info);

    virtual void validate() override;

protected:
    void postIterationCallback() override;
    cv::GOptRunArgsP outputs() override;

private:
    size_t m_num_outputs;
    ValidationInfo m_validation_info;
    std::vector<cv::optional<cv::Mat>> m_out_mats;
    std::vector<OutVec> m_per_iter_outputs;
};

PipelinedStream::PipelinedStream(cv::GStreamingCompiled&& compiled, const size_t num_outputs,
                                 const ValidationInfo& validation_info)
        : PipelinedExecutor(std::move(compiled)),
          m_num_outputs(num_outputs),
          m_validation_info(validation_info),
          m_out_mats(m_num_outputs) {
}

void PipelinedStream::postIterationCallback() {
    m_per_iter_outputs.push_back(OutVec{m_num_outputs});
    auto& iter_outputs = m_per_iter_outputs.back();

    for (size_t output_idx = 0; output_idx < m_num_outputs; ++output_idx) {
        auto opt_out = m_out_mats[output_idx];
        GAPI_Assert(opt_out.has_value());
        iter_outputs[output_idx] = opt_out->clone();
    }
}

cv::GOptRunArgsP PipelinedStream::outputs() {
    cv::GOptRunArgsP outs;
    for (auto&& m : m_out_mats) {
        // FIXME: No cv::GOptRunArgsP::operator+=
        outs.emplace_back(cv::gout(m)[0]);
    }
    return outs;
}

void PipelinedStream::validate() {
    runValidation(m_validation_info, m_per_iter_outputs);
}

StreamSimulation::StreamSimulation(Simulation::GraphBuildF&& build, const size_t num_outputs,
                                   ValidationInfo&& validation_info)
        : Simulation(std::move(build)), m_num_outputs(num_outputs), m_validation_info(std::move(validation_info)) {
}

SyncExecutor::Ptr StreamSimulation::compileSync(cv::GCompiled&& sync) {
    return std::make_shared<SyncStream>(std::move(sync), m_num_outputs, m_validation_info);
}

PipelinedExecutor::Ptr StreamSimulation::compilePipelined(cv::GStreamingCompiled&& pipelined) {
    return std::make_shared<PipelinedStream>(std::move(pipelined), m_num_outputs, m_validation_info);
}
