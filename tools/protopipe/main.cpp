//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <future>
#include <iostream>

#include <gflags/gflags.h>

#include "etests/etests_provider.hpp"

struct PerformanceMetrics {
    double first_latency_ms;
    double avg_latency_ms;
    double min_latency_ms;
    double max_latency_ms;
    int64_t total_frames;
    int64_t elapsed;
    double fps;
    int64_t dropped;
};

PerformanceMetrics calculateMetrics(const SimulationExecutor::Output& simout) {
    if (simout.latency.empty()) {
        throw std::logic_error("Latency vector is empty after simulation");
    }

    if (simout.seq_ids.empty()) {
        throw std::logic_error("Frame id vector is empty after simulation");
    }

    PerformanceMetrics metrics;

    metrics.first_latency_ms = simout.latency[0] / 1000.0;
    metrics.avg_latency_ms = utils::avg(simout.latency) / 1000.0;
    metrics.min_latency_ms = utils::min(simout.latency) / 1000.0;
    metrics.max_latency_ms = utils::max(simout.latency) / 1000.0;
    metrics.elapsed = static_cast<int64_t>(simout.elapsed / 1000.0);

    metrics.fps = simout.latency.size() / static_cast<double>(metrics.elapsed) * 1000;

    metrics.dropped = 0;
    int64_t prev_seq_id = simout.seq_ids[0];
    for (size_t i = 1; i < simout.seq_ids.size(); ++i) {
        metrics.dropped += simout.seq_ids[i] - prev_seq_id - 1;
        prev_seq_id = simout.seq_ids[i];
    }

    metrics.total_frames = simout.seq_ids.back() + 1;
    return metrics;
};

std::ostream& operator<<(std::ostream& os, const PerformanceMetrics& metrics) {
    os << "throughput: " << metrics.fps << " FPS, latency: min: " << metrics.min_latency_ms
       << " ms, avg: " << metrics.avg_latency_ms << " ms, max: " << metrics.max_latency_ms
       << " ms, frames dropped: " << metrics.dropped << "/" << metrics.total_frames;
    return os;
}

static constexpr char help_message[] = "Optional. Print the usage message.";
static constexpr char cfg_message[] = "Path to the configuration file.";
static constexpr char pipeline_message[] = "Optional. Enable pipelined execution.";
static constexpr char drop_message[] = "Optional. Drop frames if they come earlier than pipeline is completed.";
static constexpr char api1_message[] = "Optional. Use legacy Inference Engine API.";

DEFINE_bool(h, false, help_message);
DEFINE_string(cfg, "", cfg_message);
DEFINE_bool(pipeline, false, pipeline_message);
DEFINE_bool(drop_frames, false, drop_message);
DEFINE_bool(ov_api_1_0, false, api1_message);

static void showUsage() {
    std::cout << "protopipe [OPTIONS]" << std::endl;
    std::cout << std::endl;
    std::cout << " Common options: " << std::endl;
    std::cout << "    -h           " << help_message << std::endl;
    std::cout << "    -cfg <value> " << cfg_message << std::endl;
    std::cout << "    -pipeline    " << pipeline_message << std::endl;
    std::cout << "    -drop_frames " << drop_message << std::endl;
    std::cout << "    -ov_api_1_0  " << api1_message << std::endl;
    std::cout << std::endl;
}

bool parseCommandLine(int* argc, char*** argv) {
    gflags::ParseCommandLineNonHelpFlags(argc, argv, true);

    if (FLAGS_h) {
        showUsage();
        return false;
    }

    if (FLAGS_cfg.empty()) {
        throw std::invalid_argument("Path to config file is required");
    }

    std::cout << "Parameters:" << std::endl;
    std::cout << "    Config file:           " << FLAGS_cfg << std::endl;
    std::cout << "    Pipelining is enabled: " << std::boolalpha << FLAGS_pipeline << std::endl;
    std::cout << "    Use old OpenVINO API:  " << std::boolalpha << FLAGS_ov_api_1_0 << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    try {
        if (!parseCommandLine(&argc, &argv)) {
            return 0;
        }

        auto etests = std::make_shared<ETestsProvider>(FLAGS_cfg, FLAGS_ov_api_1_0);
        auto scenarios = etests->createScenarios();

        for (auto&& scenario : scenarios) {
            using O = SimulationExecutor::Output;
            std::vector<std::future<O>> results;
            std::vector<std::packaged_task<O()>> tasks;
            std::vector<SimulationExecutor::Ptr> executors;

            // NB: Setup simulation.
            for (auto&& proto : scenario.protocols) {
                auto&& sm = proto->simulation;
                std::shared_ptr<SimulationExecutor> exec;

                if (FLAGS_pipeline) {
                    exec = sm->compilePipelined(FLAGS_drop_frames, cv::descr_of(proto->inputs),
                                                std::move(proto->compile_args));
                } else {
                    exec = sm->compileSync(FLAGS_drop_frames, cv::descr_of(proto->inputs),
                                           std::move(proto->compile_args));
                }
                exec->setSource(std::move(proto->inputs));
                executors.push_back(std::move(exec));
            }

            // NB: Warmup executors.
            bool any_warmup_failed = false;
            for (int i = 0; i < executors.size(); ++i) {
                try {
                    executors[i]->runWarmup();
                } catch (const std::exception& e) {
                    any_warmup_failed = true;
                    std::cout << "Warmup failed: stream " << std::to_string(i) << ": " << e.what() << std::endl;
                }
            }

            if (any_warmup_failed) {
                return EXIT_FAILURE;
            }

            for (int i = 0; i < executors.size(); ++i) {
                std::packaged_task<O()> task([i, &executors, &scenario] {
                    return executors[i]->runLoop(scenario.protocols[i]->criterion);
                });
                results.emplace_back(task.get_future());
                tasks.emplace_back(std::move(task));
            }

            // NB: Run pipelines asynchronously and wait for all.
            std::vector<std::thread> threads;
            for (auto&& task : tasks) {
                threads.emplace_back(std::move(task));
            }
            for (auto&& thread : threads) {
                thread.join();
            }

            bool any_stream_failed = false;
            for (int i = 0; i < results.size(); ++i) {
                std::stringstream ss;
                ss << "stream " << std::to_string(i) << ": ";
                try {
                    ss << calculateMetrics(results[i].get());
                } catch (const std::exception& e) {
                    any_stream_failed = true;
                    ss << e.what();
                }
                std::cout << ss.str() << std::endl;
            }

            for (int i = 0; i < executors.size(); ++i) {
                try {
                    executors[i]->validate();
                } catch (const std::exception& e) {
                    any_stream_failed = true;
                    std::cout << "stream " << std::to_string(i) << ": " << e.what() << std::endl;
                }
            }

            if (any_stream_failed) {
                return EXIT_FAILURE;
            }
            // Probably make sense to log it not just dump into console...
            std::cout << "All accuracy checks are passed." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
    return 0;
}
