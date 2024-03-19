//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <future>
#include <iostream>

#include <gflags/gflags.h>

#include "parser/parser.hpp"
#include "scenario/scenario_graph.hpp"
#include "simulation/performance_mode.hpp"
#include "simulation/reference_mode.hpp"
#include "simulation/validation_mode.hpp"

#include "utils/error.hpp"
#include "utils/logger.hpp"

static constexpr char help_message[] = "Optional. Print the usage message.";
static constexpr char cfg_message[] = "Path to the configuration file.";
static constexpr char pipeline_message[] = "Optional. Enable pipelined execution.";
static constexpr char drop_message[] = "Optional. Drop frames if they come earlier than pipeline is completed.";
static constexpr char api1_message[] = "Optional. Use legacy Inference Engine API.";
static constexpr char mode_message[] = "Optional. Simulation mode: performance (default), reference, validation.";
static constexpr char niter_message[] = "Optional. Number of iterations. If specified overwrites termination criterion"
                                        " for all streams in configuration file.";
static constexpr char exec_time_message[] = "Optional. Time in seconds. If specified overwrites termination criterion"
                                            " for all streams in configuration file.";
static constexpr char inference_only_message[] =
        "Optional. Run only inference execution for every model excluding i/o data transfer."
        " Applicable only for \"performance\" mode. (default: true).";

DEFINE_bool(h, false, help_message);
DEFINE_string(cfg, "", cfg_message);
DEFINE_bool(pipeline, false, pipeline_message);
DEFINE_bool(drop_frames, false, drop_message);
DEFINE_bool(ov_api_1_0, false, api1_message);
DEFINE_string(mode, "performance", mode_message);
DEFINE_uint64(niter, 0, niter_message);
DEFINE_uint64(t, 0, exec_time_message);
DEFINE_bool(inference_only, true, inference_only_message);

static void showUsage() {
    std::cout << "protopipe [OPTIONS]" << std::endl;
    std::cout << std::endl;
    std::cout << " Common options:            " << std::endl;
    std::cout << "    -h                      " << help_message << std::endl;
    std::cout << "    -cfg <value>            " << cfg_message << std::endl;
    std::cout << "    -pipeline               " << pipeline_message << std::endl;
    std::cout << "    -drop_frames            " << drop_message << std::endl;
    std::cout << "    -ov_api_1_0             " << api1_message << std::endl;
    std::cout << "    -mode <value>           " << mode_message << std::endl;
    std::cout << "    -niter <value>          " << niter_message << std::endl;
    std::cout << "    -t <value>              " << exec_time_message << std::endl;
    std::cout << "    -inference_only         " << inference_only_message << std::endl;
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
    std::cout << "    Config file:             " << FLAGS_cfg << std::endl;
    std::cout << "    Pipelining is enabled:   " << std::boolalpha << FLAGS_pipeline << std::endl;
    std::cout << "    Use old OpenVINO API:    " << std::boolalpha << FLAGS_ov_api_1_0 << std::endl;
    std::cout << "    Simulation mode:         " << FLAGS_mode << std::endl;
    std::cout << "    Inference only:          " << std::boolalpha << FLAGS_inference_only << std::endl;
    return true;
}

static ICompiled::Ptr compileSimulation(ISimulation::Ptr simulation, const bool pipelined, const bool drop_frames,
                                        cv::GCompileArgs&& compile_args) {
    LOG_INFO() << "Compile simulation" << std::endl;
    if (pipelined) {
        return simulation->compilePipelined(drop_frames, std::move(compile_args));
    }
    return simulation->compileSync(drop_frames, std::move(compile_args));
};

class ThreadRunner {
public:
    using F = std::function<void()>;
    void add(F&& func) {
        m_funcs.push_back(std::move(func));
    }
    void run();

private:
    std::vector<F> m_funcs;
};

void ThreadRunner::run() {
    std::vector<std::future<void>> futures;
    futures.reserve(m_funcs.size());
    for (auto&& func : m_funcs) {
        futures.push_back(std::async(std::launch::async, std::move(func)));
    }
    for (auto& future : futures) {
        future.get();
    };
};

class Task {
public:
    Task(ICompiled::Ptr&& compiled, StreamDesc&& desc);
    void operator()();
    const Result& result() const {
        return m_result;
    }
    const std::string& name() const {
        return m_desc.name;
    }

private:
    StreamDesc m_desc;
    ICompiled::Ptr m_compiled;
    Result m_result;
};

Task::Task(ICompiled::Ptr&& compiled, StreamDesc&& desc): m_desc(std::move(desc)), m_compiled(std::move(compiled)) {
}

void Task::operator()() {
    try {
        m_result = m_compiled->run(m_desc.criterion);
    } catch (const std::exception& e) {
        m_result = Error{e.what()};
    }
}

int main(int argc, char* argv[]) {
    // NB: Intentionally wrapped into try-catch to display exceptions occur on windows.
    try {
        if (!parseCommandLine(&argc, &argv)) {
            return 0;
        }

        auto parser = std::make_shared<ScenarioParser>(FLAGS_cfg, FLAGS_ov_api_1_0);
        auto scenarios = parser->parseScenarios();

        ITermCriterion::Ptr global_criterion;
        if (FLAGS_niter != 0u) {
            LOG_INFO() << "Termination criterion of " << FLAGS_niter << " iteration(s) will be used for all streams"
                       << std::endl;
            global_criterion = std::make_shared<Iterations>(FLAGS_niter);
        }
        if (FLAGS_t != 0u) {
            if (global_criterion) {
                // TODO: In fact, it make sense to have them both enabled.
                THROW_ERROR("-niter and -t options can't be specified together!");
            }
            LOG_INFO() << "Termination criterion of " << FLAGS_t << " second(s) will be used for all streams"
                       << std::endl;
            // NB: TimeOut accepts microseconds
            global_criterion = std::make_shared<TimeOut>(FLAGS_t * 1'000'000);
        }

        for (auto&& scenario : scenarios) {
            ThreadRunner runner;
            std::vector<Task> tasks;
            tasks.reserve(scenario.streams.size());
            for (auto&& stream : scenario.streams) {
                if (global_criterion) {
                    if (stream.criterion) {
                        LOG_INFO() << "Stream: " << stream.name
                                   << " termination criterion is overwritten by CLI parameter" << std::endl;
                    }
                    stream.criterion = global_criterion->clone();
                }

                ISimulation::Ptr simulation = nullptr;
                if (FLAGS_mode == "performance") {
                    PerformanceSimulation::Options opts{FLAGS_inference_only, std::move(stream.target_latency)};
                    simulation = std::make_shared<PerformanceSimulation>(std::move(stream.graph), opts);
                } else if (FLAGS_mode == "reference") {
                    simulation = std::make_shared<CalcRefSimulation>(std::move(stream.graph));
                } else if (FLAGS_mode == "validation") {
                    ValSimulation::Options opts{stream.per_iter_outputs_path};
                    simulation = std::make_shared<ValSimulation>(std::move(stream.graph), opts);
                } else {
                    throw std::logic_error("Unsupported simulation mode: " + FLAGS_mode);
                }
                ASSERT(simulation != nullptr);

                auto compiled = compileSimulation(simulation, FLAGS_pipeline, FLAGS_drop_frames,
                                                  std::move(stream.compile_args));
                tasks.emplace_back(std::move(compiled), std::move(stream));
                runner.add(std::ref(tasks.back()));
            }

            LOG_INFO() << "Run " << tasks.size() << " stream(s) asynchronously" << std::endl;
            runner.run();
            LOG_INFO() << "Execution has finished" << std::endl;

            bool any_stream_failed = false;
            for (const auto& task : tasks) {
                if (!task.result()) {
                    any_stream_failed = true;
                }
                std::cout << "stream " << task.name() << ": " << task.result().str() << std::endl;
            }
            if (any_stream_failed) {
                return EXIT_FAILURE;
            }
        }
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        throw;
    } catch (...) {
        std::cout << "Unknown error" << std::endl;
        throw;
    }
    return 0;
}
