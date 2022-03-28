//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/al/config/runtime.hpp"

#include "vpux/al/config/common.hpp"

using namespace vpux;
using namespace ov::intel_vpux;
using namespace InferenceEngine::VPUXConfigParams;

//
// register
//

void vpux::registerRunTimeOptions(OptionsDesc& desc) {
    desc.add<EXCLUSIVE_ASYNC_REQUESTS>();
    desc.add<THROUGHPUT_STREAMS>();
    desc.add<INFERENCE_SHAVES>();
    desc.add<CSRAM_SIZE>();
    desc.add<GRAPH_COLOR_FORMAT>();
    desc.add<PREPROCESSING_SHAVES>();
    desc.add<PREPROCESSING_LPI>();
    desc.add<PREPROCESSING_PIPES>();
    desc.add<USE_M2I>();
    desc.add<USE_SHAVE_ONLY_M2I>();
    desc.add<USE_SIPP>();
    desc.add<EXECUTOR_STREAMS>();
    desc.add<INFERENCE_TIMEOUT_MS>();
    desc.add<PRINT_PROFILING>();
    desc.add<PROFILING_OUTPUT_FILE>();
    desc.add<MODEL_PRIORITY>();
}

//
// THROUGHPUT_STREAMS
//

int64_t vpux::getNumThroughputStreams(const Config& config, Optional<int> numNetStreams) {
    if (config.has<THROUGHPUT_STREAMS>()) {
        const auto numStreams = config.get<THROUGHPUT_STREAMS>();
        if (numStreams > 0) {
            return numStreams;
        }
    }

    if (numNetStreams.hasValue()) {
        // The '+ 2' part is used to adapt the configuration to the values from E#16060 ticket.
        return numNetStreams.getValue() + 2;
    }

    // The values were taken from E#16060 ticket.
    switch (config.get<PERFORMANCE_HINT>()) {
    case ov::hint::PerformanceMode::THROUGHPUT:
        return 6;
    case ov::hint::PerformanceMode::LATENCY:
    case ov::hint::PerformanceMode::UNDEFINED:
    default:
        return 3;
    }
}

//
// GRAPH_COLOR_FORMAT
//

InferenceEngine::ColorFormat vpux::GRAPH_COLOR_FORMAT::parse(StringRef val) {
    const auto extractColorString = [](ov::intel_vpux::ColorFormat format) -> std::string {
        return graph_color_format(format).second.as<std::string>();
    };

    if (val == extractColorString(ov::intel_vpux::ColorFormat::BGR)) {
        return InferenceEngine::ColorFormat::BGR;
    } else if (val == extractColorString(ov::intel_vpux::ColorFormat::RGB)) {
        return InferenceEngine::ColorFormat::RGB;
    }

    VPUX_THROW("Value '{0}' is not a valid GRAPH_COLOR_FORMAT option", val);
}

//
// PRINT_PROFILING
//

StringLiteral InferenceEngine::VPUXConfigParams::stringifyEnum(
        InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg val) {
    switch (val) {
    case InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg::JSON:
        return "JSON";
    case InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg::TEXT:
        return "TEXT";
    case InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg::NONE:
        return "NONE";
    default:
        return "<UNKNOWN>";
    }
}

InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg vpux::PRINT_PROFILING::parse(StringRef val) {
    const auto extractProfilingString = [](ov::intel_vpux::ProfilingOutputTypeArg prof) -> std::string {
        return profiling_output_file(prof).second.as<std::string>();
    };

    if (val == extractProfilingString(ov::intel_vpux::ProfilingOutputTypeArg::NONE)) {
        return cvtProfilingOutputType(ov::intel_vpux::ProfilingOutputTypeArg::NONE);
    } else if (val == extractProfilingString(ov::intel_vpux::ProfilingOutputTypeArg::TEXT)) {
        return cvtProfilingOutputType(ov::intel_vpux::ProfilingOutputTypeArg::TEXT);
    } else if (val == extractProfilingString(ov::intel_vpux::ProfilingOutputTypeArg::JSON)) {
        return cvtProfilingOutputType(ov::intel_vpux::ProfilingOutputTypeArg::JSON);
    }

    VPUX_THROW("Value '{0}' is not a valid PRINT_PROFILING option", val);
}
