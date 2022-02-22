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
}

//
// GRAPH_COLOR_FORMAT
//

InferenceEngine::ColorFormat vpux::GRAPH_COLOR_FORMAT::parse(StringRef val) {
    const auto extractColorString = [](InferenceEngine::ColorFormat format) -> std::string {
        return graph_color_format(format).second.as<std::string>();
    };

    if (val == extractColorString(InferenceEngine::ColorFormat::BGR)) {
        return InferenceEngine::ColorFormat::BGR;
    } else if (val == extractColorString(InferenceEngine::ColorFormat::RGB)) {
        return InferenceEngine::ColorFormat::RGB;
    }

    VPUX_THROW("Value '{0}' is not a valid GRAPH_COLOR_FORMAT option", val);
}

//
// PRINT_PROFILING
//

ProfilingOutputTypeArg vpux::PRINT_PROFILING::parse(StringRef val) {
    const auto extractProfilingString = [](ProfilingOutputTypeArg arg) -> std::string {
        return print_profiling(arg).second.as<std::string>();
    };

    if (val == extractProfilingString(ProfilingOutputTypeArg::NONE)) {
        return ProfilingOutputTypeArg::NONE;
    } else if (val == extractProfilingString(ProfilingOutputTypeArg::TEXT)) {
        return ProfilingOutputTypeArg::TEXT;
    } else if (val == extractProfilingString(ProfilingOutputTypeArg::JSON)) {
        return ProfilingOutputTypeArg::JSON;
    }

    VPUX_THROW("Value '{0}' is not a valid PRINT_PROFILING option", val);
}
