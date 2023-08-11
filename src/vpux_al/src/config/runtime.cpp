//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
    desc.add<PRINT_PROFILING>();
    desc.add<PROFILING_OUTPUT_FILE>();
    desc.add<MODEL_PRIORITY>();
    desc.add<CREATE_EXECUTOR>();
    desc.add<NUM_STREAMS>();
}

// Heuristically obtained number. Varies depending on the values of PLATFORM and PERFORMANCE_HINT
// Note: this is the value provided by the plugin, application should query and consider it, but may supply its own
// preference for number of parallel requests via dedicated configuration
int64_t vpux::getOptimalNumberOfInferRequestsInParallel(const Config& config) {
    switch (config.get<PLATFORM>()) {
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720: {
        if (config.get<PERFORMANCE_HINT>() == ov::hint::PerformanceMode::THROUGHPUT) {
            return 4;
        } else {
            return 1;
        }
    }
    default:
        if (config.get<PERFORMANCE_HINT>() == ov::hint::PerformanceMode::THROUGHPUT) {
            return 8;
        } else {
            return 1;
        }
    }
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

//
// MODEL_PRIORITY
//

ov::hint::Priority vpux::MODEL_PRIORITY::parse(StringRef val) {
    if (val == CONFIG_VALUE(MODEL_PRIORITY_LOW)) {
        return ov::hint::Priority::LOW;
    } else if (val == CONFIG_VALUE(MODEL_PRIORITY_MED)) {
        return ov::hint::Priority::MEDIUM;
    } else if (val == CONFIG_VALUE(MODEL_PRIORITY_HIGH)) {
        return ov::hint::Priority::HIGH;
    }

    VPUX_THROW("Value '{0}' is not a valid MODEL_PRIORITY option", val);
}

std::string vpux::MODEL_PRIORITY::toString(const ov::hint::Priority& val) {
    std::stringstream strStream;
    if (val == ov::hint::Priority::LOW) {
        strStream << "MODEL_PRIORITY_LOW";
    } else if (val == ov::hint::Priority::MEDIUM) {
        strStream << "MODEL_PRIORITY_MED";
    } else if (val == ov::hint::Priority::HIGH) {
        strStream << "MODEL_PRIORITY_HIGH";
    } else {
        VPUX_THROW("No valid string for current MODEL_PRIORITY option");
    }

    return strStream.str();
}

//
// NUM_STREAMS
//
const ov::streams::Num vpux::NUM_STREAMS::defVal = ov::streams::Num(1);

ov::streams::Num vpux::NUM_STREAMS::parse(StringRef val) {
    if (val == "AUTO") {
        return ov::streams::AUTO;
    } else if (val == "NUMA") {
        return ov::streams::NUMA;
    } else {
        try {
            return ov::streams::Num(std::stoi(val.str()));
        } catch (...) {
            VPUX_THROW("Value '{0}' is not a valid NUM_STREAMS option", val);
        }
    }
}

std::string vpux::NUM_STREAMS::toString(const ov::streams::Num& val) {
    std::stringstream strStream;
    if (val == ov::streams::AUTO) {
        strStream << "AUTO";
    } else if (val == ov::streams::NUMA) {
        strStream << "NUMA";
    } else {
        try {
            strStream << val.num;
        } catch (...) {
            VPUX_THROW("No valid string for current NUM_STREAMS option");
        }
    }

    return strStream.str();
}
