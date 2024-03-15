//
// Copyright (C) 2021-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/al/config/runtime.hpp"
#include "vpux/al/config/common.hpp"

#include <string_view>

using namespace vpux;
using namespace ov::intel_vpux;

namespace {

constexpr std::string_view MODEL_PRIORITY_LEGACY_PREFIX = "MODEL_PRIORITY_";

inline bool isLegacyModelPriorityValue(const std::string& name) {
    return !name.compare(0, MODEL_PRIORITY_LEGACY_PREFIX.length(), MODEL_PRIORITY_LEGACY_PREFIX);
}

inline ov::hint::Priority legacyToCurrentModelPriorityValue(const LegacyPriority legacyPriority) {
    switch (legacyPriority) {
    case LegacyPriority::LOW:
        return ov::hint::Priority::LOW;
    case LegacyPriority::MEDIUM:
        return ov::hint::Priority::MEDIUM;
    case LegacyPriority::HIGH:
        return ov::hint::Priority::HIGH;
    default:
        OPENVINO_THROW("Unsupported model priority value");
    }
}

}  // namespace

//
// register
//

void vpux::registerRunTimeOptions(OptionsDesc& desc) {
    desc.add<EXCLUSIVE_ASYNC_REQUESTS>();
    desc.add<PRINT_PROFILING>();
    desc.add<PROFILING_OUTPUT_FILE>();
    desc.add<PROFILING_TYPE>();
    desc.add<MODEL_PRIORITY>();
    desc.add<CREATE_EXECUTOR>();
    desc.add<NUM_STREAMS>();
    desc.add<ENABLE_CPU_PINNING>();
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

std::string_view InferenceEngine::VPUXConfigParams::stringifyEnum(
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

InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg vpux::PRINT_PROFILING::parse(std::string_view val) {
    const auto extractProfilingString = [](ov::intel_vpux::ProfilingOutputTypeArg prof) -> std::string {
        return profiling_output_file(prof).second.as<std::string>();
    };

    if (val == extractProfilingString(ov::intel_vpux::ProfilingOutputTypeArg::NONE)) {
        return ov::intel_vpux::ProfilingOutputTypeArg::NONE;
    } else if (val == extractProfilingString(ov::intel_vpux::ProfilingOutputTypeArg::TEXT)) {
        return ov::intel_vpux::ProfilingOutputTypeArg::TEXT;
    } else if (val == extractProfilingString(ov::intel_vpux::ProfilingOutputTypeArg::JSON)) {
        return ov::intel_vpux::ProfilingOutputTypeArg::JSON;
    }

    VPUX_THROW("Value '{0}' is not a valid PRINT_PROFILING option", val);
}

std::string vpux::PRINT_PROFILING::toString(const InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg& val) {
    std::stringstream strStream;
    if (val == InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg::NONE) {
        strStream << "NONE";
    } else if (val == InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg::TEXT) {
        strStream << "TEXT";
    } else if (val == InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg::JSON) {
        strStream << "JSON";
    } else {
        OPENVINO_THROW("No valid string for current PRINT_PROFILING option");
    }

    return strStream.str();
}

//
// PROFILING_TYPE
//

std::string_view InferenceEngine::VPUXConfigParams::stringifyEnum(
        InferenceEngine::VPUXConfigParams::ProfilingType val) {
    switch (val) {
    case InferenceEngine::VPUXConfigParams::ProfilingType::MODEL:
        return "MODEL";
    case InferenceEngine::VPUXConfigParams::ProfilingType::INFER:
        return "INFER";
    default:
        return "<UNKNOWN>";
    }
}

InferenceEngine::VPUXConfigParams::ProfilingType vpux::PROFILING_TYPE::parse(std::string_view val) {
    const auto extractProfilingString = [](ov::intel_vpux::ProfilingType prof) -> std::string {
        return profiling_type(prof).second.as<std::string>();
    };

    if (val == extractProfilingString(ov::intel_vpux::ProfilingType::MODEL)) {
        return ov::intel_vpux::ProfilingType::MODEL;
    } else if (val == extractProfilingString(ov::intel_vpux::ProfilingType::INFER)) {
        return ov::intel_vpux::ProfilingType::INFER;
    }

    VPUX_THROW("Value '{0}' is not a valid PROFILING_TYPE option", val);
}

std::string vpux::PROFILING_TYPE::toString(const InferenceEngine::VPUXConfigParams::ProfilingType& val) {
    std::stringstream strStream;
    if (val == InferenceEngine::VPUXConfigParams::ProfilingType::MODEL) {
        strStream << "MODEL";
    } else if (val == InferenceEngine::VPUXConfigParams::ProfilingType::INFER) {
        strStream << "INFER";
    } else {
        OPENVINO_THROW("No valid string for current PROFILING_TYPE option");
    }

    return strStream.str();
}

//
// MODEL_PRIORITY
//

ov::hint::Priority vpux::MODEL_PRIORITY::parse(std::string_view val) {
    std::istringstream stringStream = std::istringstream(std::string(val));
    ov::hint::Priority priority;

    if (!isLegacyModelPriorityValue(stringStream.str())) {
        stringStream >> priority;
    } else {
        LegacyPriority legacyPriority;

        stringStream >> legacyPriority;
        priority = legacyToCurrentModelPriorityValue(legacyPriority);
    }

    return priority;
}

std::string vpux::MODEL_PRIORITY::toString(const ov::hint::Priority& val) {
    std::ostringstream stringStream;

    stringStream << val;

    return stringStream.str();
}

//
// NUM_STREAMS
//

const ov::streams::Num vpux::NUM_STREAMS::defVal = ov::streams::Num(1);

ov::streams::Num vpux::NUM_STREAMS::parse(std::string_view val) {
    std::istringstream stringStream = std::istringstream(std::string(val));
    ov::streams::Num numberOfStreams;

    stringStream >> numberOfStreams;

    return numberOfStreams;
}

std::string vpux::NUM_STREAMS::toString(const ov::streams::Num& val) {
    std::ostringstream stringStream;

    stringStream << val;

    return stringStream.str();
}
