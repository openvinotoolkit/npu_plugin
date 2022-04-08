//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/IE/config.hpp"

#include "vpux/vpux_plugin_config.hpp"
#include "vpux_private_config.hpp"
#include "vpux_private_properties.hpp"

#include <ie_performance_hints.hpp>
#include <ie_plugin_config.hpp>
#include <openvino/runtime/properties.hpp>

namespace vpux {

//
// register
//

void registerCommonOptions(OptionsDesc& desc);

//
// PERFORMANCE_HINT
//

struct PERFORMANCE_HINT final : OptionBase<PERFORMANCE_HINT, ov::hint::PerformanceMode> {
    static StringRef key() {
        return ov::hint::performance_mode.name();
    }

    static ov::hint::PerformanceMode defaultValue() {
        return ov::hint::PerformanceMode::UNDEFINED;
    }

    static ov::hint::PerformanceMode parse(StringRef val);
};

//
// PERFORMANCE_HINT_NUM_REQUESTS
//

struct PERFORMANCE_HINT_NUM_REQUESTS final : OptionBase<PERFORMANCE_HINT_NUM_REQUESTS, uint32_t> {
    static StringRef key() {
        return ov::hint::num_requests.name();
    }

    static uint32_t parse(StringRef val) {
        return InferenceEngine::PerfHintsConfig::CheckPerformanceHintRequestValue(val.str());
    }
};

uint32_t getPerfHintNumRequests(const Config& config);

//
// PERF_COUNT
//

struct PERF_COUNT final : OptionBase<PERF_COUNT, bool> {
    static StringRef key() {
        return ov::enable_profiling.name();
    }

    static bool defaultValue() {
        return false;
    }
};

//
// LOG_LEVEL
//

ov::log::Level cvtLogLevel(LogLevel lvl);

struct LOG_LEVEL final : OptionBase<LOG_LEVEL, LogLevel> {
    static StringRef key() {
        return ov::log::level.name();
    }

#ifdef VPUX_DEVELOPER_BUILD
    static StringRef envVar() {
        return "IE_VPUX_LOG_LEVEL";
    }
#endif

    static LogLevel defaultValue() {
        return LogLevel::None;
    }
};

//
// PLATFORM
//

struct PLATFORM final : OptionBase<PLATFORM, InferenceEngine::VPUXConfigParams::VPUXPlatform> {
    static StringRef key() {
        return ov::intel_vpux::vpux_platform.name();
    }

    static InferenceEngine::VPUXConfigParams::VPUXPlatform defaultValue() {
        return ov::intel_vpux::cvtVPUXPlatform(ov::intel_vpux::VPUXPlatform::AUTO);
    }

    static InferenceEngine::VPUXConfigParams::VPUXPlatform parse(StringRef val);

    static bool isPublic() {
        return false;
    }
};

//
// DEVICE_ID
//

struct DEVICE_ID final : OptionBase<DEVICE_ID, std::string> {
    static StringRef key() {
        return ov::device::id.name();
    }

    static std::string defaultValue() {
        return {};
    }
};

}  // namespace vpux

namespace InferenceEngine {
namespace VPUXConfigParams {

vpux::StringLiteral stringifyEnum(VPUXPlatform val);

}  // namespace VPUXConfigParams
}  // namespace InferenceEngine

namespace ov {
namespace hint {

vpux::StringLiteral stringifyEnum(PerformanceMode val);

}  // namespace hint
}  // namespace ov
