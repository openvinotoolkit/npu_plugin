//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/IE/config.hpp"

#include "vpux_private_properties.hpp"

#include <openvino/runtime/internal_properties.hpp>
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
    static std::string_view key() {
        return ov::hint::performance_mode.name();
    }

    static constexpr std::string_view getTypeName() {
        return "ov::hint::PerformanceMode";
    }

    static ov::hint::PerformanceMode defaultValue() {
        return ov::hint::PerformanceMode::LATENCY;
    }

    static ov::hint::PerformanceMode parse(std::string_view val);
};

//
// PERFORMANCE_HINT_NUM_REQUESTS
//

struct PERFORMANCE_HINT_NUM_REQUESTS final : OptionBase<PERFORMANCE_HINT_NUM_REQUESTS, uint32_t> {
    static std::string_view key() {
        return ov::hint::num_requests.name();
    }

    /**
     * @brief Returns configuration value if it is valid, otherwise throws
     * @details This is the same function as "InferenceEngine::PerfHintsConfig::CheckPerformanceHintRequestValue",
     * slightly modified as to not rely on the legacy API anymore.
     * @param configuration value as string
     * @return configuration value as number
     */
    static uint32_t parse(std::string_view val) {
        int val_i = -1;
        try {
            val_i = std::stoi(val.data());
            if (val_i >= 0)
                return val_i;
            else
                throw std::logic_error("wrong val");
        } catch (const std::exception&) {
            OPENVINO_THROW("Wrong value of ", val.data(), " for property key ", ov::hint::num_requests.name(),
                           ". Expected only positive integer numbers");
        }
    }

    static uint32_t defaultValue() {
        // Default value depends on PERFORMANCE_HINT, see getOptimalNumberOfInferRequestsInParallel
        // 1 corresponds to LATENCY, UNDEFINED and default mode (hints not specified)
        return 1u;
    }
};

//
// PERF_COUNT
//

struct PERF_COUNT final : OptionBase<PERF_COUNT, bool> {
    static std::string_view key() {
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
    static std::string_view key() {
        return ov::log::level.name();
    }

    static std::string_view envVar() {
        return "OV_NPU_LOG_LEVEL";
    }

    static LogLevel defaultValue() {
        return LogLevel::None;
    }
};

//
// PLATFORM
//

struct PLATFORM final : OptionBase<PLATFORM, InferenceEngine::VPUXConfigParams::VPUXPlatform> {
    static std::string_view key() {
        return ov::intel_vpux::platform.name();
    }

    static constexpr std::string_view getTypeName() {
        return "InferenceEngine::VPUXConfigParams::VPUXPlatform";
    }

    static InferenceEngine::VPUXConfigParams::VPUXPlatform defaultValue() {
        return ov::intel_vpux::VPUXPlatform::AUTO_DETECT;
    }

#ifdef VPUX_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_PLATFORM";
    }
#endif

    static InferenceEngine::VPUXConfigParams::VPUXPlatform parse(std::string_view val);

    static std::string toString(const InferenceEngine::VPUXConfigParams::VPUXPlatform& val);

    static bool isPublic() {
        return false;
    }
};

//
// DEVICE_ID
//

struct DEVICE_ID final : OptionBase<DEVICE_ID, std::string> {
    static std::string_view key() {
        return ov::device::id.name();
    }

    static std::string defaultValue() {
        return {};
    }
};

//
// CACHE_DIR
//

struct CACHE_DIR final : OptionBase<CACHE_DIR, std::string> {
    static std::string_view key() {
        return ov::cache_dir.name();
    }

    static std::string defaultValue() {
        return {};
    }
};

//
// CACHING PROPERTIES
//

struct CACHING_PROPERTIES final : OptionBase<CACHING_PROPERTIES, std::string> {
    static std::string_view key() {
        return ov::internal::caching_properties.name();
    }

    static std::string defaultValue() {
        return {};
    }
};

//
// INTERNAL SUPPORTED PROPERTIES
//

struct INTERNAL_SUPPORTED_PROPERTIES final : OptionBase<INTERNAL_SUPPORTED_PROPERTIES, std::string> {
    static std::string_view key() {
        return ov::internal::supported_properties.name();
    }

    static std::string defaultValue() {
        return {};
    }
};

}  // namespace vpux

namespace ov {
namespace intel_vpux {

std::string_view stringifyEnum(VPUXPlatform val);

}  // namespace intel_vpux
}  // namespace ov

namespace ov {
namespace hint {

std::string_view stringifyEnum(PerformanceMode val);

}  // namespace hint
}  // namespace ov
