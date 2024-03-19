//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/IE/config.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/optional.hpp"

#include "vpux/properties.hpp"
#include "vpux_private_properties.hpp"

#include <openvino/runtime/internal_properties.hpp>
#include <openvino/runtime/properties.hpp>

namespace ov {

namespace intel_vpux {

std::string_view stringifyEnum(ProfilingOutputTypeArg val);
std::string_view stringifyEnum(ProfilingType val);

}  // namespace intel_vpux

}  // namespace ov

namespace vpux {

//
// register
//

void registerRunTimeOptions(OptionsDesc& desc);

//
// EXCLUSIVE_ASYNC_REQUESTS
//

struct EXCLUSIVE_ASYNC_REQUESTS final : OptionBase<EXCLUSIVE_ASYNC_REQUESTS, bool> {
    static std::string_view key() {
        return ov::internal::exclusive_async_requests.name();
    }

    static bool defaultValue() {
        return false;
    }

    static constexpr std::string_view getTypeName() {
        return "bool";
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

int64_t getOptimalNumberOfInferRequestsInParallel(const Config& config);

//
// PRINT_PROFILING
//

struct PRINT_PROFILING final : OptionBase<PRINT_PROFILING, InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg> {
    static std::string_view key() {
        return ov::intel_vpux::print_profiling.name();
    }

    static constexpr std::string_view getTypeName() {
        return "InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg";
    }

    static InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg defaultValue() {
        return ov::intel_vpux::ProfilingOutputTypeArg::NONE;
    }

    static InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg parse(std::string_view val);

    static std::string toString(const InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg& val);

    static OptionMode mode() {
        return OptionMode::RunTime;
    }

#ifdef VPUX_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_PRINT_PROFILING";
    }
#endif
};

struct PROFILING_OUTPUT_FILE final : OptionBase<PROFILING_OUTPUT_FILE, std::string> {
    static std::string_view key() {
        return ov::intel_vpux::profiling_output_file.name();
    }

    static std::string defaultValue() {
        return {};
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

///
/// PROFILING TYPE
///
struct PROFILING_TYPE final : OptionBase<PROFILING_TYPE, InferenceEngine::VPUXConfigParams::ProfilingType> {
    static std::string_view key() {
        return ov::intel_vpux::profiling_type.name();
    }

    static constexpr std::string_view getTypeName() {
        return "InferenceEngine::VPUXConfigParams::ProfilingType";
    }

    static InferenceEngine::VPUXConfigParams::ProfilingType defaultValue() {
        return ov::intel_vpux::ProfilingType::MODEL;
    }

    static InferenceEngine::VPUXConfigParams::ProfilingType parse(std::string_view val);

    static std::string toString(const InferenceEngine::VPUXConfigParams::ProfilingType& val);

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// MODEL_PRIORITY
//

struct MODEL_PRIORITY final : OptionBase<MODEL_PRIORITY, ov::hint::Priority> {
    static std::string_view key() {
        return ov::hint::model_priority.name();
    }

    static constexpr std::string_view getTypeName() {
        return "ov::hint::Priority";
    }

    static ov::hint::Priority defaultValue() {
        return ov::hint::Priority::MEDIUM;
    }

    static ov::hint::Priority parse(std::string_view val);

    static std::string toString(const ov::hint::Priority& val);

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// CREATE_EXECUTOR
//

struct CREATE_EXECUTOR final : OptionBase<CREATE_EXECUTOR, int64_t> {
    static std::string_view key() {
        return ov::intel_vpux::create_executor.name();
    }

    static int64_t defaultValue() {
        return 1;
    }

#ifdef VPUX_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_CREATE_EXECUTOR";
    }
#endif

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// NUM_STREAMS
//
struct NUM_STREAMS final : OptionBase<NUM_STREAMS, ov::streams::Num> {
    static std::string_view key() {
        return ov::num_streams.name();
    }

    static constexpr std::string_view getTypeName() {
        return "ov::streams::Num";
    }

    const static ov::streams::Num defVal;

    // The only supported number for currently supported platforms.
    // FIXME: update in the future
    static ov::streams::Num defaultValue() {
        return defVal;
    }

    static ov::streams::Num parse(std::string_view val);

    static std::string toString(const ov::streams::Num& val);

    static void validateValue(const ov::streams::Num& num) {
        if (defVal != num && ov::streams::AUTO != num) {
            throw std::runtime_error("NUM_STREAMS can not be set");
        }
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// ENABLE_CPU_PINNING
//
struct ENABLE_CPU_PINNING final : OptionBase<ENABLE_CPU_PINNING, bool> {
    static std::string_view key() {
        return ov::hint::enable_cpu_pinning.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};
}  // namespace vpux
