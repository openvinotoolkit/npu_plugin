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
#include "vpux/vpux_plugin_config.hpp"
#include "vpux_private_config.hpp"
#include "vpux_private_properties.hpp"

#include <ie_plugin_config.hpp>
#include <openvino/runtime/properties.hpp>

namespace InferenceEngine {

namespace VPUXConfigParams {

llvm::StringLiteral stringifyEnum(InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg val);

}  // namespace VPUXConfigParams

}  // namespace InferenceEngine

namespace vpux {

//
// register
//

void registerRunTimeOptions(OptionsDesc& desc);

//
// EXCLUSIVE_ASYNC_REQUESTS
//

struct EXCLUSIVE_ASYNC_REQUESTS final : OptionBase<EXCLUSIVE_ASYNC_REQUESTS, bool> {
    static StringRef key() {
        return CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS);
    }

    static bool defaultValue() {
        return false;
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
    static StringRef key() {
        return ov::intel_vpux::print_profiling.name();
    }

    static InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg defaultValue() {
        return cvtProfilingOutputType(ov::intel_vpux::ProfilingOutputTypeArg::NONE);
    }

    static InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg parse(StringRef val);

    static OptionMode mode() {
        return OptionMode::RunTime;
    }

#ifdef VPUX_DEVELOPER_BUILD
    static StringRef envVar() {
        return "IE_NPU_PRINT_PROFILING";
    }
#endif
};

struct PROFILING_OUTPUT_FILE final : OptionBase<PROFILING_OUTPUT_FILE, std::string> {
    static StringRef key() {
        return ov::intel_vpux::profiling_output_file.name();
    }

    static std::string defaultValue() {
        return {};
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// MODEL_PRIORITY
//

struct MODEL_PRIORITY final : OptionBase<MODEL_PRIORITY, ov::hint::Priority> {
    static StringRef key() {
        return ov::hint::model_priority.name();
    }

    static ov::hint::Priority defaultValue() {
        return ov::hint::Priority::MEDIUM;
    }

    static ov::hint::Priority parse(StringRef val);

    static std::string toString(const ov::hint::Priority& val);

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// CREATE_EXECUTOR
//

struct CREATE_EXECUTOR final : OptionBase<CREATE_EXECUTOR, int64_t> {
    static StringRef key() {
        return ov::intel_vpux::create_executor.name();
    }

    static int64_t defaultValue() {
        return 1;
    }

#ifdef VPUX_DEVELOPER_BUILD
    static StringRef envVar() {
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
    static StringRef key() {
        return ov::num_streams.name();
    }

    const static ov::streams::Num defVal;

    // The only supported number for currently supported platforms.
    // FIXME: update in the future
    static ov::streams::Num defaultValue() {
        return defVal;
    }

    static ov::streams::Num parse(StringRef val);

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
}  // namespace vpux
