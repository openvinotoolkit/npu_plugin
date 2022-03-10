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

//
// THROUGHPUT_STREAMS
//

struct THROUGHPUT_STREAMS final : OptionBase<THROUGHPUT_STREAMS, int64_t> {
    static StringRef key() {
        return ov::streams::num.name();
    }

    static SmallVector<StringRef> deprecatedKeys() {
        return {VPUX_CONFIG_KEY(THROUGHPUT_STREAMS), KMB_CONFIG_KEY(THROUGHPUT_STREAMS)};
    }

    static int64_t defaultValue() {
        return -1;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

int64_t getNumThroughputStreams(const Config& config, Optional<int> numNetStreams);

//
// INFERENCE_SHAVES
//

struct INFERENCE_SHAVES final : OptionBase<INFERENCE_SHAVES, int64_t> {
    static StringRef key() {
        return ov::intel_vpux::inference_shaves.name();
    }

    static SmallVector<StringRef> deprecatedKeys() {
        return {"VPUX_VPUAL_INFERENCE_SHAVES"};
    }

    static int64_t defaultValue() {
        return 0;
    }

    static void validateValue(int64_t v) {
        VPUX_THROW_UNLESS(0 <= v && v <= 16,
                          "Attempt to set invalid number of shaves for NnCore: '{0}', valid numbers are from 0 to 16",
                          v);
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// CSRAM_SIZE
//

struct CSRAM_SIZE final : OptionBase<CSRAM_SIZE, int64_t> {
    static StringRef key() {
        return ov::intel_vpux::csram_size.name();
    }

    static int64_t defaultValue() {
        return 0;
    }

    static void validateValue(int64_t v) {
        constexpr Byte MAX_CSRAM_SIZE = 1_GB;

        VPUX_THROW_UNLESS(v >= -1 && v <= MAX_CSRAM_SIZE.count(),
                          "Attempt to set invalid CSRAM size in bytes: '{0}', valid values are -1, 0 and up to 1 Gb",
                          v);
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// GRAPH_COLOR_FORMAT
//

struct GRAPH_COLOR_FORMAT final : OptionBase<GRAPH_COLOR_FORMAT, InferenceEngine::ColorFormat> {
    static StringRef key() {
        return ov::intel_vpux::graph_color_format.name();
    }

    static InferenceEngine::ColorFormat defaultValue() {
        return InferenceEngine::ColorFormat::BGR;
    }

    static InferenceEngine::ColorFormat parse(StringRef val);

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// PREPROCESSING_SHAVES
//

struct PREPROCESSING_SHAVES final : OptionBase<PREPROCESSING_SHAVES, int64_t> {
    static StringRef key() {
        return ov::intel_vpux::preprocessing_shaves.name();
    }

    static int64_t defaultValue() {
        return 4;
    }

    static void validateValue(int64_t v) {
        VPUX_THROW_UNLESS(0 <= v && v <= 16,
                          "Attempt to set invalid number of shaves for SIPP: '{0}', valid numbers are from 0 to 16", v);
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// PREPROCESSING_LPI
//

struct PREPROCESSING_LPI final : OptionBase<PREPROCESSING_LPI, int64_t> {
    static StringRef key() {
        return ov::intel_vpux::preprocessing_lpi.name();
    }

    static int64_t defaultValue() {
        return 8;
    }

    static void validateValue(int64_t v) {
        VPUX_THROW_UNLESS(0 < v && v <= 16 && isPowerOfTwo(v),
                          "Attempt to set invalid lpi value for SIPP: '{0}',  valid values are 1, 2, 4, 8, 16", v);
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// PREPROCESSING_PIPES
//

struct PREPROCESSING_PIPES final : OptionBase<PREPROCESSING_PIPES, int64_t> {
    static StringRef key() {
        return ov::intel_vpux::preprocessing_pipes.name();
    }

    static int64_t defaultValue() {
        return 1;
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// USE_M2I
//

struct USE_M2I final : OptionBase<USE_M2I, bool> {
    static StringRef key() {
        return ov::intel_vpux::use_m2i.name();
    }

    static SmallVector<StringRef> deprecatedKeys() {
        return {VPU_KMB_CONFIG_KEY(USE_M2I)};
    }

    static bool defaultValue() {
        return false;
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// USE_SHAVE_ONLY_M2I
//

struct USE_SHAVE_ONLY_M2I final : OptionBase<USE_SHAVE_ONLY_M2I, bool> {
    static StringRef key() {
        return ov::intel_vpux::use_shave_only_m2i.name();
    }

    static SmallVector<StringRef> deprecatedKeys() {
        return {VPU_KMB_CONFIG_KEY(USE_SHAVE_ONLY_M2I)};
    }

    static bool defaultValue() {
        return false;
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// USE_SIPP
//

struct USE_SIPP final : OptionBase<USE_SIPP, bool> {
    static StringRef key() {
        return ov::intel_vpux::use_sipp.name();
    }

    static SmallVector<StringRef> deprecatedKeys() {
        return {VPU_KMB_CONFIG_KEY(USE_SIPP)};
    }

    static bool defaultValue() {
        return true;
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// EXECUTOR_STREAMS
//

struct EXECUTOR_STREAMS final : OptionBase<EXECUTOR_STREAMS, int64_t> {
    static StringRef key() {
        return ov::intel_vpux::executor_streams.name();
    }

    static SmallVector<StringRef> deprecatedKeys() {
        return {VPU_KMB_CONFIG_KEY(EXECUTOR_STREAMS)};
    }

    static int64_t defaultValue() {
#if defined(__arm__) || defined(__aarch64__)
        return 2;
#else
        return 1;
#endif
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// INFERENCE_TIMEOUT_MS
//

struct INFERENCE_TIMEOUT_MS final : OptionBase<INFERENCE_TIMEOUT_MS, int64_t> {
    static StringRef key() {
        return ov::intel_vpux::inference_timeout.name();
    }

    static int64_t defaultValue() {
        // 5 seconds -> milliseconds
        return 5 * 1000;
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// PRINT_PROFILING
//

struct PRINT_PROFILING final : OptionBase<PRINT_PROFILING, InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg> {
    static StringRef key() {
        return ov::intel_vpux::print_profiling.name();
    }

    static InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg defaultValue() {
        return InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg::NONE;
    }

    static InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg parse(StringRef val);

    static OptionMode mode() {
        return OptionMode::RunTime;
    }

#ifdef VPUX_DEVELOPER_BUILD
    static StringRef envVar() {
        return "IE_VPUX_PRINT_PROFILING";
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

}  // namespace vpux
