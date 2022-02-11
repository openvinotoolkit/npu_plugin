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

#include "mcm_private_config.hpp"
#include "vpux/vpux_plugin_config.hpp"
#include "vpux_private_config.hpp"

#include <ie_plugin_config.hpp>

namespace vpux {

//
// register
//

void registerCompilerOptions(OptionsDesc& desc);

//
// COMPILER_TYPE
//

StringLiteral stringifyEnum(InferenceEngine::VPUXConfigParams::CompilerType val);

struct COMPILER_TYPE final : OptionBase<COMPILER_TYPE, InferenceEngine::VPUXConfigParams::CompilerType> {
    static StringRef key() {
        return VPUX_CONFIG_KEY(COMPILER_TYPE);
    }

#ifdef VPUX_DEVELOPER_BUILD
    static StringRef envVar() {
        return "IE_VPUX_COMPILER_TYPE";
    }
#endif

    static InferenceEngine::VPUXConfigParams::CompilerType defaultValue() {
#ifdef ENABLE_MLIR_COMPILER
        return InferenceEngine::VPUXConfigParams::CompilerType::MLIR;
#else
        return InferenceEngine::VPUXConfigParams::CompilerType::DRIVER;
#endif
    }

    static InferenceEngine::VPUXConfigParams::CompilerType parse(StringRef val);

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// COMPILATION_MODE
//

struct COMPILATION_MODE final : OptionBase<COMPILATION_MODE, std::string> {
    static StringRef key() {
        return VPUX_CONFIG_KEY(COMPILATION_MODE);
    }

#ifdef VPUX_DEVELOPER_BUILD
    static StringRef envVar() {
        return "IE_VPUX_COMPILATION_MODE";
    }
#endif

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// COMPILATION_MODE_PARAMS
//

struct COMPILATION_MODE_PARAMS final : OptionBase<COMPILATION_MODE_PARAMS, std::string> {
    static StringRef key() {
        return VPUX_CONFIG_KEY(COMPILATION_MODE_PARAMS);
    }

    static std::string defaultValue() {
        return {};
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// DPU_GROUPS
//

struct DPU_GROUPS final : OptionBase<DPU_GROUPS, int64_t> {
    static StringRef key() {
        return VPUX_CONFIG_KEY(DPU_GROUPS);
    }

    static SmallVector<StringRef> deprecatedKeys() {
        return {VPU_COMPILER_CONFIG_KEY(NUM_CLUSTER)};
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// CUSTOM_LAYERS
//

struct CUSTOM_LAYERS final : OptionBase<CUSTOM_LAYERS, std::string> {
    static StringRef key() {
        return VPU_COMPILER_CONFIG_KEY(CUSTOM_LAYERS);
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

}  // namespace vpux
