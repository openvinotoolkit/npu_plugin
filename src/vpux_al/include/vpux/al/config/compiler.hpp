//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/utils/IE/config.hpp"

#include "common.hpp"
#include "mcm_private_config.hpp"
#include "vpux/properties.hpp"
#include "vpux/vpux_plugin_config.hpp"
#include "vpux_private_config.hpp"
#include "vpux_private_properties.hpp"

#include <ie_plugin_config.hpp>

namespace InferenceEngine {

namespace VPUXConfigParams {

llvm::StringLiteral stringifyEnum(InferenceEngine::VPUXConfigParams::CompilerType val);

}  // namespace VPUXConfigParams

}  // namespace InferenceEngine

namespace vpux {

//
// register
//

void registerCompilerOptions(OptionsDesc& desc);

//
// COMPILER_TYPE
//

struct COMPILER_TYPE final : OptionBase<COMPILER_TYPE, InferenceEngine::VPUXConfigParams::CompilerType> {
    static StringRef key() {
        return ov::intel_vpux::compiler_type.name();
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
        return ov::intel_vpux::compilation_mode.name();
    }

#ifdef VPUX_DEVELOPER_BUILD
    static StringRef envVar() {
        return "IE_VPUX_COMPILATION_MODE";
    }
#endif

    static std::string defaultValue() {
        return "";
    }

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
        return ov::intel_vpux::compilation_mode_params.name();
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
        return ov::intel_vpux::dpu_groups.name();
    }

    static SmallVector<StringRef> deprecatedKeys() {
        return {VPU_COMPILER_CONFIG_KEY(NUM_CLUSTER)};
    }

    static int64_t defaultValue() {
        return -1;
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
        return ov::intel_vpux::custom_layers.name();
    }

    static std::string defaultValue() {
        return "";
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

}  // namespace vpux
