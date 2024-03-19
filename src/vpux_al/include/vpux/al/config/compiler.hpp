//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/IE/config.hpp"

#include "common.hpp"
#include "vpux/properties.hpp"
#include "vpux_private_properties.hpp"

namespace ov {

namespace intel_vpux {

std::string_view stringifyEnum(CompilerType val);
std::string_view stringifyEnum(ElfCompilerBackend val);

}  // namespace intel_vpux

}  // namespace ov

namespace vpux {

//
// register
//

void registerCompilerOptions(OptionsDesc& desc);

//
// COMPILER_TYPE
//

struct COMPILER_TYPE final : OptionBase<COMPILER_TYPE, InferenceEngine::VPUXConfigParams::CompilerType> {
    static std::string_view key() {
        return ov::intel_vpux::compiler_type.name();
    }

    static constexpr std::string_view getTypeName() {
        return "InferenceEngine::VPUXConfigParams::CompilerType";
    }

#ifdef VPUX_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_COMPILER_TYPE";
    }
#endif

    static InferenceEngine::VPUXConfigParams::CompilerType defaultValue() {
#if defined(VPUX_DEVELOPER_BUILD) && defined(ENABLE_MLIR_COMPILER)
        return InferenceEngine::VPUXConfigParams::CompilerType::MLIR;
#else
        return InferenceEngine::VPUXConfigParams::CompilerType::DRIVER;
#endif
    }

    static InferenceEngine::VPUXConfigParams::CompilerType parse(std::string_view val);

    static std::string toString(const InferenceEngine::VPUXConfigParams::CompilerType& val);

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
    static std::string_view key() {
        return ov::intel_vpux::compilation_mode.name();
    }

#ifdef VPUX_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_COMPILATION_MODE";
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
    static std::string_view key() {
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
    static std::string_view key() {
        return ov::intel_vpux::dpu_groups.name();
    }

    static std::vector<std::string_view> deprecatedKeys() {
        return {};
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

#ifdef VPUX_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_DPU_GROUPS";
    }
#endif
};

//
// DMA_ENGINES
//

struct DMA_ENGINES final : OptionBase<DMA_ENGINES, int64_t> {
    static std::string_view key() {
        return ov::intel_vpux::dma_engines.name();
    }

    static std::vector<std::string_view> deprecatedKeys() {
        return {};
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

#ifdef VPUX_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_DMA_ENGINES";
    }
#endif
};

//
// USE_ELF_COMPILER_BACKEND
//

struct USE_ELF_COMPILER_BACKEND final :
        OptionBase<USE_ELF_COMPILER_BACKEND, InferenceEngine::VPUXConfigParams::ElfCompilerBackend> {
    static std::string_view key() {
        return ov::intel_vpux::use_elf_compiler_backend.name();
    }

    static constexpr std::string_view getTypeName() {
        return "InferenceEngine::VPUXConfigParams::ElfCompilerBackend";
    }

#ifdef VPUX_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_USE_ELF_COMPILER_BACKEND";
    }
#endif

    static InferenceEngine::VPUXConfigParams::ElfCompilerBackend defaultValue() {
        return InferenceEngine::VPUXConfigParams::ElfCompilerBackend::AUTO;
    }

    static InferenceEngine::VPUXConfigParams::ElfCompilerBackend parse(std::string_view val);

    static std::string toString(const InferenceEngine::VPUXConfigParams::ElfCompilerBackend& val);
};

}  // namespace vpux
