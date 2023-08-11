//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/utils/IE/config.hpp"

#include "common.hpp"
#include "vpux/properties.hpp"
#include "vpux/vpux_plugin_config.hpp"
#include "vpux_private_config.hpp"
#include "vpux_private_properties.hpp"

#include <ie_plugin_config.hpp>

namespace InferenceEngine {

namespace VPUXConfigParams {

llvm::StringLiteral stringifyEnum(InferenceEngine::VPUXConfigParams::CompilerType val);
llvm::StringLiteral stringifyEnum(InferenceEngine::VPUXConfigParams::ElfCompilerBackend val);

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
#if defined(VPUX_DEVELOPER_BUILD) && defined(ENABLE_MLIR_COMPILER)
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
    static StringRef envVar() {
        return "IE_VPUX_DPU_GROUPS";
    }
#endif
};

//
// DDR_HEAP_SIZE_MB
//

struct DDR_HEAP_SIZE_MB final : OptionBase<DDR_HEAP_SIZE_MB, int64_t> {
    static StringRef key() {
        return ov::intel_vpux::ddr_heap_size_mb.name();
    }

    static int64_t defaultValue() {
        return 500;
    }

    static bool isPublic() {
        return false;
    }
};

//
// DMA_ENGINES
//

struct DMA_ENGINES final : OptionBase<DMA_ENGINES, int64_t> {
    static StringRef key() {
        return ov::intel_vpux::dma_engines.name();
    }

    static SmallVector<StringRef> deprecatedKeys() {
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
    static StringRef envVar() {
        return "IE_VPUX_DMA_ENGINES";
    }
#endif
};

//
// VPUX_FORCE_HOST_PRECISION_LAYOUT_CONVERSION
//

struct FORCE_HOST_PRECISION_LAYOUT_CONVERSION final : OptionBase<FORCE_HOST_PRECISION_LAYOUT_CONVERSION, bool> {
    static StringRef key() {
        return ov::intel_vpux::force_host_precision_layout_conversion.name();
    }

#ifdef VPUX_DEVELOPER_BUILD
    static StringRef envVar() {
        return "IE_VPUX_VPUX_FORCE_HOST_PRECISION_LAYOUT_CONVERSION";
    }
#endif

    static bool defaultValue() {
        // HACK: This is how we distinguish CCG (Windows) from IOTG (Linux) cases for now.
#ifdef _WIN32
        return true;
#else
        return false;
#endif
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

//
// FORCE_HOST_QUANTIZATION
//

struct FORCE_HOST_QUANTIZATION final : OptionBase<FORCE_HOST_QUANTIZATION, bool> {
    static StringRef key() {
        return VPU_COMPILER_CONFIG_KEY(FORCE_HOST_QUANTIZATION);
    }

#ifdef VPUX_DEVELOPER_BUILD
    static StringRef envVar() {
        return "IE_VPUX_FORCE_HOST_QUANTIZATION";
    }
#endif

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return true;
    }
};

//
// USE_ELF_COMPILER_BACKEND
//

struct USE_ELF_COMPILER_BACKEND final :
        OptionBase<USE_ELF_COMPILER_BACKEND, InferenceEngine::VPUXConfigParams::ElfCompilerBackend> {
    static StringRef key() {
        return ov::intel_vpux::use_elf_compiler_backend.name();
    }

#ifdef VPUX_DEVELOPER_BUILD
    static StringRef envVar() {
        return "IE_VPUX_USE_ELF_COMPILER_BACKEND";
    }
#endif

    static InferenceEngine::VPUXConfigParams::ElfCompilerBackend defaultValue() {
        return InferenceEngine::VPUXConfigParams::ElfCompilerBackend::YES;
    }

    static InferenceEngine::VPUXConfigParams::ElfCompilerBackend parse(StringRef val);
};

}  // namespace vpux
