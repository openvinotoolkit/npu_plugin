//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/al/config/compiler.hpp"

using namespace vpux;
using namespace ov::intel_vpux;

//
// register
//

void vpux::registerCompilerOptions(OptionsDesc& desc) {
    desc.add<COMPILER_TYPE>();
    desc.add<COMPILATION_MODE>();
    desc.add<COMPILATION_MODE_PARAMS>();
    desc.add<DPU_GROUPS>();
    desc.add<DMA_ENGINES>();
    desc.add<USE_ELF_COMPILER_BACKEND>();
}

//
// COMPILER_TYPE
//

std::string_view InferenceEngine::VPUXConfigParams::stringifyEnum(InferenceEngine::VPUXConfigParams::CompilerType val) {
    switch (val) {
    case InferenceEngine::VPUXConfigParams::CompilerType::MLIR:
        return "MLIR";
    case InferenceEngine::VPUXConfigParams::CompilerType::DRIVER:
        return "DRIVER";
    default:
        return "<UNKNOWN>";
    }
}

InferenceEngine::VPUXConfigParams::CompilerType vpux::COMPILER_TYPE::parse(std::string_view val) {
    if (val == stringifyEnum(InferenceEngine::VPUXConfigParams::CompilerType::MLIR)) {
        return InferenceEngine::VPUXConfigParams::CompilerType::MLIR;
    } else if (val == stringifyEnum(InferenceEngine::VPUXConfigParams::CompilerType::DRIVER)) {
        return InferenceEngine::VPUXConfigParams::CompilerType::DRIVER;
    }

    VPUX_THROW("Value '{0}' is not a valid COMPILER_TYPE option", val);
}

std::string vpux::COMPILER_TYPE::toString(const InferenceEngine::VPUXConfigParams::CompilerType& val) {
    std::stringstream strStream;
    if (val == InferenceEngine::VPUXConfigParams::CompilerType::MLIR) {
        strStream << "MLIR";
    } else if (val == InferenceEngine::VPUXConfigParams::CompilerType::DRIVER) {
        strStream << "DRIVER";
    } else {
        OPENVINO_THROW("No valid string for current LOG_LEVEL option");
    }

    return strStream.str();
}

//
// USE_ELF_COMPILER_BACKEND
//

std::string_view InferenceEngine::VPUXConfigParams::stringifyEnum(
        InferenceEngine::VPUXConfigParams::ElfCompilerBackend val) {
    switch (val) {
    case InferenceEngine::VPUXConfigParams::ElfCompilerBackend::AUTO:
        return "AUTO";
    case InferenceEngine::VPUXConfigParams::ElfCompilerBackend::NO:
        return "NO";
    case InferenceEngine::VPUXConfigParams::ElfCompilerBackend::YES:
        return "YES";
    default:
        return "<UNKNOWN>";
    }
}

InferenceEngine::VPUXConfigParams::ElfCompilerBackend vpux::USE_ELF_COMPILER_BACKEND::parse(std::string_view val) {
    if (val == stringifyEnum(InferenceEngine::VPUXConfigParams::ElfCompilerBackend::AUTO)) {
        return InferenceEngine::VPUXConfigParams::ElfCompilerBackend::AUTO;
    } else if (val == stringifyEnum(InferenceEngine::VPUXConfigParams::ElfCompilerBackend::NO)) {
        return InferenceEngine::VPUXConfigParams::ElfCompilerBackend::NO;
    } else if (val == stringifyEnum(InferenceEngine::VPUXConfigParams::ElfCompilerBackend::YES)) {
        return InferenceEngine::VPUXConfigParams::ElfCompilerBackend::YES;
    }

    VPUX_THROW("Value '{0}' is not a valid USE_ELF_COMPILER_BACKEND option", val);
}

std::string vpux::USE_ELF_COMPILER_BACKEND::toString(const InferenceEngine::VPUXConfigParams::ElfCompilerBackend& val) {
    std::stringstream strStream;
    if (val == InferenceEngine::VPUXConfigParams::ElfCompilerBackend::AUTO) {
        strStream << "AUTO";
    } else if (val == InferenceEngine::VPUXConfigParams::ElfCompilerBackend::NO) {
        strStream << "NO";
    } else if (val == InferenceEngine::VPUXConfigParams::ElfCompilerBackend::YES) {
        strStream << "YES";
    } else {
        OPENVINO_THROW("No valid string for current USE_ELF_COMPILER_BACKEND option");
    }

    return strStream.str();
}
