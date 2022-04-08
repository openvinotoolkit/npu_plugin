//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
    desc.add<CUSTOM_LAYERS>();
}

//
// COMPILER_TYPE
//

StringLiteral InferenceEngine::VPUXConfigParams::stringifyEnum(InferenceEngine::VPUXConfigParams::CompilerType val) {
    switch (val) {
    case InferenceEngine::VPUXConfigParams::CompilerType::MCM:
        return "MCM";
    case InferenceEngine::VPUXConfigParams::CompilerType::MLIR:
        return "MLIR";
    case InferenceEngine::VPUXConfigParams::CompilerType::DRIVER:
        return "DRIVER";
    default:
        return "<UNKNOWN>";
    }
}

InferenceEngine::VPUXConfigParams::CompilerType vpux::COMPILER_TYPE::parse(StringRef val) {
    if (val == stringifyEnum(InferenceEngine::VPUXConfigParams::CompilerType::MCM)) {
        return InferenceEngine::VPUXConfigParams::CompilerType::MCM;
    } else if (val == stringifyEnum(InferenceEngine::VPUXConfigParams::CompilerType::MLIR)) {
        return InferenceEngine::VPUXConfigParams::CompilerType::MLIR;
    } else if (val == stringifyEnum(InferenceEngine::VPUXConfigParams::CompilerType::DRIVER)) {
        return InferenceEngine::VPUXConfigParams::CompilerType::DRIVER;
    }

    VPUX_THROW("Value '{0}' is not a valid COMPILER_TYPE option", val);
}
