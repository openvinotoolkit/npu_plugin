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

StringLiteral vpux::stringifyEnum(InferenceEngine::VPUXConfigParams::CompilerType val) {
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
