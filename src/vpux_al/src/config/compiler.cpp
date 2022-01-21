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
using namespace InferenceEngine::VPUXConfigParams;

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

StringLiteral vpux::stringifyEnum(CompilerType val) {
    switch (val) {
    case CompilerType::MCM:
        return "MCM";
    case CompilerType::MLIR:
        return "MLIR";
    case CompilerType::DRIVER:
        return "DRIVER";
    default:
        return "<UNKNOWN>";
    }
}

CompilerType vpux::COMPILER_TYPE::parse(StringRef val) {
    if (val == VPUX_CONFIG_VALUE(MCM)) {
        return CompilerType::MCM;
    } else if (val == VPUX_CONFIG_VALUE(MLIR)) {
        return CompilerType::MLIR;
    } else if (val == VPUX_CONFIG_VALUE(DRIVER)) {
        return CompilerType::DRIVER;
    }

    VPUX_THROW("Value '{0}' is not a valid COMPILER_TYPE option", val);
}
