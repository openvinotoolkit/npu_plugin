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

#include "vpux/al/config/common.hpp"

using namespace vpux;
using namespace InferenceEngine::VPUXConfigParams;

//
// register
//

void vpux::registerCommonOptions(OptionsDesc& desc) {
    desc.add<PERFORMANCE_HINT>();
    desc.add<PERF_COUNT>();
    desc.add<LOG_LEVEL>();
    desc.add<PLATFORM>();
    desc.add<DEVICE_ID>();
}

//
// PERFORMANCE_HINT
//

StringLiteral vpux::stringifyEnum(PerformanceHint val) {
    switch (val) {
    case PerformanceHint::Latency:
        return "Latency";
    case PerformanceHint::Throughput:
        return "Throughput";
    default:
        return "<UNKNOWN>";
    }
}

PerformanceHint vpux::PERFORMANCE_HINT::parse(StringRef val) {
    if (val == CONFIG_VALUE(LATENCY)) {
        return PerformanceHint::Latency;
    } else if (val == CONFIG_VALUE(THROUGHPUT)) {
        return PerformanceHint::Throughput;
    }

    VPUX_THROW("Value '{0}' is not a valid PERFORMANCE_HINT option");
}

//
// PLATFORM
//

StringLiteral vpux::stringifyEnum(VPUXPlatform val) {
    switch (val) {
    case VPUXPlatform::AUTO:
        return "AUTO";
    case VPUXPlatform::VPU3400_A0:
        return "VPU3400_A0";
    case VPUXPlatform::VPU3400:
        return "VPU3400";
    case VPUXPlatform::VPU3700:
        return "VPU3700";
    case VPUXPlatform::VPU3800:
        return "VPU3800";
    case VPUXPlatform::VPU3900:
        return "VPU3900";
    case VPUXPlatform::VPU3720:
        return "VPU3720";
    case VPUXPlatform::EMULATOR:
        return "EMULATOR";
    default:
        return "<UNKNOWN>";
    }
}

VPUXPlatform vpux::PLATFORM::parse(StringRef val) {
    // TODO: Remove deprecated platform names with VPU prefix in future releases

    if (val == "AUTO") {
        return VPUXPlatform::AUTO;
    } else if (val == "3400_A0" || val == "VPU3400_A0") {
        return VPUXPlatform::VPU3400_A0;
    } else if (val == "3400" || val == "VPU3400") {
        return VPUXPlatform::VPU3400;
    } else if (val == "3700" || val == "VPU3700") {
        return VPUXPlatform::VPU3700;
    } else if (val == "3800" || val == "VPU3800") {
        return VPUXPlatform::VPU3800;
    } else if (val == "3900" || val == "VPU3900") {
        return VPUXPlatform::VPU3900;
    } else if (val == "3720" || val == "VPU3720") {
        return VPUXPlatform::VPU3720;
    } else if (val == "3400_A0_EMU" || val == "3400_EMU" || val == "3700_EMU" || val == "3800_EMU" ||
               val == "3900_EMU" || val == "3720_EMU") {
        return VPUXPlatform::EMULATOR;
    }

    VPUX_THROW("Value '{0}' is not a valid PLATFORM option");
}

//
// LOG_LEVEL
//

vpu::LogLevel vpux::toOldLogLevel(LogLevel lvl) {
    switch (lvl) {
    case LogLevel::None:
        return vpu::LogLevel::None;
    case LogLevel::Fatal:
        return vpu::LogLevel::Fatal;
    case LogLevel::Error:
        return vpu::LogLevel::Error;
    case LogLevel::Warning:
        return vpu::LogLevel::Warning;
    case LogLevel::Info:
        return vpu::LogLevel::Info;
    case LogLevel::Debug:
        return vpu::LogLevel::Debug;
    case LogLevel::Trace:
        return vpu::LogLevel::Trace;
    }

    VPUX_THROW("LogLevel '{0}' is not supported", lvl);
}
