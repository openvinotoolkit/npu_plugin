//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/al/config/common.hpp"
#include "vpux/al/config/compiler.hpp"

#include "vpux/compiler/options_mapper.hpp"

#include <device_helpers.hpp>

using namespace vpux;

//
// getArchKind
//

VPU::ArchKind vpux::getArchKind(const Config& config) {
    auto platform = config.get<PLATFORM>();
    if (platform == InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR) {
        platform = utils::getPlatformByEMUDeviceName(config.get<DEVICE_ID>());
    }

    switch (platform) {
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::AUTO_DETECT:
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR:
        return VPU::ArchKind::UNKNOWN;
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3400:
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3700:
        return VPU::ArchKind::VPUX30XX;
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3800:
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3900:
        return VPU::ArchKind::VPUX311X;
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720:
        return VPU::ArchKind::VPUX37XX;
    default:
        VPUX_THROW("Unsupported VPUX platform");
    }
}

//
// getCompilationMode
//

VPU::CompilationMode vpux::getCompilationMode(const Config& config) {
    if (!config.has<COMPILATION_MODE>()) {
        return VPU::CompilationMode::DefaultHW;
    }

    const auto parsed = VPU::symbolizeCompilationMode(config.get<COMPILATION_MODE>());
    VPUX_THROW_UNLESS(parsed.hasValue(), "Unsupported compilation mode '{0}'", config.get<COMPILATION_MODE>());
    return parsed.getValue();
}

//
// getNumberOfDPUGroups
//

Optional<int> vpux::getNumberOfDPUGroups(const Config& config) {
    if (config.has<DPU_GROUPS>()) {
        return checked_cast<int>(config.get<DPU_GROUPS>());
    }

    switch (config.get<PLATFORM>()) {
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720: {
        switch (config.get<PERFORMANCE_HINT>()) {
        case ov::hint::PerformanceMode::THROUGHPUT:
        case ov::hint::PerformanceMode::UNDEFINED:
        case ov::hint::PerformanceMode::LATENCY:
        default:
            return checked_cast<int>(VPU::getMaxDPUClusterNum(vpux::getArchKind(config)));
        }
        break;
    }
    default: {
        switch (config.get<PERFORMANCE_HINT>()) {
        case ov::hint::PerformanceMode::THROUGHPUT:
            return 1;
        case ov::hint::PerformanceMode::LATENCY:
        case ov::hint::PerformanceMode::UNDEFINED:
        default:
            return checked_cast<int>(VPU::getMaxDPUClusterNum(vpux::getArchKind(config)));
        }
        break;
    }
    }
}

//
// getNumberOfDMAEngines
//

Optional<int> vpux::getNumberOfDMAEngines(const Config& config) {
    if (config.has<DMA_ENGINES>()) {
        return checked_cast<int>(config.get<DMA_ENGINES>());
    }
    switch (config.get<PLATFORM>()) {
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720: {
        switch (config.get<PERFORMANCE_HINT>()) {
        case ov::hint::PerformanceMode::THROUGHPUT:
        case ov::hint::PerformanceMode::UNDEFINED:
        case ov::hint::PerformanceMode::LATENCY:
        default:
            return checked_cast<int>(VPU::getMaxDMAPorts(vpux::getArchKind(config)));
        }
        break;
    }
    default: {
        switch (config.get<PERFORMANCE_HINT>()) {
        case ov::hint::PerformanceMode::THROUGHPUT:
            return 1;
        case ov::hint::PerformanceMode::LATENCY:
        case ov::hint::PerformanceMode::UNDEFINED:
        default:
            return checked_cast<int>(VPU::getMaxDMAPorts(vpux::getArchKind(config)));
        }
        break;
    }
    }
}

//
// getDDRHeapSize
//

Optional<int> vpux::getDDRHeapSize(const Config& config) {
    if (config.has<DDR_HEAP_SIZE_MB>()) {
        return checked_cast<int>(config.get<DDR_HEAP_SIZE_MB>());
    }

    return 500;
}
