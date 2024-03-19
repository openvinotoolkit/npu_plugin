//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/al/config/common.hpp"
#include "vpux/al/config/compiler.hpp"

#include "vpux/compiler/compiler.hpp"
#include "vpux/compiler/options_mapper.hpp"

#include <device_helpers.hpp>

using namespace vpux;

//
// getArchKind
//

VPU::ArchKind vpux::getArchKind(const Config& config) {
    auto platform = config.get<PLATFORM>();

    switch (platform) {
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::AUTO_DETECT:
        return VPU::ArchKind::UNKNOWN;
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3700:
        return VPU::ArchKind::VPUX30XX;
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
    VPUX_THROW_UNLESS(parsed.has_value(), "Unsupported compilation mode '{0}'", config.get<COMPILATION_MODE>());
    return parsed.value();
}

//
// getNumberOfDPUGroups
//

std::optional<int> vpux::getNumberOfDPUGroups(const Config& config) {
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

std::optional<int> vpux::getNumberOfDMAEngines(const Config& config) {
    if (config.has<DMA_ENGINES>()) {
        return checked_cast<int>(config.get<DMA_ENGINES>());
    }

    auto archKind = vpux::getArchKind(config);
    auto numOfDpuGroups = getNumberOfDPUGroups(config);
    int maxDmaPorts = VPU::getMaxDMAPorts(archKind);

    auto getNumOfDmaPortsWithDpuCountLimit = [&]() {
        return std::min(maxDmaPorts, numOfDpuGroups.value_or(maxDmaPorts));
    };

    switch (config.get<PLATFORM>()) {
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720: {
        switch (config.get<PERFORMANCE_HINT>()) {
        case ov::hint::PerformanceMode::THROUGHPUT:
        case ov::hint::PerformanceMode::UNDEFINED:
        case ov::hint::PerformanceMode::LATENCY:
        default:
            return getNumOfDmaPortsWithDpuCountLimit();
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
            return maxDmaPorts;
        }
        break;
    }
    }
}
