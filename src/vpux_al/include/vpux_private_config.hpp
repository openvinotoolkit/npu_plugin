//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

/**
 * @brief VPUX platform configuration
 *
 * @deprecated Configuration API v1.0 would be deprecated in 2023.1 release.
 * It was left due to backward compatibility needs.
 * As such usage of this version of API is discouraged.
 * Prefer Configuration API v2.0.
 *
 */

#pragma once

#include <string>
#include <vpux/vpux_plugin_config.hpp>

namespace InferenceEngine {
namespace VPUXConfigParams {

/**
 * @enum VPUXPlatform
 * @brief VPUX device
 */
enum class VPUXPlatform : int {
    AUTO_DETECT = 0,  // Auto detection
    EMULATOR = 1,     // Emulator
    VPU3700 = 2,      // VPU30XX
    VPU3720 = 3,      // VPU37XX
};

/**
 * @brief [Only for VPUX Plugin]
 * Type: Arbitrary string.
 * This option allows to specify device.
 * If specified device is not available then creating infer request will throw an exception.
 */
DECLARE_VPUX_CONFIG_KEY(PLATFORM);

inline std::ostream& operator<<(std::ostream& os, const VPUXPlatform& vpux_platform) {
    switch (vpux_platform) {
    case VPUXPlatform::AUTO_DETECT:
        return os << "AUTO_DETECT";
    case VPUXPlatform::EMULATOR:
        return os << "EMULATOR";
    case VPUXPlatform::VPU3700:
        return os << "VPU3700";
    case VPUXPlatform::VPU3720:
        return os << "VPU3720";
    default:
        return os << "0x" << std::hex << static_cast<uint8_t>(vpux_platform);
    }
}

/**
 * @brief [Only for VPUX Plugin]
 * Type: string, default is MLIR.
 * Type of VPU compiler to be used for compilation of a network
 */
enum class CompilerType { MLIR, DRIVER };
DECLARE_VPUX_CONFIG_KEY(COMPILER_TYPE);
DECLARE_VPUX_CONFIG_VALUE(MLIR);
DECLARE_VPUX_CONFIG_VALUE(DRIVER);

DECLARE_VPUX_CONFIG_KEY(COMPILATION_MODE);

/**
 * @brief [Only for VPUX compiler]
 * Type: std::string, default is empty.
 * Config for HW-mode's pipeline
 * Available values: low-precision=true/low-precision=false
 */
DECLARE_VPUX_CONFIG_KEY(COMPILATION_MODE_PARAMS);

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is None
 * Number of DPU groups
 */
DECLARE_VPUX_CONFIG_KEY(DPU_GROUPS);

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is None
 * Number of DMA engines
 */
DECLARE_VPUX_CONFIG_KEY(DMA_ENGINES);

/**
 * @brief [Only for VPUX Plugin]
 * Type: string, default is NONE
 * NONE - do not print profiling info
 * TEXT, JSON - print detailed profiling info during inference in requested format
 */
enum class ProfilingOutputTypeArg { NONE, TEXT, JSON };
DECLARE_VPUX_CONFIG_KEY(PRINT_PROFILING);
DECLARE_VPUX_CONFIG_VALUE(NONE);
DECLARE_VPUX_CONFIG_VALUE(TEXT);
DECLARE_VPUX_CONFIG_VALUE(JSON);

inline std::ostream& operator<<(std::ostream& os, const ProfilingOutputTypeArg& profiling_output_type) {
    switch (profiling_output_type) {
    case ProfilingOutputTypeArg::NONE:
        return os << "NONE";
    case ProfilingOutputTypeArg::TEXT:
        return os << "TEXT";
    case ProfilingOutputTypeArg::JSON:
        return os << "JSON";
    default:
        return os << "0x" << std::hex << static_cast<uint8_t>(profiling_output_type);
    }
}

/**
 * @brief [Only for VPUX Plugin]
 * Type: string, default is empty.
 * File that contains profiling output.
 * std::cout is used if this string is empty
 */
DECLARE_VPUX_CONFIG_KEY(PROFILING_OUTPUT_FILE);

/**
 * @brief [Only for VPUX Plugin]
 * Type: String. Default is "AUTO".
 * This option is added for enabling ELF backend.
 * Possible values: "AUTO", "YES", "NO".
 */

enum class ElfCompilerBackend {
    AUTO = 0,
    NO = 1,
    YES = 2,
};

DECLARE_VPUX_CONFIG_KEY(USE_ELF_COMPILER_BACKEND);

inline std::ostream& operator<<(std::ostream& os, const ElfCompilerBackend& elf_compiler_backend) {
    switch (elf_compiler_backend) {
    case ElfCompilerBackend::AUTO:
        return os << "AUTO";
    case ElfCompilerBackend::NO:
        return os << "NO";
    case ElfCompilerBackend::YES:
        return os << "YES";
    default:
        return os << "0x" << std::hex << static_cast<uint8_t>(elf_compiler_backend);
    }
}

}  // namespace VPUXConfigParams
}  // namespace InferenceEngine
