//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <openvino/runtime/properties.hpp>
#include <string>
#include <vpux/utils/core/error.hpp>
#include <vpux/vpux_plugin_config.hpp>

#include "vpux_private_config.hpp"

namespace ov {
namespace intel_vpux {

/**
 * @enum ColorFormat
 * @brief Extra information about input color format for preprocessing
 * @note Configuration API v 2.0
 */
enum ColorFormat : uint32_t {
    RAW = 0u,  ///< Plain blob (default), no extra color processing required
    RGB,       ///< RGB color format
    BGR,       ///< BGR color format, default in DLDT
    RGBX,      ///< RGBX color format with X ignored during inference
    BGRX,      ///< BGRX color format with X ignored during inference
};

/**
 * @brief Converts ov::intel_vpux::ColorFormat to InferenceEngine::ColorFormat
 * @param fmt ov::intel_vpux::ColorFormat value to convert
 * @return the corresponding analogue of fmt in InferenceEngine::ColorFormat
 * @note Configuration API v 2.0
 */
inline InferenceEngine::ColorFormat cvtColorFormat(ColorFormat fmt) {
    switch (fmt) {
    case ColorFormat::RAW:
        return InferenceEngine::ColorFormat::RAW;
    case ColorFormat::RGB:
        return InferenceEngine::ColorFormat::RGB;
    case ColorFormat::BGR:
        return InferenceEngine::ColorFormat::BGR;
    case ColorFormat::RGBX:
        return InferenceEngine::ColorFormat::RGBX;
    case ColorFormat::BGRX:
        return InferenceEngine::ColorFormat::BGRX;
    default:
        VPUX_THROW("Unknown ColorFormat {0} to onvert to InferenceEngine::ColorFormat", static_cast<uint32_t>(fmt));
    }
}

/**
 * @brief Prints a string representation of ov::intel_vpux::ColorFormat to a stream
 * @param out An output stream to send to
 * @param fmt A color format value to print to a stream
 * @return A reference to the `out` stream
 * @note Configuration API v 2.0
 */
inline std::ostream& operator<<(std::ostream& out, const ColorFormat& fmt) {
    switch (fmt) {
    case ColorFormat::RAW: {
        out << "RAW";
    } break;
    case ColorFormat::RGB: {
        out << "RGB";
    } break;
    case ColorFormat::BGR: {
        out << "BGR";
    } break;
    case ColorFormat::RGBX: {
        out << "RGBX";
    } break;
    case ColorFormat::BGRX: {
        out << "BGRX";
    } break;
    default:
        out << static_cast<uint32_t>(fmt);
        break;
    }
    return out;
}

/**
 * @enum VPUXPlatform
 * @brief VPUX device
 * @note Configuration API v 2.0
 */
enum class VPUXPlatform : int {
    AUTO_DETECT = 0,  // Auto detection
    EMULATOR = 1,     // Emulator
    VPU3700 = 2,      // VPU30XX
    VPU3720 = 3,      // VPU37XX
};

/**
 * @brief Converts ov::intel_vpux::VPUXPlatform to InferenceEngine::VPUXConfigParams::VPUXPlatform
 * @param fmt ov::intel_vpux::VPUXPlatform value to convert
 * @return the corresponding analogue of fmt in InferenceEngine::VPUXConfigParams::VPUXPlatform
 * @note Configuration API v 2.0
 */
inline InferenceEngine::VPUXConfigParams::VPUXPlatform cvtVPUXPlatform(VPUXPlatform fmt) {
    switch (fmt) {
    case VPUXPlatform::AUTO_DETECT:
        return InferenceEngine::VPUXConfigParams::VPUXPlatform::AUTO_DETECT;
    case VPUXPlatform::EMULATOR:
        return InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR;
    case VPUXPlatform::VPU3700:
        return InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3700;
    case VPUXPlatform::VPU3720:
        return InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720;
    default:
        VPUX_THROW("Unknown VPUXPlatform {0} to convert to InferenceEngine::VPUXConfigParams::VPUXPlatform",
                   static_cast<uint32_t>(fmt));
    }
}

/**
 * @brief Prints a string representation of ov::intel_vpux::VPUXPlatform to a stream
 * @param out An output stream to send to
 * @param fmt A VPUX platform value to print to a stream
 * @return A reference to the `out` stream
 * @note Configuration API v 2.0
 */
inline std::ostream& operator<<(std::ostream& out, const VPUXPlatform& fmt) {
    switch (fmt) {
    case VPUXPlatform::AUTO_DETECT: {
        out << "AUTO_DETECT";
    } break;
    case VPUXPlatform::EMULATOR: {
        out << "EMULATOR";
    } break;
    case VPUXPlatform::VPU3700: {
        out << "VPU3700";
    } break;
    case VPUXPlatform::VPU3720: {
        out << "VPU3720";
    } break;
    default:
        out << static_cast<uint32_t>(fmt);
        break;
    }
    return out;
}

/**
 * @brief [Only for VPUX Plugin]
 * Type: string, default is NONE
 * NONE - do not print profiling info
 * TEXT, JSON - print detailed profiling info during inference in requested format
 * @note Configuration API v 2.0
 */
enum class ProfilingOutputTypeArg { NONE, TEXT, JSON };

/**
 * @brief Converts ov::intel_vpux::ProfilingOutputTypeArg to InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg
 * @param fmt ov::intel_vpux::ProfilingOutputTypeArg value to convert
 * @return the corresponding analogue of fmt in InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg
 * @note Configuration API v 2.0
 */
inline InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg cvtProfilingOutputType(ProfilingOutputTypeArg fmt) {
    switch (fmt) {
    case ProfilingOutputTypeArg::NONE:
        return InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg::NONE;
    case ProfilingOutputTypeArg::TEXT:
        return InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg::TEXT;
    case ProfilingOutputTypeArg::JSON:
        return InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg::JSON;
    default:
        VPUX_THROW("Unknown ProfilingOutputType {0} to convert to "
                   "InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg",
                   static_cast<uint32_t>(fmt));
    }
}

/**
 * @brief Prints a string representation of ov::intel_vpux::ProfilingOutputTypeArg to a stream
 * @param out An output stream to send to
 * @param fmt A Profiling output type value to print to a stream
 * @return A reference to the `out` stream
 * @note Configuration API v 2.0
 */
inline std::ostream& operator<<(std::ostream& out, const ProfilingOutputTypeArg& fmt) {
    switch (fmt) {
    case ProfilingOutputTypeArg::NONE: {
        out << "NONE";
    } break;
    case ProfilingOutputTypeArg::TEXT: {
        out << "TEXT";
    } break;
    case ProfilingOutputTypeArg::JSON: {
        out << "JSON";
    } break;
    default:
        out << static_cast<uint32_t>(fmt);
        break;
    }
    return out;
}

/**
 * @brief [Only for VPUX Plugin]
 * Type: string, default is MLIR.
 * Type of VPU compiler to be used for compilation of a network
 * @note Configuration API v 2.0
 */
enum class CompilerType { MLIR, DRIVER };

/**
 * @brief Converts ov::intel_vpux::CompilerType to InferenceEngine::VPUXConfigParams::CompilerType
 * @param fmt ov::intel_vpux::CompilerType value to convert
 * @return the corresponding analogue of fmt in InferenceEngine::VPUXConfigParams::CompilerType
 * @note Configuration API v 2.0
 */
inline InferenceEngine::VPUXConfigParams::CompilerType cvtCompilerType(CompilerType fmt) {
    switch (fmt) {
    case CompilerType::MLIR:
        return InferenceEngine::VPUXConfigParams::CompilerType::MLIR;
    case CompilerType::DRIVER:
        return InferenceEngine::VPUXConfigParams::CompilerType::DRIVER;
    default:
        VPUX_THROW("Unknown CompilerType {0} to convert to "
                   "InferenceEngine::VPUXConfigParams::CompilerType",
                   static_cast<uint32_t>(fmt));
    }
}

/**
 * @brief Prints a string representation of ov::intel_vpux::CompilerType to a stream
 * @param out An output stream to send to
 * @param fmt A compiler type value to print to a stream
 * @return A reference to the `out` stream
 * @note Configuration API v 2.0
 */
inline std::ostream& operator<<(std::ostream& out, const CompilerType& fmt) {
    switch (fmt) {
    case CompilerType::MLIR: {
        out << "MLIR";
    } break;
    case CompilerType::DRIVER: {
        out << "DRIVER";
    } break;
    default:
        out << static_cast<uint32_t>(fmt);
        break;
    }
    return out;
}

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

/**
 * @brief Converts ov::intel_vpux::CompilerType to InferenceEngine::VPUXConfigParams::ElfCompilerBackend
 * @param fmt ov::intel_vpux::CompilerType value to convert
 * @return the corresponding analogue of fmt in InferenceEngine::VPUXConfigParams::ElfCompilerBackend
 * @note Configuration API v 2.0
 */
inline InferenceEngine::VPUXConfigParams::ElfCompilerBackend cvtCompilerType(ElfCompilerBackend fmt) {
    switch (fmt) {
    case ElfCompilerBackend::AUTO:
        return InferenceEngine::VPUXConfigParams::ElfCompilerBackend::AUTO;
    case ElfCompilerBackend::NO:
        return InferenceEngine::VPUXConfigParams::ElfCompilerBackend::NO;
    case ElfCompilerBackend::YES:
        return InferenceEngine::VPUXConfigParams::ElfCompilerBackend::YES;
    default:
        VPUX_THROW("Unknown ElfCompilerBackend {0} to convert to "
                   "InferenceEngine::VPUXConfigParams::ElfCompilerBackend",
                   static_cast<uint32_t>(fmt));
    }
}

/**
 * @brief Prints a string representation of ov::intel_vpux::ElfCompilerBackend to a stream
 * @param out An output stream to send to
 * @param fmt A elf compiler backend value to print to a stream
 * @return A reference to the `out` stream
 * @note Configuration API v 2.0
 */
inline std::ostream& operator<<(std::ostream& out, const ElfCompilerBackend& fmt) {
    switch (fmt) {
    case ElfCompilerBackend::AUTO: {
        out << "AUTO";
    } break;
    case ElfCompilerBackend::NO: {
        out << "NO";
    } break;
    case ElfCompilerBackend::YES: {
        out << "YES";
    } break;
    default:
        out << static_cast<uint32_t>(fmt);
        break;
    }
    return out;
}

/**
 * @brief [Only for VPUX Plugin]
 * Type: Arbitrary string.
 * This option allows to specify device.
 * If specified device is not available then creating infer request will throw an exception.
 */
static constexpr ov::Property<VPUXPlatform> vpux_platform{"NPU_PLATFORM"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: string, default is MLIR for DEVELOPER_BUILD, DRIVER otherwise.
 * Type of VPU compiler to be used for compilation of a network
 */
static constexpr ov::Property<CompilerType> compiler_type{"NPU_COMPILER_TYPE"};

static constexpr ov::Property<std::string> compilation_mode{"NPU_COMPILATION_MODE"};

/**
 * @brief [Only for VPUX compiler]
 * Type: std::string, default is empty.
 * Config for HW-mode's pipeline
 * Available values: low-precision=true/low-precision=false
 */
static constexpr ov::Property<std::string> compilation_mode_params{"NPU_COMPILATION_MODE_PARAMS"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is None
 * Number of DPU groups
 */
static constexpr ov::Property<int64_t> dpu_groups{"NPU_DPU_GROUPS"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is None
 * Number of DMA engines
 */
static constexpr ov::Property<int64_t> dma_engines{"NPU_DMA_ENGINES"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: string, default is NONE
 * NONE - do not print profiling info
 * TEXT, JSON - print detailed profiling info during inference in requested format
 */
static constexpr ov::Property<ProfilingOutputTypeArg> print_profiling{"NPU_PRINT_PROFILING"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: string, default is empty.
 * File that contains profiling output.
 * std::cout is used if this string is empty
 */
static constexpr ov::Property<std::string> profiling_output_file{"NPU_PROFILING_OUTPUT_FILE"};

/**
 * @brief
 * Type: String. Default is "AUTO".
 * This option is added for enabling ELF backend.
 * Possible values: "AUTO", "YES", "NO".
 */
static constexpr ov::Property<ElfCompilerBackend> use_elf_compiler_backend{"NPU_USE_ELF_COMPILER_BACKEND"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is 1
 * This option allows to omit creating an executor and therefore to omit running an inference when its value is 0
 */
static constexpr ov::Property<int64_t> create_executor{"NPU_CREATE_EXECUTOR"};

}  // namespace intel_vpux
}  // namespace ov
