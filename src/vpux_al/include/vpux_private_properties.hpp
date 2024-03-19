//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <openvino/runtime/properties.hpp>
#include <vpux/utils/core/error.hpp>

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
    VPU3700 = 2,      // VPU30XX
    VPU3720 = 3,      // VPU37XX
};

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
 * Type: string, default is MODEL.
 * Type of profiling to execute. Can be Model (default) or INFER (based on npu timestamps)
 * @note Configuration API v 2.0
 */
enum class ProfilingType { MODEL, INFER };

/**
 * @brief Prints a string representation of ov::intel_vpux::ProfilingType to a stream
 * @param out An output stream to send to
 * @param fmt A profiling type value to print to a stream
 * @return A reference to the `out` stream
 * @note Configuration API v 2.0
 */
inline std::ostream& operator<<(std::ostream& out, const ProfilingType& fmt) {
    switch (fmt) {
    case ProfilingType::MODEL: {
        out << "MODEL";
    } break;
    case ProfilingType::INFER: {
        out << "INFER";
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
static constexpr ov::Property<VPUXPlatform> platform{"NPU_PLATFORM"};

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
 * @brief [Only for VPUX Plugin]
 * Type: string, default is empty.
 * MODEL - model layer profiling is done
 * INFER - vpu inference performance numbers are measured
 * Model layers profiling are used if this string is empty
 */
static constexpr ov::Property<ProfilingType> profiling_type{"NPU_PROFILING_TYPE"};

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

namespace InferenceEngine {

namespace VPUXConfigParams = ::ov::intel_vpux;

}
