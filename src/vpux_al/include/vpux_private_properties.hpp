//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <openvino/runtime/properties.hpp>
#include <string>
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
    NV12,      ///< NV12 color format represented as compound Y+UV blob
    I420,      ///< I420 color format represented as compound Y+U+V blob
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
    case ColorFormat::NV12:
        return InferenceEngine::ColorFormat::NV12;
    case ColorFormat::I420:
        return InferenceEngine::ColorFormat::I420;
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
    case ColorFormat::NV12: {
        out << "NV12";
    } break;
    case ColorFormat::I420: {
        out << "I420";
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
    AUTO = 0,        // Auto detection
    VPU3400_A0 = 1,  // Keem bay A0
    VPU3400 = 2,     // Keem bay B0 500 MHz
    VPU3700 = 3,     // Keem bay B0 700 MHz
    VPU3800 = 4,     // Thunder bay harbor Prime
    VPU3900 = 5,     // Thunder bay harbor Full
    VPU3720 = 6,     // VPU3720
    EMULATOR = 7,    // Emulator
    VPU3720ELF = 8,  // VPU3720 ELF
    VPU4000 = 9,     // VPU4000
};

/**
 * @brief Converts ov::intel_vpux::VPUXPlatform to InferenceEngine::VPUXConfigParams::VPUXPlatform
 * @param fmt ov::intel_vpux::VPUXPlatform value to convert
 * @return the corresponding analogue of fmt in InferenceEngine::VPUXConfigParams::VPUXPlatform
 * @note Configuration API v 2.0
 */
inline InferenceEngine::VPUXConfigParams::VPUXPlatform cvtVPUXPlatform(VPUXPlatform fmt) {
    switch (fmt) {
    case VPUXPlatform::AUTO:
        return InferenceEngine::VPUXConfigParams::VPUXPlatform::AUTO;
    case VPUXPlatform::VPU3400_A0:
        return InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3400_A0;
    case VPUXPlatform::VPU3400:
        return InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3400;
    case VPUXPlatform::VPU3700:
        return InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3700;
    case VPUXPlatform::VPU3800:
        return InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3800;
    case VPUXPlatform::VPU3900:
        return InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3900;
    case VPUXPlatform::VPU3720:
        return InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720;
    case VPUXPlatform::VPU3720ELF:
        return InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720ELF;
    case VPUXPlatform::VPU4000:
        return InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU4000;
    case VPUXPlatform::EMULATOR:
        return InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR;
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
    case VPUXPlatform::AUTO: {
        out << "AUTO";
    } break;
    case VPUXPlatform::VPU3400_A0: {
        out << "VPU3400_A0";
    } break;
    case VPUXPlatform::VPU3400: {
        out << "VPU3400";
    } break;
    case VPUXPlatform::VPU3700: {
        out << "VPU3700";
    } break;
    case VPUXPlatform::VPU3800: {
        out << "VPU3800";
    } break;
    case VPUXPlatform::VPU3900: {
        out << "VPU3900";
    } break;
    case VPUXPlatform::VPU3720: {
        out << "VPU3720";
    } break;
    case VPUXPlatform::VPU3720ELF: {
        out << "VPU3720ELF";
    } break;
    case VPUXPlatform::VPU4000: {
        out << "VPU4000";
    } break;
    case VPUXPlatform::EMULATOR: {
        out << "EMULATOR";
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
 * Type: Arbitrary string.
 * This option allows to specify device.
 * If specified device is not available then creating infer request will throw an exception.
 */
static constexpr ov::Property<InferenceEngine::VPUXConfigParams::VPUXPlatform> vpux_platform{"VPUX_PLATFORM"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: "RGB", "BGR", default is "BGR"
 * This option allows to specify output format of image after SIPP preprocessing.
 * Does not affect preprocessing running on CPU. If a wrong value specified an exception will be thrown
 */
static constexpr ov::Property<ColorFormat> graph_color_format{"VPUX_GRAPH_COLOR_FORMAT"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is 4.
 * Number of shaves to be used by SIPP during preprocessing
 */
static constexpr ov::Property<int64_t> preprocessing_shaves{"VPUX_PREPROCESSING_SHAVES"};

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: integer, default is 8.
 * Lines per iteration value to be used by SIPP during preprocessing
 */
static constexpr ov::Property<int64_t> preprocessing_lpi{"VPUX_PREPROCESSING_LPI"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is 1.
 * Number of preprocessing pipelines to be used by particular network,
 * these pipelines will work in parallel and make preprocessing
 * for all infer requests of this network
 */
static constexpr ov::Property<int64_t> preprocessing_pipes{"VPUX_PREPROCESSING_PIPES"};

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: "YES", "NO", default is "NO"
 * This option allows to use Media-to-Inference (M2I) module for image pre-processing
 */
static constexpr ov::Property<bool> use_m2i{"VPUX_USE_M2I"};

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: "YES", "NO", default is "NO"
 * This option allows to use Media-to-Inference (M2I)
 * SHAVE only version module for image pre-processing
 */
static constexpr ov::Property<bool> use_shave_only_m2i{"VPUX_USE_SHAVE_ONLY_M2I"};

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: "YES", "NO", default is "YES"
 * This option allows to use Streaming Image Processing Pipeline (SIPP) for image pre-processing
 */
static constexpr ov::Property<bool> use_sipp{"VPUX_USE_SIPP"};

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: integer, default is 1.
 * Number of executor streams
 */
static constexpr ov::Property<int64_t> executor_streams{"VPUX_EXECUTOR_STREAMS"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is 5 minutes = 60 * 1000 * 5.
 * Time interval during which to wait for backend pull to complete
 */
static constexpr ov::Property<int64_t> inference_timeout{"VPUX_INFERENCE_TIMEOUT"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: string, default is MLIR.
 * Type of VPU compiler to be used for compilation of a network
 */
static constexpr ov::Property<InferenceEngine::VPUXConfigParams::CompilerType> compiler_type{"VPUX_COMPILER_TYPE"};

static constexpr ov::Property<std::string> compilation_mode{"VPUX_COMPILATION_MODE"};

/**
 * @brief [Only for VPUX compiler]
 * Type: std::string, default is empty.
 * Config for HW-mode's pipeline
 * Available values: low-precision=true/low-precision=false
 */
static constexpr ov::Property<std::string> compilation_mode_params{"VPUX_COMPILATION_MODE_PARAMS"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is None
 * Number of DPU groups
 */
static constexpr ov::Property<int64_t> dpu_groups{"VPUX_DPU_GROUPS"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: string, default is NONE
 * NONE - do not print profiling info
 * TEXT, JSON - print detailed profiling info during inference in requested format
 */
static constexpr ov::Property<InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg> print_profiling{
        "VPUX_PRINT_PROFILING"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: string, default is empty.
 * File that contains profiling output.
 * std::cout is used if this string is empty
 */
static constexpr ov::Property<std::string> profiling_output_file{"VPUX_PROFILING_OUTPUT_FILE"};

}  // namespace intel_vpux
}  // namespace ov
