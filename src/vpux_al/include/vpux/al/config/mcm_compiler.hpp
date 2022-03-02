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

#pragma once

#include "vpux/utils/IE/config.hpp"

#include "mcm_private_config.hpp"
#include "mcm_private_properties.hpp"
#include "vpux/properties.hpp"
#include "vpux/vpux_compiler_config.hpp"

#include <openvino/runtime/properties.hpp>

namespace vpux {

//
// register
//

void registerMcmCompilerOptions(OptionsDesc& desc);

//
// TARGET_DESCRIPTOR_PATH
//

struct MCM_TARGET_DESCRIPTOR_PATH final : OptionBase<MCM_TARGET_DESCRIPTOR_PATH, std::string> {
    static StringRef key() {
        return ov::intel_vpux::target_descriptor_path.name();
    }

    static std::string defaultValue() {
        return "mcm_config/target";
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

//
// TARGET_DESCRIPTOR
//

struct MCM_TARGET_DESCRIPTOR final : OptionBase<MCM_TARGET_DESCRIPTOR, std::string> {
    static StringRef key() {
        return ov::intel_vpux::target_descriptor.name();
    }

    static std::string defaultValue() {
        return "release_kmb";
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

//
// COMPILATION_DESCRIPTOR_PATH
//

struct MCM_COMPILATION_DESCRIPTOR_PATH final : OptionBase<MCM_COMPILATION_DESCRIPTOR_PATH, std::string> {
    static StringRef key() {
        return ov::intel_vpux::compilation_descriptor_path.name();
    }

    static std::string defaultValue() {
        return "mcm_config/compilation";
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

//
// COMPILATION_DESCRIPTOR
//

struct MCM_COMPILATION_DESCRIPTOR final : OptionBase<MCM_COMPILATION_DESCRIPTOR, std::string> {
    static StringRef key() {
        return ov::intel_vpux::compilation_descriptor.name();
    }

    static std::string defaultValue() {
        return "release_kmb";
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

//
// LOG_LEVEL
//

struct MCM_LOG_LEVEL final : OptionBase<MCM_LOG_LEVEL, LogLevel> {
    static StringRef key() {
        return ov::intel_vpux::log_level.name();
    }

    static LogLevel defaultValue() {
        return LogLevel::None;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

//
// ELTWISE_SCALES_ALIGNMENT
//

struct MCM_ELTWISE_SCALES_ALIGNMENT final : OptionBase<MCM_ELTWISE_SCALES_ALIGNMENT, bool> {
    static StringRef key() {
        return ov::intel_vpux::eltwise_scales_alignment.name();
    }

    static bool defaultValue() {
        return true;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

//
// CONCAT_SCALES_ALIGNMENT
//

struct MCM_CONCAT_SCALES_ALIGNMENT final : OptionBase<MCM_CONCAT_SCALES_ALIGNMENT, bool> {
    static StringRef key() {
        return ov::intel_vpux::concat_scales_alignment.name();
    }

    static bool defaultValue() {
        return true;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

//
// WEIGHTS_ZERO_POINTS_ALIGNMENT
//

struct MCM_WEIGHTS_ZERO_POINTS_ALIGNMENT final : OptionBase<MCM_WEIGHTS_ZERO_POINTS_ALIGNMENT, bool> {
    static StringRef key() {
        return ov::intel_vpux::weights_zero_points_alignment.name();
    }

    static bool defaultValue() {
        return true;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

//
// COMPILATION_PASS_BAN_LIST
//

struct MCM_COMPILATION_PASS_BAN_LIST final : OptionBase<MCM_COMPILATION_PASS_BAN_LIST, std::string> {
    static StringRef key() {
        return ov::intel_vpux::compilation_pass_ban_list.name();
    }

    static std::string defaultValue() {
        return "";
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

//
// SCALE_FUSE_INPUT
//

struct MCM_SCALE_FUSE_INPUT final : OptionBase<MCM_SCALE_FUSE_INPUT, bool> {
    static StringRef key() {
        return ov::intel_vpux::scale_fuse_input.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

//
// ALLOW_NCHW_MCM_INPUT
//

struct MCM_ALLOW_NCHW_MCM_INPUT final : OptionBase<MCM_ALLOW_NCHW_MCM_INPUT, bool> {
    static StringRef key() {
        return ov::intel_vpux::allow_nchw_mcm_input.name();
    }

    static bool defaultValue() {
        return true;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

//
// REMOVE_PERMUTE_NOOP
//

struct MCM_REMOVE_PERMUTE_NOOP final : OptionBase<MCM_REMOVE_PERMUTE_NOOP, bool> {
    static StringRef key() {
        return ov::intel_vpux::remove_permute_noop.name();
    }

    static bool defaultValue() {
        return true;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

//
// SERIALIZE_CNN_BEFORE_COMPILE_FILE
//

struct MCM_SERIALIZE_CNN_BEFORE_COMPILE_FILE final : OptionBase<MCM_SERIALIZE_CNN_BEFORE_COMPILE_FILE, std::string> {
    static StringRef key() {
        return ov::intel_vpux::serialize_cnn_before_compile_file.name();
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// REFERENCE_MODE
//

struct MCM_REFERENCE_MODE final : OptionBase<MCM_REFERENCE_MODE, bool> {
    static StringRef key() {
        return ov::intel_vpux::reference_mode.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// ALLOW_U8_INPUT_FOR_FP16_MODELS
//

struct MCM_ALLOW_U8_INPUT_FOR_FP16_MODELS final : OptionBase<MCM_ALLOW_U8_INPUT_FOR_FP16_MODELS, bool> {
    static StringRef key() {
        return ov::intel_vpux::allow_u8_input_for_fp16_mode.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// SCALESHIFT_FUSING
//

struct MCM_SCALESHIFT_FUSING final : OptionBase<MCM_SCALESHIFT_FUSING, bool> {
    static StringRef key() {
        return ov::intel_vpux::scale_shift_fusing.name();
    }

    static bool defaultValue() {
        return true;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// ALLOW_PERMUTE_ND
//

struct MCM_ALLOW_PERMUTE_ND final : OptionBase<MCM_ALLOW_PERMUTE_ND, bool> {
    static StringRef key() {
        return ov::intel_vpux::allow_permute_nd.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// OPTIMIZE_INPUT_PRECISION
//

struct MCM_OPTIMIZE_INPUT_PRECISION final : OptionBase<MCM_OPTIMIZE_INPUT_PRECISION, bool> {
    static StringRef key() {
        return ov::intel_vpux::optimize_input_precision.name();
    }

    static bool defaultValue() {
        return true;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// FORCE_PLUGIN_INPUT_QUANTIZATION
//

struct MCM_FORCE_PLUGIN_INPUT_QUANTIZATION final : OptionBase<MCM_FORCE_PLUGIN_INPUT_QUANTIZATION, bool> {
    static StringRef key() {
        return ov::intel_vpux::force_plugin_input_quantization.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// OUTPUT_FP16_TO_FP32_HOST_CONVERSION
//

struct MCM_OUTPUT_FP16_TO_FP32_HOST_CONVERSION final : OptionBase<MCM_OUTPUT_FP16_TO_FP32_HOST_CONVERSION, bool> {
    static StringRef key() {
        return ov::intel_vpux::output_fp16_to_fp32_host_conversion.name();
    }

    static bool defaultValue() {
        // TODO:
        // Windows-Yocto scenario (autonomous mode)
        // Hardcoded "true" might decrease a performance
        // Need to be investigated if we use it (currently it's unused)
        // [Track number: E#22196]
        // Windows dKMB scenario (discrete mode)
        // Hardcoded "true" brings some performance boost for some networks
        // Besides of that, there is CPU usage grow as well
        // Consider to keep "false" until further investigation is completed
#ifdef _WIN32
        return false;
#else
        return false;
#endif
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// COMPILATION_RESULTS_PATH
//

struct MCM_COMPILATION_RESULTS_PATH final : OptionBase<MCM_COMPILATION_RESULTS_PATH, std::string> {
    static StringRef key() {
        return ov::intel_vpux::compilation_results_path.name();
    }

    static std::string defaultValue() {
        return ".";
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// COMPILATION_RESULTS
//

struct MCM_COMPILATION_RESULTS final : OptionBase<MCM_COMPILATION_RESULTS, std::string> {
    static StringRef key() {
        return ov::intel_vpux::compilation_results.name();
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// GENERATE_BLOB
//

struct MCM_GENERATE_BLOB final : OptionBase<MCM_GENERATE_BLOB, bool> {
    static StringRef key() {
        return ov::intel_vpux::generate_blob.name();
    }

    static bool defaultValue() {
        return true;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// PARSING_ONLY
//

struct MCM_PARSING_ONLY final : OptionBase<MCM_PARSING_ONLY, bool> {
    static StringRef key() {
        return ov::intel_vpux::parsing_only.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// GENERATE_JSON
//

struct MCM_GENERATE_JSON final : OptionBase<MCM_GENERATE_JSON, bool> {
    static StringRef key() {
        return ov::intel_vpux::generate_json.name();
    }

    static bool defaultValue() {
        return true;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// GENERATE_DOT
//

struct MCM_GENERATE_DOT final : OptionBase<MCM_GENERATE_DOT, bool> {
    static StringRef key() {
        return ov::intel_vpux::generate_dot.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// LAYER_SPLIT_STRATEGIES
//

struct MCM_LAYER_SPLIT_STRATEGIES final : OptionBase<MCM_LAYER_SPLIT_STRATEGIES, std::string> {
    static StringRef key() {
        return ov::intel_vpux::layer_split_strategies.name();
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// LAYER_STREAM_STRATEGIES
//

struct MCM_LAYER_STREAM_STRATEGIES final : OptionBase<MCM_LAYER_STREAM_STRATEGIES, std::string> {
    static StringRef key() {
        return ov::intel_vpux::layer_stream_strategies.name();
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// LAYER_SPARSITY_STRATEGIES
//

struct MCM_LAYER_SPARSITY_STRATEGIES final : OptionBase<MCM_LAYER_SPARSITY_STRATEGIES, std::string> {
    static StringRef key() {
        return ov::intel_vpux::layer_sparsity_strategies.name();
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// LAYER_LOCATION_STRATEGIES
//

struct MCM_LAYER_LOCATION_STRATEGIES final : OptionBase<MCM_LAYER_LOCATION_STRATEGIES, std::string> {
    static StringRef key() {
        return ov::intel_vpux::layer_location_strategies.name();
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

}  // namespace vpux
