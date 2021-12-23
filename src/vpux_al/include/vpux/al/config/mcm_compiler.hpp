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
#include "vpux/vpux_compiler_config.hpp"

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
        return VPU_COMPILER_CONFIG_KEY(TARGET_DESCRIPTOR_PATH);
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
        return VPU_COMPILER_CONFIG_KEY(TARGET_DESCRIPTOR);
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
        return VPU_COMPILER_CONFIG_KEY(COMPILATION_DESCRIPTOR_PATH);
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
        return VPU_COMPILER_CONFIG_KEY(COMPILATION_DESCRIPTOR);
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
        return VPU_COMPILER_CONFIG_KEY(LOG_LEVEL);
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
        return VPU_COMPILER_CONFIG_KEY(ELTWISE_SCALES_ALIGNMENT);
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
        return VPU_COMPILER_CONFIG_KEY(CONCAT_SCALES_ALIGNMENT);
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
        return VPU_COMPILER_CONFIG_KEY(WEIGHTS_ZERO_POINTS_ALIGNMENT);
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
        return VPU_COMPILER_CONFIG_KEY(COMPILATION_PASS_BAN_LIST);
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
        return VPU_COMPILER_CONFIG_KEY(SCALE_FUSE_INPUT);
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
        return VPU_COMPILER_CONFIG_KEY(ALLOW_NCHW_MCM_INPUT);
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
        return VPU_COMPILER_CONFIG_KEY(REMOVE_PERMUTE_NOOP);
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
        return VPU_COMPILER_CONFIG_KEY(SERIALIZE_CNN_BEFORE_COMPILE_FILE);
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
        return VPU_COMPILER_CONFIG_KEY(REFERENCE_MODE);
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
        return VPU_COMPILER_CONFIG_KEY(ALLOW_U8_INPUT_FOR_FP16_MODELS);
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
        return VPU_COMPILER_CONFIG_KEY(SCALESHIFT_FUSING);
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
        return VPU_COMPILER_CONFIG_KEY(ALLOW_PERMUTE_ND);
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
        return VPU_COMPILER_CONFIG_KEY(OPTIMIZE_INPUT_PRECISION);
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
        return VPU_COMPILER_CONFIG_KEY(FORCE_PLUGIN_INPUT_QUANTIZATION);
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
        return VPU_COMPILER_CONFIG_KEY(OUTPUT_FP16_TO_FP32_HOST_CONVERSION);
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
        return VPU_COMPILER_CONFIG_KEY(COMPILATION_RESULTS_PATH);
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
        return VPU_COMPILER_CONFIG_KEY(COMPILATION_RESULTS);
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
        return VPU_COMPILER_CONFIG_KEY(GENERATE_BLOB);
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
        return VPU_COMPILER_CONFIG_KEY(PARSING_ONLY);
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
        return VPU_COMPILER_CONFIG_KEY(GENERATE_JSON);
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
        return VPU_COMPILER_CONFIG_KEY(GENERATE_DOT);
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
        return VPU_COMPILER_CONFIG_KEY(LAYER_SPLIT_STRATEGIES);
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
        return VPU_COMPILER_CONFIG_KEY(LAYER_STREAM_STRATEGIES);
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
        return VPU_COMPILER_CONFIG_KEY(LAYER_SPARSITY_STRATEGIES);
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
        return VPU_COMPILER_CONFIG_KEY(LAYER_LOCATION_STRATEGIES);
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

}  // namespace vpux
