//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/al/config/mcm_compiler.hpp"

using namespace vpux;
using namespace ov::intel_vpux;

//
// register
//

void vpux::registerMcmCompilerOptions(OptionsDesc& desc) {
    desc.add<MCM_TARGET_DESCRIPTOR_PATH>();
    desc.add<MCM_TARGET_DESCRIPTOR>();
    desc.add<MCM_COMPILATION_DESCRIPTOR_PATH>();
    desc.add<MCM_COMPILATION_DESCRIPTOR>();
    desc.add<MCM_LOG_LEVEL>();
    desc.add<MCM_ELTWISE_SCALES_ALIGNMENT>();
    desc.add<MCM_CONCAT_SCALES_ALIGNMENT>();
    desc.add<MCM_WEIGHTS_ZERO_POINTS_ALIGNMENT>();
    desc.add<MCM_COMPILATION_PASS_BAN_LIST>();
    desc.add<MCM_SCALE_FUSE_INPUT>();
    desc.add<MCM_ALLOW_NCHW_MCM_INPUT>();
    desc.add<MCM_REMOVE_PERMUTE_NOOP>();
    desc.add<MCM_SERIALIZE_CNN_BEFORE_COMPILE_FILE>();
    desc.add<MCM_REFERENCE_MODE>();
    desc.add<MCM_ALLOW_U8_INPUT_FOR_FP16_MODELS>();
    desc.add<MCM_SCALESHIFT_FUSING>();
    desc.add<MCM_ALLOW_PERMUTE_ND>();
    desc.add<MCM_OPTIMIZE_INPUT_PRECISION>();
    desc.add<MCM_FORCE_PLUGIN_INPUT_QUANTIZATION>();
    desc.add<MCM_OUTPUT_FP16_TO_FP32_HOST_CONVERSION>();
    desc.add<MCM_COMPILATION_RESULTS_PATH>();
    desc.add<MCM_COMPILATION_RESULTS>();
    desc.add<MCM_GENERATE_BLOB>();
    desc.add<MCM_PARSING_ONLY>();
    desc.add<MCM_GENERATE_JSON>();
    desc.add<MCM_GENERATE_DOT>();
    desc.add<MCM_LAYER_SPLIT_STRATEGIES>();
    desc.add<MCM_LAYER_STREAM_STRATEGIES>();
    desc.add<MCM_LAYER_SPARSITY_STRATEGIES>();
    desc.add<MCM_LAYER_LOCATION_STRATEGIES>();
}
