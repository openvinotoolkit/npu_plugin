// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_conv_utils.hpp"

#include <conv_ref.hpp>
#include <vpu_layers_tests.hpp>

#include "common_test_utils/common_layers_params.hpp"

// Wrappers are used because IE functions getConvWeightsSize and getConvBiasesByteSize
// support only 'FP32', 'FP16' and 'U8' precisions
size_t getConvWeightsByteSize(const std::vector<size_t>& inShape, const CommonTestUtils::conv_common_params& params,
    const std::string& precision) {
    return getConvWeightsSize(inShape, params, "U8") * precisionToBytesize(precision);
}

size_t getConvBiasesByteSize(const CommonTestUtils::conv_common_params& params, const std::string& precision) {
    return getConvBiasesSize(params, "U8") * precisionToBytesize(precision);
}

std::string instantiateConvTestIR(const convolution_test_desc& convTestParam) {
    std::string ir = convTestParam.ir;
    auto input_dims = convTestParam.input_dim;
    auto conv_params = convTestParam.conv_params;
    SizeVector output_dims;
    CommonTestUtils::getConvOutShape(input_dims, conv_params, output_dims);

    size_t weightsByteSize =
        getConvWeightsByteSize(convTestParam.input_dim, convTestParam.conv_params, convTestParam.weights_precision);
    size_t biasByteSize = getConvBiasesByteSize(convTestParam.conv_params, convTestParam.bias_precision);

    REPLACE_WITH_NUM(ir, "_INPUT_BATCH_", input_dims[0]);
    REPLACE_WITH_NUM(ir, "_INPUT_CHANNEL_", input_dims[1]);
    REPLACE_WITH_NUM(ir, "_INPUT_HEIGHT_", input_dims[2]);
    REPLACE_WITH_NUM(ir, "_INPUT_WIDTH_", input_dims[3]);

    REPLACE_WITH_STR(ir, "_NET_PRECISION_", convTestParam.net_precision);
    REPLACE_WITH_STR(ir, "_CONV_PRECISION_", convTestParam.conv_precision);
    REPLACE_WITH_STR(ir, "_WEIGHTS_PRECISION_", convTestParam.weights_precision);
    REPLACE_WITH_STR(ir, "_BIAS_PRECISION_", convTestParam.bias_precision);

    REPLACE_WITH_NUM(ir, "_WEIGHTS_OFFSET_", convTestParam.weightsBufferOffset);
    REPLACE_WITH_NUM(ir, "_WEIGHTS_BYTE_SIZE_", weightsByteSize);

    REPLACE_WITH_NUM(ir, "_BIAS_OFFSET_", convTestParam.weightsBufferOffset + weightsByteSize);
    REPLACE_WITH_NUM(ir, "_BIAS_BYTE_SIZE_", biasByteSize);

    REPLACE_WITH_NUM(ir, "_KERNEL_SIZE_", conv_params.kernel[0]);
    REPLACE_WITH_NUM_VECTOR(ir, "_KERNEL_", conv_params.kernel);
    REPLACE_WITH_NUM(ir, "_KERNELY_", conv_params.kernel[0]);
    REPLACE_WITH_NUM(ir, "_KERNELX_", conv_params.kernel[1]);
    REPLACE_WITH_NUM_VECTOR(ir, "_STRIDE_", conv_params.stride);
    REPLACE_WITH_NUM_VECTOR(ir, "_PADS_BEGIN_", conv_params.pads_begin);
    REPLACE_WITH_NUM_VECTOR(ir, "_PADS_END_", conv_params.pads_end);

    REPLACE_WITH_NUM(ir, "_OUTPUT_BATCH_", output_dims[0]);
    REPLACE_WITH_NUM(ir, "_OUTPUT_CHANNEL_", output_dims[1]);
    REPLACE_WITH_NUM(ir, "_OUTPUT_HEIGHT_", output_dims[2]);
    REPLACE_WITH_NUM(ir, "_OUTPUT_WIDTH_", output_dims[3]);

    REPLACE_WITH_NUM(ir, "_WEIGHTS_OFFSET_", convTestParam.weightsBufferOffset);
    REPLACE_WITH_NUM(ir, "_WEIGHTS_BYTE_SIZE_", weightsByteSize);

    REPLACE_WITH_NUM(ir, "_BIAS_OFFSET_", convTestParam.weightsBufferOffset + weightsByteSize);
    REPLACE_WITH_NUM(ir, "_BIAS_BYTE_SIZE_", biasByteSize);

    return ir;
}
