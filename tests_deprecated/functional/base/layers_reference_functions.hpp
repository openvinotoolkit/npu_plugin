// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <inference_engine.hpp>

typedef std::map<std::string, std::string> ParamsStruct;

typedef float (*eltwise_kernel)(float a, float b);

struct param_size {
    size_t x;
    size_t y;
};

struct paddings4 {
    size_t left;
    size_t top;
    size_t right;
    size_t bottom;
};

struct tensor_test_params {
    size_t n;
    size_t c;
    size_t h;
    size_t w;
    friend std::ostream& operator<<(std::ostream& os, tensor_test_params const& tst) {
        return os << "tensor (" << tst.n << ", " << tst.c << ", " << tst.h << ", " << tst.w << ")";
    };
};

/* Wrappers to gen subnets:
 reference function signature should have following structure:
    input blob,
    output blob,
    pointer to weights (if they are required)
    weights number (if pointer to weights is set)
    bias number (if pointer to weights is set)
    other parameters

*/
static inline void PrintTo(const param_size& sz, std::ostream* os) {
    *os << "{" << std::setw(2) << sz.x << ", " << std::setw(2) << sz.y << "}";
};

void ref_innerproduct_wrap(const InferenceEngine::Blob::Ptr src, InferenceEngine::Blob::Ptr dst,
    const uint16_t* weights, size_t weightsSize, size_t biasSize, const ParamsStruct& params);

void ref_ReLU_wrap(
    const InferenceEngine::Blob::Ptr inTensor, InferenceEngine::Blob::Ptr outTensor, const ParamsStruct& params);

void ref_Clamp_wrap(
    const InferenceEngine::Blob::Ptr inTensor, InferenceEngine::Blob::Ptr outTensor, const ParamsStruct& params);

void ref_pooling_wrap(const InferenceEngine::Blob::Ptr src, InferenceEngine::Blob::Ptr dst, const ParamsStruct& params);

void ref_copy_wrap(const InferenceEngine::Blob::Ptr src, InferenceEngine::Blob::Ptr dst, const ParamsStruct& params);

void ref_convolution_wrap(const InferenceEngine::Blob::Ptr src, InferenceEngine::Blob::Ptr dst, const uint16_t* weights,
    size_t weightsSize, size_t biasSize, const ParamsStruct& params);

void ref_deconvolution_wrap(const InferenceEngine::Blob::Ptr src, InferenceEngine::Blob::Ptr dst,
    const uint16_t* weights, size_t weightsSize, size_t biasSize, const ParamsStruct& params);

void ref_tanh_wrap(
    const InferenceEngine::Blob::Ptr inTensor, InferenceEngine::Blob::Ptr outTensor, const ParamsStruct& params);

void ref_sigmoid_wrap(const InferenceEngine::Blob::Ptr src, InferenceEngine::Blob::Ptr dst, const ParamsStruct& params);

void ref_PReLU_wrap(const InferenceEngine::Blob::Ptr src, InferenceEngine::Blob::Ptr dst, const uint16_t* weights,
    size_t weightsSize, size_t biasSize, const ParamsStruct& params);

void ref_RegionYolo_wrap(
    InferenceEngine::Blob::Ptr inTensor, InferenceEngine::Blob::Ptr outTensor, const ParamsStruct& params);

void ref_reshape_wrap(const InferenceEngine::Blob::Ptr src, InferenceEngine::Blob::Ptr dst, const ParamsStruct& params);

void ref_permute_wrap(const InferenceEngine::Blob::Ptr src, InferenceEngine::Blob::Ptr dst, const ParamsStruct& params);

/* Original functions*/

void ref_innerproduct(const InferenceEngine::Blob::Ptr src, InferenceEngine::Blob::Ptr dst, const uint16_t* weights,
    size_t weightsSize, size_t biasSize, uint32_t OC);

void ref_convolution(const InferenceEngine::Blob::Ptr src, InferenceEngine::Blob::Ptr dst,
    const InferenceEngine::ie_fp16* weights_data, const InferenceEngine::ie_fp16* bias_data, param_size kernel,
    param_size stride, param_size pad, size_t group, param_size dilation = {1, 1});

void ref_maxPooling(const InferenceEngine::Blob::Ptr src, InferenceEngine::Blob::Ptr dst, param_size kernel,
    param_size stride, param_size pad, bool exclude_pad = false);

void ref_avgPooling(const InferenceEngine::Blob::Ptr src, InferenceEngine::Blob::Ptr dst, param_size kernel,
    param_size stride, param_size pad, bool exclude_pad = false);

void ref_ReLU(const InferenceEngine::Blob::Ptr inTensor, InferenceEngine::Blob::Ptr outTensor, float negative_slope);

void ref_copy(const InferenceEngine::Blob::Ptr src, InferenceEngine::Blob::Ptr dst);

void ref_tanh(const InferenceEngine::Blob::Ptr src, InferenceEngine::Blob::Ptr dst);

void ref_sigmoid(const InferenceEngine::Blob::Ptr src, InferenceEngine::Blob::Ptr dst);

void ref_PReLU(
    const InferenceEngine::Blob::Ptr src, InferenceEngine::Blob::Ptr dst, const uint16_t* weights, size_t weightsSize);

void ref_eltwise(const InferenceEngine::Blob::Ptr src1, const InferenceEngine::Blob::Ptr src2,
    InferenceEngine::Blob::Ptr dst, eltwise_kernel fun, std::vector<float> coeff);

void ref_RegionYolo(
    const InferenceEngine::Blob::Ptr src, InferenceEngine::Blob::Ptr dst, int coords, int classes, int num);

void ref_Permute(const InferenceEngine::Blob::Ptr src, InferenceEngine::Blob::Ptr dst, const int* order_hwc);

void ref_softMax(const InferenceEngine::Blob::Ptr& src, InferenceEngine::Blob::Ptr& dst, int axis);
void ref_reshape(const InferenceEngine::Blob::Ptr src, InferenceEngine::Blob::Ptr dst);

void ref_Clamp(const InferenceEngine::Blob::Ptr inTensor, InferenceEngine::Blob::Ptr outTensor, float min, float max);

static constexpr char const PRELU_PARAM[] = "channel_shared";
