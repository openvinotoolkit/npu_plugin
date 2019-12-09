// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <ie_common.h>
#include <ie_layers.h>
#include <precision_utils.h>

#include <common_layers_params.hpp>
#include <ie_icnn_network_stats.hpp>
#include <ie_util_internal.hpp>
#include <vpu/kmb_plugin_config.hpp>

#include "conv_ref.hpp"

using namespace InferenceEngine;

struct convolution_test_desc {
    InferenceEngine::SizeVector input_dim;
    conv_common_params conv_params;
    std::string net_precision;
    std::string conv_precision;
    std::string weights_precision;
    size_t weightsBufferOffset;
    std::string bias_precision;
    std::string test_name;
    std::string& ir;
};

// Wrappers are used because IE functions getConvWeightsSize and getConvBiasesByteSize
// support only 'FP32', 'FP16' and 'U8' precisions
size_t getConvWeightsByteSize(
    const std::vector<size_t>& inShape, const conv_common_params& params, const std::string& precision);

size_t getConvBiasesByteSize(const conv_common_params& params, const std::string& precision);

std::string instantiateConvTestIR(const convolution_test_desc& convTestParam);

template <typename T>
void fillDiagonalKernel(Blob::Ptr kernelWeightsBlob, T value) {
    SizeVector& kernelDims = kernelWeightsBlob->getTensorDesc().getDims();
    T* kernelWeightsData = kernelWeightsBlob->buffer().as<T*>();
    const int kernelCubeSize = kernelDims[1] * kernelDims[2] * kernelDims[3];
    // size of the unary tensor is W*H*N^2, since #OC == #IC
    for (size_t n = 0; n < kernelDims[0]; ++n) {  // go through output channels
        // for square HW planes:
        IE_ASSERT(kernelCubeSize * n + n < kernelWeightsBlob->size());
        kernelWeightsData[kernelCubeSize * n + n] = static_cast<T>(value);
    }
}

template <class srcType, class dstType>
static void testOverflow(const Blob::Ptr& blob) {
    auto data = blob->buffer().as<srcType*>();
    auto maxValue = std::numeric_limits<dstType>::max();
    auto minValue = std::numeric_limits<dstType>::min();
    for (size_t i = 0; i < blob->size(); ++i) {
        if (data[i] < minValue || maxValue < data[i]) {
            THROW_IE_EXCEPTION << "Blob contains value " << data[i] << " that exceeds desired range [" << minValue
                               << ", " << maxValue << "]";
        }
    }
}

// ref_conv_common copy with explicit precision for source and destination
template <typename wei_data_t, typename bias_data_t>
void ref_conv_common_prec(const std::vector<InferenceEngine::Blob::Ptr> srcs, Blob& dst, const wei_data_t* weights_data,
    size_t weights_size, const bias_data_t* bias_data, size_t bias_size, const conv_common_params& prm,
    Precision precision) {
    if (srcs[0]->getTensorDesc().getLayout() != Layout::NCHW && srcs[0]->getTensorDesc().getLayout() != Layout::NCDHW)
        THROW_IE_EXCEPTION << "Reference FP32 convolution supports NCHW and NCDHW layouts only";
    size_t KW = prm.kernel[X_AXIS];
    size_t KH = prm.kernel[Y_AXIS];
    size_t KD = prm.kernel.size() > Z_AXIS ? prm.kernel[Z_AXIS] : 1lu;

    size_t SW = prm.stride[X_AXIS];
    size_t SH = prm.stride[Y_AXIS];
    size_t SD = prm.stride.size() > Z_AXIS ? prm.stride[Z_AXIS] : 0lu;

    size_t DW = prm.dilation[X_AXIS];
    size_t DH = prm.dilation[Y_AXIS];
    size_t DD = prm.dilation.size() > Z_AXIS ? prm.dilation[Z_AXIS] : 0lu;

    size_t PW = prm.pads_begin[X_AXIS];
    size_t PH = prm.pads_begin[Y_AXIS];
    size_t PD = prm.pads_begin.size() > Z_AXIS ? prm.pads_begin[Z_AXIS] : 0lu;

    size_t GC = prm.group;

    auto src_dims = srcs[0]->getTensorDesc().getDims();
    size_t IC = src_dims[1];
    size_t ID = (src_dims.size() == 5lu) ? src_dims[2] : 1lu;
    size_t IH = src_dims.at(src_dims.size() - 2);
    size_t IW = src_dims.back();

    auto dst_dims = dst.getTensorDesc().getDims();
    size_t OW = dst_dims.back();
    size_t OH = dst_dims.at(dst_dims.size() - 2);
    size_t OD = (dst_dims.size() == 5lu) ? dst_dims[2] : 1lu;
    size_t OC = prm.out_c;

    const auto src_buffer = srcs[0]->cbuffer();
    auto* dst_dataFP = dst.buffer().as<float*>();
    auto* dst_dataU8 = dst.buffer().as<uint8_t*>();
    auto* dst_dataI8 = dst.buffer().as<int8_t*>();

    IE_ASSERT(KW * KH * KD * OC * IC / GC == weights_size);
    IE_ASSERT(OC == bias_size);

    for (uint32_t g = 0; g < GC; g++) {
        for (uint32_t oc = 0; oc < OC / GC; oc++) {
            for (uint32_t od = 0; od < OD; od++) {
                for (uint32_t oh = 0; oh < OH; oh++) {
                    for (uint32_t ow = 0; ow < OW; ow++) {
                        size_t oidx = g * OC / GC * OD * OH * OW + oc * OD * OH * OW + od * OH * OW + oh * OW + ow;
                        if (bias_data) {
                            if (precision == Precision::U8) {
                                dst_dataU8[oidx] = bias_data[g * OC / GC + oc];
                            } else if (precision == Precision::I8) {
                                dst_dataI8[oidx] = bias_data[g * OC / GC + oc];
                            } else {
                                dst_dataFP[oidx] = bias_data[g * OC / GC + oc];
                            }
                        }
                        for (size_t ic = 0; ic < IC / GC; ic++) {
                            for (size_t kd = 0; kd < KD; kd++) {
                                for (size_t kh = 0; kh < KH; kh++) {
                                    for (size_t kw = 0; kw < KW; kw++) {
                                        int32_t iw = ow * SW - PW + kw * DW;
                                        int32_t ih = oh * SH - PH + kh * DH;
                                        int32_t id = od * SD - PD + kd * DD;
                                        if (iw < 0 || iw >= (int32_t)IW || ih < 0 || ih >= (int32_t)IH || id < 0 ||
                                            id >= (int32_t)ID)
                                            continue;
                                        size_t iidx = g * IC / GC * ID * IH * IW + ic * ID * IH * IW + id * IH * IW +
                                                      ih * IW + iw;
                                        size_t widx = g * OC / GC * IC / GC * KD * KH * KW +
                                                      oc * IC / GC * KD * KH * KW + ic * KD * KH * KW + kd * KH * KW +
                                                      kh * KW + kw;

                                        if (precision == Precision::U8) {
                                            dst_dataU8[oidx] +=
                                                (src_buffer.as<const uint8_t*>())[iidx] * weights_data[widx];
                                        } else if (precision == Precision::I8) {
                                            dst_dataI8[oidx] +=
                                                (src_buffer.as<const int8_t*>())[iidx] * weights_data[widx];
                                        } else {
                                            dst_dataFP[oidx] +=
                                                (src_buffer.as<const float*>())[iidx] * weights_data[widx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// CompareCommonExact copy interpreting data as template parameter
template <typename T, typename = std::enable_if<std::is_pointer<T>::value>>
void CompareCommonExactTyped(const InferenceEngine::Blob::Ptr& actual, const InferenceEngine::Blob::Ptr& expected) {
    ASSERT_NE(actual, nullptr);
    ASSERT_NE(expected, nullptr);
    const T* res_ptr = actual->cbuffer().as<const T*>();
    const T* ref_ptr = expected->cbuffer().as<const T*>();
    bool differ = false;
    size_t actualFirstErrIdx = 0;
    size_t expectedFirstErrIdx = 0;
    std::function<void(size_t, size_t)> exactErrorUpdater = [&](size_t actualIdx, size_t expectedIdx) {
        auto actual = res_ptr[actualIdx];
        auto expected = ref_ptr[expectedIdx];
        if ((actual != expected) && !differ) {
            actualFirstErrIdx = actualIdx;
            expectedFirstErrIdx = expectedIdx;
            differ = true;
        }
    };
    CompareCommon(actual, expected, exactErrorUpdater);
    ASSERT_EQ(differ, false) << "expectedFirstErrIdx = " << expectedFirstErrIdx
                             << " actualFirstErrIdx = " << actualFirstErrIdx;
}
