// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <inference_engine/precision_utils.h>
#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"
#include <math.h>

using namespace InferenceEngine;

void ref_RegionYolo_wrap(InferenceEngine::Blob::Ptr inTensor,
              InferenceEngine::Blob::Ptr outTensor,
              const ParamsStruct& params) {

    ASSERT_FALSE(params.empty());
    /* default parameters */
    int coords  = 4;
    int classes = 20;
    int num     = 5;

    auto iter = params.find("coords");
    if (iter != params.end()) {
        coords = std::stoi(iter->second);
    }
    iter = params.find("classes");
    if (iter != params.end()) {
        classes = std::stoi(iter->second);
    }
    iter = params.find("num");
    if (iter != params.end()) {
        num = std::stoi(iter->second);
    }
    ref_RegionYolo(inTensor, outTensor, coords, classes, num);
}

static int entry_index(int w, int h, int outputs, int coords_classes, int batch, int location, int entry)
{
    int n = location / (w * h);
    int loc = location % (w * h);
    return batch * outputs + n * w * h * coords_classes + entry * w * h + loc;
}

static inline uint16_t logistic_activate(float x)
{
    float res = 1./(1. + exp(-x));
    return PrecisionUtils::f32tof16(res);
}

static void activate_array(uint16_t *x, const int n)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = logistic_activate(PrecisionUtils::f16tof32(x[i]));
    }
}

static void softmax_FP16(const uint16_t *input, int n,
                    float temp, int stride,
                    uint16_t *output)
{
    int i;
    float sum = 0;
    float largest = -100.0;
    std::vector<float> data(n);
    for(i = 0; i < n; ++i){
        data[i] = PrecisionUtils::f16tof32(input[i*stride]);
        if(data[i] > largest) 
            largest = data[i];
    }
    for(i = 0; i < n; ++i){
        float e = exp(data[i]/temp - largest/temp);
        sum += e;
        data[i] = e;
    }
    for(i = 0; i < n; ++i){
        float tmp = data[i];
        tmp /= sum;
        output[i*stride] = PrecisionUtils::f32tof16(tmp);
    }
}

static void softmax_cpu_FP16(const uint16_t *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, uint16_t *output)
{
    int g, b;
    for(b = 0; b < batch; ++b){
        for(g = 0; g < groups; ++g){
            softmax_FP16(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}

void ref_RegionYolo(const InferenceEngine::Blob::Ptr src,
                    InferenceEngine::Blob::Ptr dst,
                    int coords,
                    int classes,
                    int num) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    uint16_t *srcData = src->buffer();
    uint16_t *dstData = dst->buffer();
    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(dstData, nullptr);
    auto dims = src->dims();
    ASSERT_EQ(src->precision(), InferenceEngine::Precision::FP16);
    ASSERT_EQ(dst->precision(), InferenceEngine::Precision::FP16);
    int iw = dims[0];
    int ih = dims[1];
    int ic = dims[2];
    std::vector<uint16_t> ref_data(src->size());
    uint16_t* inputBlobRawDataFp16 = ref_data.data();
    switch(src->layout()) {
    case InferenceEngine::NCHW:
        std::memcpy(ref_data.data(), srcData, src->size() * sizeof(uint16_t));
        break;
    case InferenceEngine::NHWC:
        for (int h = 0 ; h < ih; ++h) {
            for (int w = 0 ; w < iw; ++w) {
                for (int c = 0 ; c < ic; ++c) {
                    int dst_i = w + iw * h + iw * ih * c;
                    int src_i = c + ic * w + iw * ic * h;
                    inputBlobRawDataFp16[dst_i] = srcData[src_i];
                }
            }
        }
        break;
    }
    std::memcpy(dstData, ref_data.data(), src->size() * sizeof(uint16_t));

    int coords_classes = coords + classes + 1;
    int batch = 1;
    int outputs = num * ih * iw * coords_classes;
    int inWidth = iw;
    int inHeight = ih;
    for (int b = 0; b < batch; ++b) {
        for(int n = 0; n < num; ++n) {
            int index = entry_index(inWidth, inHeight, outputs, coords_classes, b, n * inWidth * inHeight, 0);
            activate_array(dstData + index, 2 * inWidth * inHeight);
            index = entry_index(inWidth, inHeight, outputs, coords_classes, b, n * inHeight * inWidth, coords);
            activate_array(dstData + index, inWidth * inHeight);
        }
    }
    int index = entry_index(inWidth, inHeight, outputs, coords_classes, 0, 0, coords + 1);
    softmax_cpu_FP16(inputBlobRawDataFp16 + index, classes + 0, batch * num, outputs / num, inHeight * inWidth, 1, inHeight * inWidth, 1, dstData + index);
}
