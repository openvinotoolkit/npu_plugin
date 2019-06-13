// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <inference_engine/precision_utils.h>
#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"

using namespace InferenceEngine;

int getOffset(const SizeVector& coordinates, const SizeVector& strides) {
    int offset = 0;
    for(int i = 0; i < coordinates.size(); ++i) {
        offset += coordinates[i] * strides[i];
    }
    return offset;
}

void incrementCoordinates(SizeVector& coordinates, const SizeVector& dims) {
    for(int d = 0, nAdd = 1; d < coordinates.size() && nAdd == 1 ; ++d)
    {
        coordinates[d] = (coordinates[d] == dims[d] - 1) ? 0 : coordinates[d] + 1;
        nAdd = (coordinates[d] == 0) ? 1 : 0;
    }
}

void ref_softMax(const Blob::Ptr& src, Blob::Ptr& dst, int axis) {
    SizeVector tensorSizes = src->dims();
    SizeVector tensorStrides(tensorSizes.size());
    axis = tensorSizes.size() - 1 - axis;
    const ie_fp16 *src_data = src->cbuffer().as<const ie_fp16*>();
    ie_fp16 *dst_data = dst->buffer().as<ie_fp16*>();
    const ie_fp16 *srcLine;
    ie_fp16 *dstLine;

    size_t totalElements = 1;
    size_t totalLines = 1;

    for (int i = 0; i < tensorSizes.size(); ++i) {
        tensorStrides[i] = totalElements;
        totalElements *= tensorSizes[i];
    }
    size_t axisSize = tensorSizes[axis];
    size_t axisStride = tensorStrides[axis];
    tensorSizes.erase(tensorSizes.begin() + axis);
    tensorStrides.erase(tensorStrides.begin() + axis);
    totalLines = totalElements / axisSize;

    std::vector<float> temp(axisSize);

    SizeVector tensorCoordinates(tensorSizes.size());

    for (int nLine = 0; nLine < totalLines; ++nLine) {
        int offset = getOffset(tensorCoordinates, tensorStrides);

        srcLine = src_data + offset;
        dstLine = dst_data + offset;
        float largest = std::numeric_limits<float>::lowest();
        for (int i2 = 0; i2 < axisSize; ++i2) {
            int ind = i2 * axisStride;
            float val = PrecisionUtils::f16tof32(srcLine[ind]);
            largest = std::max(val, largest);
        }
        float sum = 0.0f;
        for (int i2 = 0; i2 < axisSize; ++i2) {
            int ind = i2 * axisStride;
            float val = PrecisionUtils::f16tof32(srcLine[ind]);
            temp[i2] = std::exp(val - largest);
            sum += temp[i2];
        }
        for (int i2 = 0; i2 < axisSize; ++i2) {
            int ind = i2 * axisStride;
            dstLine[ind] = PrecisionUtils::f32tof16(temp[i2] / sum);
        }
        incrementCoordinates(tensorCoordinates, tensorSizes);
    }
}
