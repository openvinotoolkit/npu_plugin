//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <mvSubspaces.h>
#include <limits.h>
#include "common_types.h"
#include <string.h>

namespace nn {

/**
 * Tensor utility functions and the tensor descriptor class itself.
 * All manipulation and access of tensor data comes from this lib.
 * This lib is use in the Inference Runtime, it therefore may not
 * have a dependency on FlatBuffer libs.
 */

using namespace subspace;

uint32_t getBpp(DataType type);

struct TensorRefNDData
{
    DataType dType = NN_FP16;
    NDOrder ndOrder = ND_NHWC;
    uint8_t* addr = nullptr;
    int32_t ndims = 0;
    int32_t dims[MAX_ND_DIMS]{};
    int32_t strides[MAX_ND_DIMS]{};
    int64_t stridesBits[MAX_ND_DIMS]{};
    int32_t bitsPerPixel = 16;
};

static inline bool fillTensorRefNDData(TensorRefNDData& dstTensor,
        int32_t dims[],
        int32_t strides[],
        DataType dType,
        NDOrder ndOrder,
        int32_t bitsPP,
        uint8_t* addr = nullptr) {
    bool success = false;
    NDDims perm = orderNDToPermutation(ndOrder, success);
    int prevStride = 1;
    for (int i = 0; i < perm.ndims(); i++) {
        if (strides[i] < prevStride) return false;
        prevStride = strides[i];
    }
    dstTensor.dType = dType;
    dstTensor.ndOrder = ndOrder;
    dstTensor.ndims = perm.ndims();
    dstTensor.addr = addr;
    dstTensor.bitsPerPixel = bitsPP;
    for (int i = 0; i < perm.ndims(); i++) {
        dstTensor.dims[i] = dims[i];
        dstTensor.strides[i] = strides[i];
    }
    return true;
}

// Arguments:
//     * buffer - TensorRefNDData to be checked
//       upToDim - check only 'upToDim' lower dimensions
// Returns true if data is continuous (gapless) in memory, false otherwise.

bool isContinuous(const TensorRefNDData& buffer, int32_t upToDim = MAX_ND_DIMS);

//
// isEmpty checks whether a buffer has no present data
//
// Arguments:
//     * buffer - TensorRefNDData to be checked
// Returns true if data is not present in memory, false otherwise.
//
bool isEmpty(const TensorRefNDData& buffer);

//
// element computes a pointer to buffer element, pointed by its indices set
//
// Arguments:
//     * buffer  - TensorRefNDData to be considered (in)
//     * indices - an index set to be applied
// Returns pointer to indexed buffer element.
//
void* element(const TensorRefNDData& buffer, const int32_t indices[]);

//
// dim/stride returns a dimension/stride value, explicit or default (if absent)
//
// Arguments:
//     * tensor        - TensorRefNDData to be considered (in)
//     * d             - logical dimension index
//     * default_value - default value returned if 'd'th dim is absent in storage order
// Returns dimension size, explicit or default.
//
int32_t dim(const TensorRefNDData& tensor, subspace::LogDimIndex logDim, int32_t default_value = -1);

bool isNDDataLayoutFit(NDOrder ndOrder, const nn::TensorRefNDData& tensor);

class TensorRef : public TensorRefNDData {
public:
    TensorRef() = default;
    TensorRef(TensorRefNDData data) : TensorRefNDData(data) {
    };
    TensorRef(const sw_params::MemRefData& data) {
        dType = static_cast<sw_params::DataType>(data.dataType);
        ndOrder = data.dimsOrder;
        addr = reinterpret_cast<uint8_t *>(data.dataAddr);
        ndims = (data.numDims > (uint32_t)MAX_ND_DIMS) ? (uint32_t)MAX_ND_DIMS : data.numDims;
        memcpy_s(dims, ndims * sizeof(int32_t), reinterpret_cast<uint8_t *>(data.dimsAddr), ndims * sizeof(int32_t));
        memcpy_s(stridesBits, ndims * sizeof(int64_t), reinterpret_cast<uint8_t *>(data.stridesAddr), ndims * sizeof(int64_t));
        for (int i = 0; i < ndims; i++) {
            strides[i] = stridesBits[i] / CHAR_BIT;
        }
        bitsPerPixel = nn::getBpp(dType) * CHAR_BIT;
    };

    unsigned int getDataSize() const;
    unsigned int getFullDataSize() const;
    unsigned int getNumElements() const;

    // Helper method to decode order, dims and strides of TensorRefND
    // in accordance with sequence of dimensions in memory using
    // newDims and newStrides arrays ordered as [W, H, C, N, ...]
    // newStrides array has special format:
    // newStrides[0] = <stride of inner tensor dimension (stride to the next 'scalar' tensor element)>
    // newStride[i + 1] stride for the i-th tensor dimension in order of [W, H, C, N, ...]
    //
    // baseLineFullOrder is default order in which the dimensions will set when several dimensions match.
    // Make sense when several dimension are equal 1
    bool setByStrides(DataType dataType, const long unsigned int newDims[],
         const long unsigned int newStrides[], const uint64_t newStridesBits[],
         int dimensionality, NDOrder baseLineFullOrder = FULL_ND_NHWC);
    bool set(void* addr, uint32_t dataType, NDOrder order, const int32_t dims[], const int32_t strides[]);

    int32_t getDim(subspace::LogDimIndex dim, int32_t defaultValue = -1) const;
    int32_t getStride(subspace::LogDimIndex dim, int32_t defaultValue = -1) const;
    int64_t getStrideBits(subspace::LogDimIndex dim, int64_t defaultValue = -1) const;

    //  Attention: the following methods dimN..strideW..strideBitsW make sense only in specific cases, mainly 4D and 3D.
    //  Please avoid using them. They should be eliminated soon.
    //  Use getDim, getStride instead.
    int dimN(int defaultValue = -1) const;
    int dimC(int defaultValue = -1) const;
    int dimH(int defaultValue = -1) const;
    int dimW(int defaultValue = -1) const;

    int strideN(int defaultValue = -1) const;
    int strideC(int defaultValue = -1) const;
    int strideH(int defaultValue = -1) const;
    int strideW(int defaultValue = -1) const;

    int64_t strideBitsN(int64_t defaultValue = -1) const;
    int64_t strideBitsC(int64_t defaultValue = -1) const;
    int64_t strideBitsH(int64_t defaultValue = -1) const;
    int64_t strideBitsW(int64_t defaultValue = -1) const;
    subspace::LogDimIndex getLogicalC() const;
    sw_params::MemRefData toMemRefData(sw_params::Location loc = sw_params::Location::DDR, bool doCopy = false) const;
};

} // namespace nn
