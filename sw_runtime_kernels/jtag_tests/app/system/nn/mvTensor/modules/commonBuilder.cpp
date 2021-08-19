// {% copyright %}

#include "commonBuilder.hpp"
#include "mvTensorUtil.h"
#include "mvTensorDebug.h"
#include <iostream>
#include <vector>
#include <mvSubspaces.h>

MVCNN::DType CommonFBFuilder::buildDtype(t_MvTensorDataType type) {
    switch (type) {
        case t_fp16:
            return MVCNN::DType_FP16;
        case t_u8f:
            return MVCNN::DType_U8;
        case t_int:
            return MVCNN::DType_I32;
        case t_fp32:
            return MVCNN::DType_FP32;
        case t_i8 :
            return MVCNN::DType_I8;
        default:
            return MVCNN::DType_NOT_SET;
    }
}

uint32_t CommonFBFuilder::buildAxisIndex(int memAxisInd, t_MvTensorStorageOrder order) {

    mvTensorAssert(memAxisInd >= 0 || memAxisInd < subspace::MAX_DIMS, "memAxisIndex should be >= 0 and < subspace::MAX_DIMS");

    int32_t inPerm[subspace::MAX_DIMS] = {};
    auto ndims = orderToPermutation(order, inPerm);

    return ndims - 1 - inPerm[memAxisInd];
}

uint64_t orderReverseDigits(uint32_t order) {
    int32_t perm[subspace::MAX_DIMS] = {};
    int ndims = subspace::orderToPermutation(order, perm);
    auto minPtr = std::min_element(perm, perm + ndims);
    auto maxPtr = std::max_element(perm, perm + ndims);
    int32_t maxOldDimNum = *maxPtr;
    for (int i = 0; i < ndims; i++) {
        minPtr = std::min_element(perm, perm + ndims);
        *minPtr = 3 * maxOldDimNum + i;
    }
    int32_t maxDimNum = *minPtr;
    for (int i = 0; i < ndims; i++) {
        perm[i] = maxDimNum - perm[i];
    }

    uint32_t newOrder = permutationToOrder(perm, ndims);

    return newOrder;
}

std::unique_ptr<MVCNN::TensorReferenceT> CommonFBFuilder::buildTensorReferenceT(const Buffer &b) {
    std::unique_ptr<MVCNN::TensorReferenceT> toBuild = std::unique_ptr<MVCNN::TensorReferenceT>(new MVCNN::TensorReferenceT());

    int32_t permutation[subspace::MAX_DIMS] = {};
    subspace::orderToPermutation(b.order, permutation);

    auto max = std::max_element(std::begin(permutation), std::end(permutation));

    std::vector<uint32_t> dimensions(*max + 1, 0);
    std::vector<float> numericStrides(*max + 1, 0);

    for (int i = 0; i < b.ndims; i++) {
        auto dim = b.dims[i];
        auto stride = b.strides[i];
        dimensions[permutation[i]] = dim;
        numericStrides[permutation[i]] = stride;
    }
    for (int i = *max - 1; i >= 0; i--) {
        if (dimensions[i] <= 0) {
            dimensions.erase(dimensions.begin() + i);
            numericStrides.erase(numericStrides.begin() + i);
        }
    }
    // system/nn/blob/2490/schema/src/schema/memoryManagement.fbs:192 (TensorReference::stride)
    // The first value of stride is always the stride of a single element.
    // This is usually the size of the datatype.
    numericStrides.push_back(b.strides[0]);

    std::reverse(dimensions.begin(), dimensions.end());
    std::reverse(numericStrides.begin(), numericStrides.end());

    toBuild->dimensions = dimensions;
    toBuild->strides = numericStrides;

    toBuild->data = std::unique_ptr<MVCNN::IndirectDataReferenceT>(new MVCNN::IndirectDataReferenceT());

    toBuild->data->data_index = 0;
    toBuild->locale_index = std::vector<long unsigned int>(1,0);

    toBuild->locale = MVCNN::MemoryLocation::MemoryLocation_VPU_DDR_BSS;
    toBuild->data_dtype = buildDtype(static_cast<t_MvTensorDataType>(b.dType));

    toBuild->order = orderReverseDigits(b.order);

    return toBuild;
}
