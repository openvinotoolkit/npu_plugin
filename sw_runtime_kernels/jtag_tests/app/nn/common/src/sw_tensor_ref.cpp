//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "sw_tensor_ref.h"
#include <stdint.h>

#ifdef CONFIG_TARGET_SOC_3720
extern unsigned char actShaveData[];
extern unsigned int actShaveDataReserved;
#include "nn_math.h"
#include "nn_memory.h"
#include "sw_shave_lib_common.h"
#include <dma_shave_nn.h>
#include <nn_cache.h>
#endif

namespace {

    int64_t strideBits(const nn::TensorRefNDData& tensor, subspace::LogDimIndex logDim, int64_t default_value) {
        bool success = false;
        nn::NDDims indices = subspace::orderNDToIndices(tensor.ndOrder, success);
        if (!success || indices.getElement(logDim, -1) == -1) {
            return default_value;
        } else {
            return tensor.stridesBits[indices[logDim]];
        }
    }
}

namespace nn {
using namespace subspace;

u32 getBpp(DataType type) {
    u32 bpp = 0;
    switch (type) {
        case NN_INT16:
        case NN_FP16:
            bpp = 2;
            break;

        case NN_U8:
        case NN_I8:
            bpp = 1;
            break;

        case NN_INT32:
        case NN_FP32:
            bpp = 4;
            break;

        case NN_UNDEFINED:
        default:
            bpp = 0;
            break;
    }

    return bpp;
}

bool isContinuous(const TensorRefNDData& buffer, int32_t upToDim) {
    upToDim = std::min(upToDim, buffer.ndims);
    auto packedSize = static_cast<int32_t>(getBpp(buffer.dType));

    for (int32_t i = 0; i < upToDim; ++i) {
        if (packedSize != buffer.strides[i]) {
            return false;
        }
        packedSize *= buffer.dims[i];
    }

    return true;
}

void* element(const TensorRefNDData& buffer, const int32_t indices[]) {
    unsigned offset = 0;
    for (int i = 0; i < buffer.ndims; ++i) {
        offset += buffer.strides[i] * indices[i];
    }
    return (buffer.addr + offset);
}

int32_t dim(const TensorRefNDData& tensor, subspace::LogDimIndex logDim, int32_t default_value) {
    bool success = false;
    NDDims indices = orderNDToIndices(tensor.ndOrder, success);
    if (!success || indices.getElement(logDim, -1) == -1) {
        return default_value;
    } else {
        return tensor.dims[indices[logDim]];
    }
}

unsigned int TensorRef::getDataSize() const {
    auto bytesPerElem = getBpp(dType);
    return this->getNumElements() * bytesPerElem;
}
unsigned int TensorRef::getFullDataSize() const {
    return (ndims > 0) ? strides[ndims - 1] * dims[ndims - 1] : 0;
}

unsigned int TensorRef::getNumElements() const {
    if(ndims == 0) return 0;
    unsigned int numElements = 1;
    for(int i = 0; i < ndims; i++) {
        numElements *= dims[i];
    }
    return numElements;
}

bool isNDDataLayoutFit(NDOrder ndOrder, const nn::TensorRefNDData& tensor) {
    long unsigned int lDims[MAX_ND_DIMS];
    long unsigned int lStrides[MAX_ND_DIMS];
    bool success = false;
    auto tensorPerm = subspace::orderNDToPermutation(tensor.ndOrder, success);
    if (!success || tensor.ndims != tensorPerm.ndims()) return false;
    for (int i = 0; i < tensor.ndims; i++) {
        lDims[tensorPerm[i]] = tensor.dims[i];
        lStrides[tensorPerm[i]] = tensor.strides[i];
    }
    return subspace::isLayoutFit(ndOrder, lDims, lStrides, tensor.ndims);
}

bool TensorRef::setByStrides(DataType dataType, const long unsigned int newDims[],
    const long unsigned int newStrides[], const uint64_t newStridesBits[],
    int dimensionality, NDOrder baseLineOrder) {
    bool success = false;
    ndOrder = extractLayoutFromShape(newDims, newStrides + 1, dimensionality, baseLineOrder, success);

    if (!success) return false;
    dType = dataType;

    ndims = dimensionality;
    bitsPerPixel = newStrides[0] * 8;
    auto perm = subspace::orderNDToPermutation(ndOrder, success);
    if (!success) return false;
    for(int i = 0; i < dimensionality; i++) {
        dims[i] = newDims[perm[i]];
        strides[i] = newStrides[perm[i] + 1];
        stridesBits[i] = newStridesBits[perm[i] + 1];
    }
    return true;
}

int32_t TensorRef::getDim(subspace::LogDimIndex dim, int32_t defaultValue) const {
    return nn::dim(*this, dim, defaultValue);
}

int32_t TensorRef::getStride(subspace::LogDimIndex dim, int32_t defaultValue) const {
    return strideBits(*this, dim, defaultValue) / CHAR_BIT;
}

int64_t TensorRef::getStrideBits(subspace::LogDimIndex dim, int64_t defaultValue) const {
    return strideBits(*this, dim, defaultValue);
}

bool TensorRef::set(void* addr, uint32_t dataType, NDOrder order, const int32_t dims[], const int32_t strides[]) {
    if (!subspace::isOrderNDValid(order)) {
        return false;
    }

    this->addr = static_cast<uint8_t*>(addr);

    this->dType = static_cast<DataType>(dataType);
    this->ndOrder = order;
    this->ndims = subspace::orderNDToNumDims(order);

    for (int32_t i = 0; i < this->ndims; ++i) {
        this->dims[i] = dims[i];
        this->strides[i] = strides[i];
        this->stridesBits[i] = strides[i] * CHAR_BIT;
    }
    for (int32_t i = this->ndims; i < MAX_ND_DIMS; ++i) {
        this->dims[i] = -1;
        this->strides[i] = 0;
    }
    this->bitsPerPixel = nn::getBpp(this->dType) * CHAR_BIT;
    return true;
}

//  Attention: the following methods dimN..strideW..strideBitsW make sense only in specific cases, mainly 4D and 3D.
//  Please avoid using them. They should be eliminated soon.
//  Use getDim, getStride instead.
int TensorRef::dimN(int defaultValue) const {
    switch(ndims) {
    case 3:
        return 1;
    case 4:
        return TensorRef::getDim(0, defaultValue);
    default:
        return defaultValue;
    }
}

int TensorRef::dimC(int defaultValue) const {
    switch(ndims) {
    case 3:
        return TensorRef::getDim(0, defaultValue);
    case 4:
        return TensorRef::getDim(1, defaultValue);
    default:
        return defaultValue;
    }
}

int TensorRef::dimH(int defaultValue) const {
    switch(ndims) {
    case 3:
        return TensorRef::getDim(1, defaultValue);
    case 4:
        return TensorRef::getDim(2, defaultValue);
    default:
        return defaultValue;
    }
}

int TensorRef::dimW(int defaultValue) const {
    switch(ndims) {
    case 3:
        return TensorRef::getDim(2, defaultValue);
    case 4:
        return TensorRef::getDim(3, defaultValue);
    default:
        return defaultValue;
    }
}

int TensorRef::strideN(int defaultValue) const {
    return static_cast<int> (strideBitsN(defaultValue) / CHAR_BIT);
}

int TensorRef::strideC(int defaultValue) const {
    return static_cast<int> (strideBitsC(defaultValue) / CHAR_BIT);
}

int TensorRef::strideH(int defaultValue) const {
    return static_cast<int> (strideBitsH(defaultValue) / CHAR_BIT);
}

int TensorRef::strideW(int defaultValue) const {
    return static_cast<int> (strideBitsW(defaultValue) / CHAR_BIT);
}

int64_t TensorRef::strideBitsN(int64_t defaultValue) const {
    switch(ndims) {
    case 3:
    case 4:
        return TensorRef::getStrideBits(0, defaultValue);
    default:
        return defaultValue;
    }
}

int64_t TensorRef::strideBitsC(int64_t defaultValue) const {
    switch(ndims) {
    case 3:
        return TensorRef::getStrideBits(0, defaultValue);
    case 4:
        return TensorRef::getStrideBits(1, defaultValue);
    default:
        return defaultValue;
    }
}

int64_t TensorRef::strideBitsH(int64_t defaultValue) const {
    switch(ndims) {
    case 3:
        return TensorRef::getStrideBits(1, defaultValue);
    case 4:
        return TensorRef::getStrideBits(2, defaultValue);
    default:
        return defaultValue;
    }
}

int64_t TensorRef::strideBitsW(int64_t defaultValue) const {
    switch(ndims) {
    case 3:
        return TensorRef::getStrideBits(2, defaultValue);
    case 4:
        return TensorRef::getStrideBits(3, defaultValue);
    default:
        return defaultValue;
    }
}

sw_params::MemRefData TensorRef::toMemRefData(sw_params::Location loc, bool doCopy) const {
    sw_params::MemRefData ret = {
            reinterpret_cast<uint32_t>(this->addr), // dataAddr
            true, // isStatic
            static_cast<uint32_t>(this->ndims), // uint32_t numDims;
            reinterpret_cast<uint32_t>(this->dims),//uint32_t dimsAddr;      // Pointer to the buffer with dimensions (int32_t[]).
            reinterpret_cast<uint32_t>(this->stridesBits), //uint32_t stridesAddr;   // Pointer to the buffer with strides in bits (int64_t[]).
            static_cast<uint32_t>(this->dType), //  uint32_t dataType;      // An enum, which should be aligned between kernels and the compiler.
            static_cast<uint64_t>(ndOrder), //uint64_t dimsOrder;     // Packed permutation array.
            loc
    };
#ifdef CONFIG_TARGET_SOC_3720
    if (loc == sw_params::Location::NN_CMX || loc == sw_params::Location::UPA_CMX) {
        unsigned int usedBytes = (ndims > 0) ? dims[ndims - 1] * strides[ndims - 1] : 0;
        auto bytesToAllocate = nn::math::round_up_power_of_2(NN_CACHE_LINE_LENGTH, usedBytes);

        if (bytesToAllocate <= SHAVE_LIB_DATA_SIZE - actShaveDataReserved) {
            ret.dataAddr = reinterpret_cast<uint32_t>(actShaveData + actShaveDataReserved);
            actShaveDataReserved += bytesToAllocate;
            if (doCopy) {
                DmaAlShave dmaTask;
                dmaTask.start(reinterpret_cast<uint8_t*>(this->addr), reinterpret_cast<uint8_t*>(ret.dataAddr),
                        usedBytes);
                dmaTask.wait();
            }
        } else {
            ret.location = sw_params::Location::DDR;
        }
    }
    nn::cache::flush(ret);
    nn::cache::flush(this->dims, this->ndims * sizeof(uint32_t));
    nn::cache::flush(this->stridesBits, this->ndims * sizeof(uint64_t));
#endif
    return ret;
}

subspace::LogDimIndex TensorRef::getLogicalC() const {
    switch(ndims) {
    case 3:
        return subspace::LogDimIndex(0);
    case 4:
        return subspace::LogDimIndex(1);
    default:
        return subspace::LogDimIndex(MAX_ND_DIMS);
    }
}

}  // namespace nn
