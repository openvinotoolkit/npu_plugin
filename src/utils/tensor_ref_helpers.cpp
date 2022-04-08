//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "tensor_ref_helpers.hpp"

#if defined(__arm__) || defined(__aarch64__)
#include <climits>
#include <sstream>

namespace vpu {
namespace utils {

off_t bitsPerPixel(const DataType dType) {
    switch (dType) {
    case DataType::NN_FP64:
    case DataType::NN_U64:
    case DataType::NN_I64:
        return 64;
    case DataType::NN_FP32:
    case DataType::NN_U32:
    case DataType::NN_I32:
        return 32;
    case DataType::NN_FP16:
    case DataType::NN_U16:
    case DataType::NN_I16:
        return 16;
    case DataType::NN_FP8:
    case DataType::NN_U8:
    case DataType::NN_I8:
        return 8;
    case DataType::NN_I4:
        return 4;
    case DataType::NN_I2:
        return 2;
    case DataType::NN_BIN:
        return 1;
    default:
        throw std::runtime_error("bitsPerPixel: unknown data type");
    }
}

off_t getTotalSize(const TensorRefNDData& tensorRef) {
    off_t numElements = 1;
    for (int32_t dimIdx = 0; dimIdx < tensorRef.ndims; dimIdx++) {
        numElements *= tensorRef.dims[dimIdx];
    }
    return numElements * bitsPerPixel(tensorRef.dType) / CHAR_BIT;
}

std::string serializeShape(const TensorRefNDData& tensorRef) {
    std::string shapeStr = "{ ";
    for (int32_t idx = 0; idx < tensorRef.ndims; idx++) {
        shapeStr += std::to_string(tensorRef.dims[idx]) + " ";
    }
    shapeStr += "}";
    return shapeStr;
}

std::string serializeStrides(const TensorRefNDData& tensorRef) {
    std::string stridesStr = "{ ";
    for (int32_t idx = 0; idx < tensorRef.ndims; idx++) {
        stridesStr += std::to_string(tensorRef.stridesBits[idx]) + " ";
    }
    stridesStr += "}";
    return stridesStr;
}

std::string serializeDType(const TensorRefNDData& tensorRef) {
    switch (tensorRef.dType) {
    case DataType::NN_FP64:
        return "float64";
    case DataType::NN_FP32:
        return "float32";
    case DataType::NN_FP16:
        return "float16";
    case DataType::NN_FP8:
        return "float8";
    case DataType::NN_U64:
        return "uint64";
    case DataType::NN_U32:
        return "uint32";
    case DataType::NN_U16:
        return "uint16";
    case DataType::NN_U8:
        return "uint8";
    case DataType::NN_I64:
        return "int64";
    case DataType::NN_I32:
        return "int32";
    case DataType::NN_I16:
        return "int16";
    case DataType::NN_I8:
        return "int8";
    case DataType::NN_I4:
        return "int4";
    case DataType::NN_I2:
        return "int2";
    case DataType::NN_BIN:
        return "binary";
    default:
        return "undefined";
    }
}

std::string serializeOrder(const TensorRefNDData& tensorRef) {
    switch (tensorRef.ndOrder) {
    case NDOrder::ND_NHWC:
        return "NHWC";
    case NDOrder::ND_NHCW:
        return "NHCW";
    case NDOrder::ND_NCHW:
        return "NCHW";
    case NDOrder::ND_NCWH:
        return "NCWH";
    case NDOrder::ND_NWHC:
        return "NWHC";
    case NDOrder::ND_NWCH:
        return "NWCH";
    case NDOrder::ND_HWC:
        return "HWC";
    case NDOrder::ND_CHW:
        return "CHW";
    case NDOrder::ND_WHC:
        return "WHC";
    case NDOrder::ND_HCW:
        return "HCW";
    case NDOrder::ND_WCH:
        return "WCH";
    case NDOrder::ND_CWH:
        return "CWH";
    case NDOrder::ND_NC:
        return "NC";
    case NDOrder::ND_CN:
        return "CN";
    case NDOrder::ND_C:
        return "C";
    default:
        std::ostringstream hexToStr;
        hexToStr << std::hex;
        hexToStr << static_cast<uint64_t>(tensorRef.ndOrder);
        return hexToStr.str();
    }
}

}  // namespace utils
}  // namespace vpu
#endif
