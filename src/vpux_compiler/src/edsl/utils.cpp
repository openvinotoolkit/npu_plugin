//
// Copyright 2021 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/edsl/utils.hpp"

#include <mlir/Support/DebugStringHelper.h>

using namespace mlir;

namespace vpux {
namespace edsl {

SmallVector<uint32_t, 4> padShapeTo4Dim(ArrayRef<int64_t> from) {
    SmallVector<uint32_t, 4> into;
    if (from.size() <= 4) {
        for (unsigned i = 0; i < 4 - from.size(); ++i) {
            into.emplace_back(1);
        }
    }
    for (auto dim : from) {
        into.emplace_back(static_cast<uint32_t>(dim));
    }
    return into;
}

MVCNN::DataType getSchemaDataType(Type type) {
    if (type.isF16())
        return MVCNN::DataType_FLOAT16;
    if (type.isF32())
        return MVCNN::DataType_FLOAT32;
    if (type.isF64())
        return MVCNN::DataType_FLOAT64;
    if (type.isSignlessInteger()) {
        switch (type.getIntOrFloatBitWidth()) {
        case 8:
            return MVCNN::DataType_UINT8;
        case 16:
            return MVCNN::DataType_UINT16;
        case 32:
            return MVCNN::DataType_UINT32;
        case 64:
            return MVCNN::DataType_UINT64;
        }
    }
    VPUX_THROW("Unsupported DType: {0}", debugString(type));
}

MVCNN::InitValue convertInitValue(Attribute attr) {
    if (attr) {
        if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
            return MVCNN::InitValue{/*needInit=*/true, /*isInt=*/true,
                                    /*intValue=*/intAttr.getInt(),
                                    /*floatValue=*/0.0};
        }
        if (auto floatAttr = attr.dyn_cast<FloatAttr>()) {
            return MVCNN::InitValue{/*needInit=*/true, /*isInt=*/false,
                                    /*intValue=*/0,
                                    /*floatValue=*/floatAttr.getValueAsDouble()};
        }
    }
    return MVCNN::InitValue{};
}

}  // namespace edsl
}  // namespace vpux
