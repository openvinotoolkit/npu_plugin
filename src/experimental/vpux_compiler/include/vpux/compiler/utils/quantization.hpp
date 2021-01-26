//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include "vpux/compiler/core/attributes/const_content.hpp"
#include "vpux/compiler/core/ops_interfaces.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>

#include <cstdint>
#include <tuple>

namespace vpux {

//
// FakeQuantize support
//

std::tuple<double, int64_t> calcScaleAndZeroPoint(int64_t qMin, int64_t qMax, double rMin, double rMax);

mlir::quant::QuantizedType getQuantizedType(ConstantInterface lowConst, ConstantInterface highConst, uint32_t levels,
                                            mlir::FloatType realType, mlir::Location loc);

mlir::LogicalResult getFakeQuantParams(mlir::ShapedType qType, uint32_t& levels, mlir::RankedTensorType& attrType,
                                       mlir::DenseElementsAttr& rMinAttr, mlir::DenseElementsAttr& rMaxAttr,
                                       mlir::Location loc);

//
// Quantize support
//

// I16 should be enough to hold all supported quantized storage types
int16_t quantize(float realVal, double scale, int64_t zeroPoint, int64_t qMin, int64_t qMax);

mlir::DenseElementsAttr quantize(ConstContentAttr input, mlir::ShapedType qType, mlir::Location loc);

//
// Dequantize support
//

float dequantize(int64_t qVal, double scale, int64_t zeroPoint);

mlir::DenseElementsAttr dequantize(ConstContentAttr input, mlir::ShapedType qType, mlir::Location loc);

}  // namespace vpux
