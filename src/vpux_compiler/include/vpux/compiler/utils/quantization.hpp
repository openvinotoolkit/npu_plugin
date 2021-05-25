//
// Copyright 2020 Intel Corporation.
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
