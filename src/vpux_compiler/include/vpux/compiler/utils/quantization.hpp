//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/EMU/graph-schema/blob_writer.hpp"
#include "vpux/compiler/dialect/IE/attributes/enums.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>

#include <cstdint>
#include <tuple>

namespace vpux {

//
// Utilities for quantized types
//

mlir::LogicalResult validateQuantElemType(mlir::Location loc, vpux::NDTypeInterface mainType);

mlir::Type normalizeQuantStorageType(mlir::quant::QuantizedType qType);

mlir::quant::UniformQuantizedPerAxisType expandScalesAndZP(mlir::quant::UniformQuantizedPerAxisType perAxisQType,
                                                           ShapeRef padBefore, ShapeRef padAfter);

mlir::quant::UniformQuantizedPerAxisType tileScalesAndZP(mlir::quant::UniformQuantizedPerAxisType perAxisQType,
                                                         ShapeRef shape, ShapeRef offsets);

mlir::quant::UniformQuantizedPerAxisType changeAxis(mlir::quant::UniformQuantizedPerAxisType perAxisQType,
                                                    int32_t axis);

mlir::quant::QuantizedType changeStorageType(mlir::quant::QuantizedType qType, mlir::Type storageType);

bool canBeMerged(mlir::quant::UniformQuantizedPerAxisType type1, mlir::quant::UniformQuantizedPerAxisType type2);
mlir::quant::UniformQuantizedPerAxisType concatScalesAndZP(ArrayRef<mlir::quant::UniformQuantizedPerAxisType> types);

using Scales = SmallVector<double>;
using ZeroPoints = SmallVector<int64_t>;

std::pair<Scales, ZeroPoints> extractScalesAndZeroPoints(mlir::Type tensorElemType);

class QuantizationApproximation {
public:
    QuantizationApproximation(vpux::VPU::ArchKind architecture, double target);

    int64_t mult() const;
    int64_t shift() const;
    int64_t postShift() const;
    void setMult(uint16_t mult);
    void setShift(uint8_t shift);

private:
    uint16_t _mult;
    uint8_t _shift;
    int8_t _postShift;
};

class EltwiseQuantizationApproximation {
public:
    EltwiseQuantizationApproximation(vpux::VPU::ArchKind architecture, double input1Target, double input2Target,
                                     double outputTarget);

    QuantizationApproximation input1() const;
    QuantizationApproximation input2() const;
    QuantizationApproximation output() const;

private:
    QuantizationApproximation _input1;
    QuantizationApproximation _input2;
    QuantizationApproximation _output;
};

class PReLUApproximation {
public:
    PReLUApproximation(vpux::VPU::ArchKind architecture, double alpha);

    int64_t mult() const;
    int64_t shift() const;

private:
    // VPUX30XX mult is int8_t, VPUX37XX mult is uint16_t - using int32_t as common storage
    int32_t _mult;
    uint8_t _shift;
};

std::pair<int64_t, int64_t> getClampValuesForQuantizedOps(mlir::quant::QuantizedType outElemQType,
                                                          mlir::Type outElemType);

//
// FakeQuantize support
//

mlir::quant::QuantizedType getQuantizedType(mlir::Attribute lowConstAttr, mlir::Attribute highConstAttr, int64_t levels,
                                            mlir::FloatType realType, bool isSigned, mlir::Location loc,
                                            IE::AutoBroadcastType broadcast = IE::AutoBroadcastType::NONE_OR_EXPLICIT);

void getFakeQuantParams(mlir::quant::UniformQuantizedType qElemType, int64_t& levels, float& rMin, float& rMax);

void getFakeQuantParams(mlir::quant::UniformQuantizedPerAxisType qElemType, int64_t& levels,
                        SmallVectorImpl<float>& rMinVals, SmallVectorImpl<float>& rMaxVals);

void getFakeQuantParams(vpux::NDTypeInterface qType, int64_t& levels, mlir::RankedTensorType& attrType,
                        mlir::DenseElementsAttr& rMinAttr, mlir::DenseElementsAttr& rMaxAttr);

std::tuple<double, int64_t> calcScaleAndZeroPoint(int64_t qMin, int64_t qMax, double rMin, double rMax, bool isSigned);

//
// Dequantize support
//

float dequantize(int64_t qVal, double scale, int64_t zeroPoint);

//
// Convert real numbers to fixed point S16.16 format.
//

int32_t toFixedPoint(const double realVal);

//
//  Serialize Scales And ZeroPoints for Emulator serializer
//

std::pair<EMU::BlobWriter::Vector<uint16_t>, EMU::BlobWriter::Vector<uint16_t>> serializeScalesAndZeroPointsEmu(
        mlir::Value input, mlir::Value output, EMU::BlobWriter& writer);

// Broadcasting

template <typename T>
void broadcastRange(SmallVectorImpl<T>& lowVals, SmallVectorImpl<T>& highVals, IE::AutoBroadcastType broadcast) {
    if (lowVals.size() == highVals.size()) {
        return;
    }
    if (broadcast == IE::AutoBroadcastType::NONE_OR_EXPLICIT) {
        return;
    }

    const auto numpyBroadcast = [](SmallVectorImpl<T>& smaller, SmallVectorImpl<T>& larger) {
        VPUX_THROW_UNLESS(smaller.size() == 1, "One of the dimensions should be 1 for broadcasting.");
        return SmallVector<T>(larger.size(), smaller[0]);
    };

    if (broadcast == IE::AutoBroadcastType::NUMPY) {
        if (lowVals.size() < highVals.size()) {
            lowVals = numpyBroadcast(lowVals, highVals);
        } else {
            highVals = numpyBroadcast(highVals, lowVals);
        }
        return;
    }

    VPUX_THROW("Unsupported broadcast type '{0}'", broadcast);
}

}  // namespace vpux
