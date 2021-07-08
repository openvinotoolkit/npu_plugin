//
// Copyright Intel Corporation.
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

#include "vpux/compiler/utils/quantization.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/subspaces.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/IE/loop.hpp"

#include <cmath>

using namespace vpux;

//
// FakeQuantize support
//

std::tuple<double, int64_t> vpux::calcScaleAndZeroPoint(int64_t qMin, int64_t qMax, double rMin, double rMax) {
    VPUX_THROW_UNLESS(qMax > qMin, "Wrong quantized storage values range ['{0}', '{1}']", qMin, qMax);
    VPUX_THROW_UNLESS(rMax > rMin, "Wrong real values range ['{0}', '{1}']", rMin, rMax);

    //
    // Determine the scale.
    //

    const double qMinFP = static_cast<double>(qMin);
    const double qMaxFP = static_cast<double>(qMax);
    const double scale = (rMax - rMin) / (qMaxFP - qMinFP);

    VPUX_THROW_UNLESS(std::fabs(scale) > std::numeric_limits<double>::epsilon(),
                      "Quantization scale is too small : '{0}'", scale);

    //
    // Zero point computation.
    // In float, solve the affine equation for any known pair
    // (real value, corresponding quantized value), of which, two such pairs
    // are known: (rmin, qmin), (rmax, qmax).
    // The arithmetic error on the zero point computed from either pair will be
    // roughly machine_epsilon * (sum of absolute values of terms).
    // Use the variant that adds the smaller error.
    //

    const double zpFromMin = qMinFP - rMin / scale;
    const double zpFromMax = qMaxFP - rMax / scale;

    const double zpFromMinErr = std::fabs(qMinFP) + std::fabs(rMin / scale);
    const double zpFromMaxErr = std::fabs(qMaxFP) + std::fabs(rMax / scale);

    const double zpFP = (zpFromMinErr < zpFromMaxErr) ? zpFromMin : zpFromMax;
    const int64_t zp = static_cast<int64_t>(std::round(zpFP));

    return std::make_tuple(scale, zp);
}

mlir::quant::QuantizedType vpux::getQuantizedType(Const::ContentAttr lowConst, Const::ContentAttr highConst,
                                                  uint32_t levels, mlir::FloatType realType, mlir::Location loc) {
    if (lowConst == nullptr || highConst == nullptr) {
        (void)errorAt(loc, "Got non constant quantization parameters (low and high values)");
        return nullptr;
    }

    // TODO: how to choose this?
    const bool isSigned = false;

    mlir::Type storageType;
    int64_t qMin = 0;
    int64_t qMax = 0;
    switch (levels) {
    case 256:
        if (isSigned) {
            storageType = getSInt8Type(lowConst.getContext());
            qMin = -128;
            qMax = 127;
        } else {
            storageType = getUInt8Type(lowConst.getContext());
            qMin = 0;
            qMax = 255;
        }
        break;

    case 255:
        if (isSigned) {
            storageType = getSInt8Type(lowConst.getContext());
            qMin = -127;
            qMax = 127;
        } else {
            storageType = getUInt8Type(lowConst.getContext());
            qMin = 1;
            qMax = 255;
        }
        break;

    case 16:
        if (isSigned) {
            storageType = getSInt4Type(lowConst.getContext());
            qMin = -8;
            qMax = 7;
        } else {
            storageType = getUInt4Type(lowConst.getContext());
            qMin = 0;
            qMax = 15;
        }
        break;

    case 15:
        if (isSigned) {
            storageType = getSInt4Type(lowConst.getContext());
            qMin = -7;
            qMax = 7;
        } else {
            storageType = getUInt4Type(lowConst.getContext());
            qMin = 1;
            qMax = 15;
        }
        break;

    default:
        (void)errorAt(loc, "Got unsupported levels '{0}'", levels);
        return nullptr;
    }

    const auto lowAttr = lowConst.fold();
    const auto highAttr = highConst.fold();

    if (lowAttr.isSplat() && highAttr.isSplat()) {
        const auto low = lowAttr.getSplatValue<double>();
        const auto high = highAttr.getSplatValue<double>();

        double scale = 0.0;
        int64_t zeroPoint = 0;
        std::tie(scale, zeroPoint) = calcScaleAndZeroPoint(qMin, qMax, low, high);

        return mlir::quant::UniformQuantizedType::getChecked(loc, isSigned ? mlir::quant::QuantizationFlags::Signed : 0,
                                                             storageType, realType, scale, zeroPoint, qMin, qMax);
    } else {
        const auto lowShape = lowAttr.getShape();
        const auto highShape = highAttr.getShape();

        if (lowShape != highShape) {
            (void)errorAt(loc, "Low values shape '{0}' doesn't match with high values shape '{1}'", lowShape,
                          highShape);
            return nullptr;
        }

        Optional<int32_t> axisInd;
        for (size_t i = 0; i < lowShape.size(); ++i) {
            if (lowShape[Dim(i)] == 1) {
                continue;
            }

            if (axisInd.hasValue()) {
                (void)errorAt(loc, "Can't get axis index from shape '{0}'", lowShape);
                return nullptr;
            }

            axisInd = checked_cast<int32_t>(i);
        }
        if (!axisInd.hasValue()) {
            (void)errorAt(loc, "Can't get axis index from shape '{0}'", lowShape);
            return nullptr;
        }

        const auto lowVals = lowAttr.getValues<double>();
        const auto highVals = highAttr.getValues<double>();
        VPUX_THROW_UNLESS(lowVals.size() == highVals.size(), "Low/high attributes size mismatch : '{0}' vs '{1}'",
                          lowVals.size(), highVals.size());

        SmallVector<double> scales(lowVals.size());
        SmallVector<int64_t> zeroPoints(lowVals.size());

        loop_1d(LoopExecPolicy::Parallel, lowVals.size(), [&](size_t i) {
            std::tie(scales[i], zeroPoints[i]) = calcScaleAndZeroPoint(qMin, qMax, lowVals[i], highVals[i]);
        });

        return mlir::quant::UniformQuantizedPerAxisType::getChecked(
                loc, isSigned ? mlir::quant::QuantizationFlags::Signed : 0, storageType, realType, scales, zeroPoints,
                axisInd.getValue(), qMin, qMax);
    }
}

mlir::LogicalResult vpux::getFakeQuantParams(mlir::ShapedType qType, uint32_t& levels, mlir::RankedTensorType& attrType,
                                             mlir::DenseElementsAttr& rMinAttr, mlir::DenseElementsAttr& rMaxAttr,
                                             mlir::Location loc) {
    const auto qElemType = qType.getElementType().dyn_cast<mlir::quant::QuantizedType>();
    if (qElemType == nullptr) {
        return errorAt(loc, "Unsupported Quantized Type '{0}'", qType.getElementType());
    }

    const auto qMin = qElemType.getStorageTypeMin();
    const auto qMax = qElemType.getStorageTypeMax();

    levels = checked_cast<uint32_t>(qMax - qMin + 1);

    if (const auto uniformType = qElemType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        const auto scale = uniformType.getScale();
        const auto zeroPoint = uniformType.getZeroPoint();

        const auto rMin = dequantize(qMin, scale, zeroPoint);
        const auto rMax = dequantize(qMax, scale, zeroPoint);

        attrType = mlir::RankedTensorType::get({}, mlir::Float32Type::get(qType.getContext()));
        rMinAttr = mlir::DenseElementsAttr::get(attrType, rMin);
        rMaxAttr = mlir::DenseElementsAttr::get(attrType, rMax);

        return mlir::success();
    } else if (const auto uniformType = qElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto scales = uniformType.getScales();
        const auto zeroPoints = uniformType.getZeroPoints();
        const auto axis = Dim(uniformType.getQuantizedDimension());

        SmallVector<float> rMinVals(scales.size());
        SmallVector<float> rMaxVals(scales.size());

        VPUX_THROW_UNLESS(zeroPoints.size() == scales.size(), "Wrong zeroPoints size '{0}', expected '{1}'",
                          zeroPoints.size(), scales.size());

        loop_1d(LoopExecPolicy::Parallel, scales.size(), [&](size_t i) {
            rMinVals[i] = dequantize(qMin, scales[i], zeroPoints[i]);
            rMaxVals[i] = dequantize(qMax, scales[i], zeroPoints[i]);
        });

        Shape attrShape(qType.getRank(), 1);
        attrShape[axis] = scales.size();

        attrType = mlir::RankedTensorType::get(attrShape.raw(), mlir::Float32Type::get(qType.getContext()));
        rMinAttr = mlir::DenseElementsAttr::get(attrType, makeArrayRef(rMinVals));
        rMaxAttr = mlir::DenseElementsAttr::get(attrType, makeArrayRef(rMaxVals));

        return mlir::success();
    } else {
        return errorAt(loc, "Unsupported Quantized Type '{0}'", qElemType);
    }
}

mlir::Type vpux::normalizeQuantStorageType(mlir::Type type) {
    if (const auto intType = type.dyn_cast<mlir::IntegerType>()) {
        // add sign semantic
        if (intType.isSignless()) {
            return mlir::IntegerType::get(intType.getContext(), intType.getWidth(), mlir::IntegerType::Unsigned);
        }
    }
    return type;
}

//
// Dequantize support
//

float vpux::dequantize(int64_t qVal, double scale, int64_t zeroPoint) {
    return static_cast<float>((qVal - zeroPoint) * scale);
}
