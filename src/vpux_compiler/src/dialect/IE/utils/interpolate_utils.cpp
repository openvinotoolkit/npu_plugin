//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/utils/core/numeric.hpp"

#include "vpux/compiler/utils/error.hpp"

#include <numeric>

namespace vpux {
namespace IE {

mlir::FailureOr<SmallVector<int64_t>> extractIntVector(mlir::Location loc, const mlir::Value value,
                                                       const Optional<mlir::ArrayAttr>& attr) {
    if (attr.has_value() && attr.value() != nullptr) {
        return parseIntArrayAttr<int64_t>(attr.value());
    } else if (value != nullptr) {
        auto valueConst = value.getDefiningOp<Const::DeclareOp>();
        if (valueConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for interpolate attribute");
        }

        const auto valueContent = valueConst.getContent();
        return to_small_vector(valueContent.getValues<int64_t>());
    }
    return errorAt(loc, "Parameter were not provided");
}

mlir::FailureOr<SmallVector<double>> extractFPVector(mlir::Location loc, const mlir::Value value,
                                                     const Optional<mlir::ArrayAttr>& attr) {
    if (attr.has_value() && attr.value() != nullptr) {
        return parseFPArrayAttr<double>(attr.value());
    } else if (value != nullptr) {
        auto valueConst = value.getDefiningOp<Const::DeclareOp>();

        if (valueConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for interpolate attribute");
        }

        const auto valueContent = valueConst.getContent();
        return to_small_vector(valueContent.getValues<double>());
    }
    return errorAt(loc, "Parameter were not provided");
}

void applyInterpPads(MutableArrayRef<int64_t> outShape, ArrayRef<int64_t> padsBegin, ArrayRef<int64_t> padsEnd) {
    // pads might be zero initialized
    if (padsBegin.size() != padsEnd.size() || padsBegin.size() != outShape.size()) {
        return;
    }
    // naive implementation only apply pads to calculated output shape
    for (auto& d : outShape | indexed) {
        d.value() += padsBegin[d.index()] + padsEnd[d.index()];
    }
}

mlir::FailureOr<SmallVector<int64_t>> propagateShape(mlir::Location loc, mlir::FailureOr<SmallVector<int64_t>> axes,
                                                     ArrayRef<int64_t> origShape,
                                                     mlir::FailureOr<ArrayRef<int64_t>> padsBegin,
                                                     mlir::FailureOr<ArrayRef<int64_t>> padsEnd,
                                                     vpux::IE::InterpolateCalcMode calcMode,
                                                     mlir::FailureOr<ArrayRef<int64_t>> sizes,
                                                     mlir::FailureOr<ArrayRef<double>> scales, vpux::Logger log) {
    log.trace("Interp propagate shape: input = {0}", origShape);
    const auto axesVal = axes.value();
    auto inferedShape = to_small_vector(origShape);

    for (auto axis : axesVal) {
        VPUX_THROW_UNLESS(checked_cast<size_t>(axis) < origShape.size(), "Invalid axis {0} for shape of rank {1}", axis,
                          origShape.size());
    }

    if (calcMode == IE::InterpolateCalcMode::SIZES) {
        const auto sizesVal = sizes.value();

        if (sizesVal.size() != axesVal.size()) {
            return errorAt(loc,
                           "Num of elements in sizes tensor: {0} should be equal to number of indices in axes: {1}",
                           sizesVal.size(), axesVal.size());
        }
        auto sizesIter = sizesVal.begin();

        for (const auto& i : axesVal) {
            log.trace("Interp sizes - axis: {0}", i);
            inferedShape[i] = *sizesIter++;
        }
    } else if (calcMode == IE::InterpolateCalcMode::SCALES) {
        const auto scales_val = scales.value();

        if (scales_val.size() != axesVal.size()) {
            return errorAt(loc,
                           "Num of elements in scales tensor: {0} should be equal to number of indices in axes: {1}",
                           scales_val.size(), axesVal.size());
        }

        auto scalesIter = scales_val.begin();

        for (const auto& i : axesVal) {
            log.trace("Interp scales - axis: {0}", i);
            inferedShape[i] = static_cast<int64_t>(floor((*scalesIter++) * origShape[i]));
        }

    } else {
        return errorAt(loc, "Doesn't support shape_calculation_mode: {0}", calcMode);
    }

    // meaning pads provided in attributes
    if (mlir::succeeded(padsBegin) && mlir::succeeded(padsEnd)) {
        applyInterpPads(inferedShape, padsBegin.value(), padsEnd.value());
    }

    log.trace("Interp propagate shape: output = {0}", inferedShape);

    return inferedShape;
}

SmallVector<int64_t> getDefaultInterpolateAxes(IE::InterpolateOpAdaptor interpolate) {
    SmallVector<int64_t> axes(interpolate.input().getType().cast<mlir::ShapedType>().getRank());
    std::iota(axes.begin(), axes.end(), 0);

    return axes;
}

mlir::FailureOr<SmallVector<int64_t>> calcOutputShapes(mlir::Location loc, IE::InterpolateOpAdaptor interpolate,
                                                       vpux::Logger log) {
    const auto axesAttr = interpolate.axes_attr();
    const bool validAxesAttr = (axesAttr.has_value() && axesAttr.value() != nullptr);
    const auto axes = (interpolate.axes() == nullptr && !validAxesAttr)
                              ? getDefaultInterpolateAxes(interpolate)
                              : extractIntVector(loc, interpolate.axes(), interpolate.axes_attr());
    const auto beginPads = extractIntVector(loc, {}, interpolate.attr().getPadsBegin());
    const auto endPads = extractIntVector(loc, {}, interpolate.attr().getPadsEnd());

    const auto inType = interpolate.input().getType().cast<mlir::ShapedType>();
    const auto inputShape = inType.getShape();

    return propagateShape(loc, axes, inputShape, beginPads, endPads, interpolate.attr().getShapeCalcMode().getValue(),
                          extractIntVector(loc, interpolate.sizes(), interpolate.sizes_attr()),
                          extractFPVector(loc, interpolate.scales(), interpolate.scales_attr()), log);
}

bool isBroadCastInterpolate(IE::InterpolateOp op) {
    const auto inputType = op.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape();

    auto padValueIsNull = [](mlir::ArrayAttr pad) {
        if (pad == nullptr) {
            return true;
        }
        auto padValue = parseIntArrayAttr<int64_t>(pad);
        return llvm::all_of(padValue, [](int64_t value) {
            return value == 0;
        });
    };

    if (!padValueIsNull(op.attr().getPadsBegin()) || !padValueIsNull(op.attr().getPadsEnd())) {
        return false;
    }

    const auto axes = extractIntVector(op.getLoc(), op.axes(), op.axes_attr());
    VPUX_THROW_UNLESS(mlir::succeeded(axes), "Cannot get axes Attr");
    const auto axesValues = axes.value();
    const auto calcMode = op.attr().getShapeCalcMode().getValue();
    if (calcMode == IE::InterpolateCalcMode::SIZES) {
        const auto size = extractIntVector(op.getLoc(), op.sizes(), op.sizes_attr());
        VPUX_THROW_UNLESS(mlir::succeeded(axes), "Cannot get size Attr");
        const auto sizeValues = size.value();
        VPUX_THROW_UNLESS(sizeValues.size() == axesValues.size(),
                          "Num of elements sizes tensor tensor: {0} should be equal to number of indices in axes: {1}",
                          sizeValues.size(), axesValues.size());

        const auto shapes = zip(axesValues, sizeValues);
        return llvm::all_of(shapes, [&](const std::tuple<int64_t, int64_t>& dimSize) {
            const auto inDimSize = inputShape[Dim(std::get<0>(dimSize))];
            const auto outDimSize = std::get<1>(dimSize);
            return (inDimSize == outDimSize) || (inDimSize != outDimSize && inDimSize == 1);
        });
    } else if (calcMode == IE::InterpolateCalcMode::SCALES) {
        const auto scales = extractFPVector(op.getLoc(), op.scales(), op.scales_attr());
        VPUX_THROW_UNLESS(mlir::succeeded(scales), "Cannot get scales Attr");
        const auto scalesVal = scales.value();
        VPUX_THROW_UNLESS(scalesVal.size() == axesValues.size(),
                          "Num of elements scales tensor tensor: {0} should be equal to number of indices in axes: {1}",
                          scalesVal.size(), axesValues.size());

        const auto shapes = zip(axesValues, scalesVal);
        return llvm::all_of(shapes, [&](const std::tuple<int64_t, double>& dimSize) {
            const auto inDimSize = inputShape[Dim(std::get<0>(dimSize))];
            const auto scaleSize = std::get<1>(dimSize);
            return isDoubleEqual(scaleSize, 1.0) || (!isDoubleEqual(scaleSize, 1.0) && inDimSize == 1);
        });
    }

    VPUX_THROW("Doesn't support shape_calculation_mode: {0}", calcMode);
}

std::pair<double, double> computeFractionCoefficients(double fraction) {
    return std::pair<double, double>{1.0 - fraction, fraction};
}

//
// Coordinate transformation mode functions
//

// HalfPixel coordintate transformation mode
std::pair<int32_t, double> mapCoordHalfPixel(int32_t x, double scale, int32_t, int32_t) {
    auto fractionX = scale * (x + 0.5) - 0.5;
    auto integerX = static_cast<int32_t>(std::floor(fractionX));
    fractionX -= integerX;
    return std::pair<int32_t, double>{integerX, fractionX};
}

// PytorchHalfPixel coordintate transformation mode
std::pair<int32_t, double> mapCoordPytorchHalfPixel(int32_t x, double scale, int32_t outputSize, int32_t) {
    auto fractionX = scale * (x + 0.5) - 0.5;
    if (outputSize <= 1) {
        fractionX = 0;
    }
    auto integerX = static_cast<int32_t>(std::floor(fractionX));
    fractionX -= integerX;
    return std::pair<int32_t, double>{integerX, fractionX};
}

// Asymmetric coordintate transformation mode
std::pair<int32_t, double> mapCoordAsymmetric(int32_t x, double scale, int32_t, int32_t) {
    auto fractionX = scale * x;
    auto integerX = static_cast<int32_t>(std::floor(fractionX));
    fractionX -= integerX;
    return std::pair<int32_t, double>{integerX, fractionX};
}

// TfHalfPixelForNN coordintate transformation mode
std::pair<int32_t, double> mapCoordTfHalfPixelForNN(int32_t x, double scale, int32_t, int32_t) {
    auto fractionX = scale * (x + 0.5);
    auto integerX = static_cast<int32_t>(std::floor(fractionX));
    fractionX -= integerX;
    return std::pair<int32_t, double>{integerX, fractionX};
}

// AlignCorners coordintate transformation mode
std::pair<int32_t, double> mapCoordAlignCorners(int32_t x, double, int32_t outputSize, int32_t inputSize) {
    double fractionX;
    if (outputSize == 1) {
        fractionX = 0;
    } else {
        fractionX = static_cast<double>(x) * (inputSize - 1) / (outputSize - 1);
    }
    auto integerX = static_cast<int32_t>(std::floor(fractionX));
    fractionX -= integerX;
    return std::pair<int32_t, double>{integerX, fractionX};
}

MapCoordFuncT getMapCoordMethod(InterpolateCoordMode coordMode) {
    if (coordMode == IE::InterpolateCoordMode::HALF_PIXEL) {
        return mapCoordHalfPixel;
    } else if (coordMode == IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL) {
        return mapCoordPytorchHalfPixel;
    } else if (coordMode == IE::InterpolateCoordMode::ASYMMETRIC) {
        return mapCoordAsymmetric;
    } else if (coordMode == IE::InterpolateCoordMode::TF_HALF_PIXEL_FOR_NN) {
        return mapCoordTfHalfPixelForNN;
    } else if (coordMode == IE::InterpolateCoordMode::ALIGN_CORNERS) {
        return mapCoordAlignCorners;
    } else {
        VPUX_THROW("Unsupported coodintate transformation mode: {0}.", coordMode);
    }
}

}  // namespace IE
}  // namespace vpux
