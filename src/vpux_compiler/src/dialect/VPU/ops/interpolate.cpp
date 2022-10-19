//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

namespace {

mlir::FailureOr<SmallVector<int64_t>> extractIntVector(mlir::Location loc, const mlir::Value& value,
                                                       const mlir::ArrayAttr& attr) {
    if (attr != nullptr) {
        return parseIntArrayAttr<int64_t>(attr);
    } else if (value != nullptr) {
        auto valueConst = value.getDefiningOp<Const::DeclareOp>();
        if (valueConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for interpolate attribute");
        }

        const auto valueContent = valueConst.content();
        return to_small_vector(valueContent.getValues<int64_t>());
    }
    return errorAt(loc, "Parameter were not provided");
}

mlir::FailureOr<SmallVector<double>> extractFPVector(mlir::Location loc, const mlir::Value& value,
                                                     const mlir::ArrayAttr& attr) {
    if (attr != nullptr) {
        return parseFPArrayAttr<double>(attr);
    } else if (value != nullptr) {
        auto valueConst = value.getDefiningOp<Const::DeclareOp>();

        if (valueConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for interpolate attribute");
        }

        const auto valueContent = valueConst.content();
        return to_small_vector(valueContent.getValues<double>());
    }
    return errorAt(loc, "Parameter were not provided");
}

mlir::FailureOr<SmallVector<int64_t>> calcOutputShapes(mlir::Location loc, VPU::InterpolateOpAdaptor interpolate) {
    const auto axes = extractIntVector(loc, interpolate.axes(), interpolate.axes_attr());
    const auto axesVal = axes.getValue();

    const auto inType = interpolate.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inType.getShape().raw();

    SmallVector<int64_t> outShape;

    for (size_t i = 0; i < inputShape.size(); ++i) {
        outShape.emplace_back(inputShape[i]);
    }

    auto calcMode = interpolate.attr().shape_calc_mode().getValue();

    if (calcMode == IE::InterpolateCalcMode::SIZES) {
        const auto sizes = extractIntVector(loc, interpolate.sizes(), interpolate.sizes_attr());
        const auto sizes_val = sizes.getValue();

        if (sizes_val.size() != axesVal.size()) {
            return errorAt(loc,
                           "Num of elements in sizes tensor: {0} should be equal to number of indices in axes: {1}",
                           sizes_val.size(), axesVal.size());
        }
        auto sizesIter = sizes_val.begin();

        for (const auto& i : axesVal)
            outShape[i] = *sizesIter++;

        return outShape;

    } else if (calcMode == IE::InterpolateCalcMode::SCALES) {
        const auto scales = extractFPVector(loc, interpolate.scales(), interpolate.scales_attr());
        const auto scales_val = scales.getValue();

        if (scales_val.size() != axesVal.size()) {
            return errorAt(loc,
                           "Num of elements in scales tensor: {0} should be equal to number of indices in axes: {1}",
                           scales_val.size(), axesVal.size());
        }

        auto scalesIter = scales_val.begin();

        for (const auto& i : axesVal) {
            outShape[i] = static_cast<int64_t>(floor((*scalesIter++) * inputShape[i]));
        }

        return outShape;

    } else
        return errorAt(loc, "Doesn't support shape_calculation_mode: {0}", calcMode);
}

}  // namespace

mlir::LogicalResult vpux::VPU::InterpolateOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               mlir::Optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::InterpolateOpAdaptor interpolate(operands, attrs);
    if (mlir::failed(interpolate.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = interpolate.input().getType().cast<vpux::NDTypeInterface>();

    auto outShape = calcOutputShapes(loc, interpolate);

    if (mlir::failed(outShape)) {
        return mlir::failure();
    }
    const auto outType = inType.changeShape(Shape(outShape.getValue()));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::InterpolateOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::InterpolateParamsBuilder builder(writer);

    const auto mode = attr().mode().getValue();
    const auto coord_mode = attr().coord_mode().getValue();
    const auto nearest_mode = attr().nearest_mode().getValue();
    const auto antialias = attr().antialias().getValue();

    const auto interpolateModeIter = VPUIP::supportedInterpModeMap.find(mode);
    VPUX_THROW_UNLESS(interpolateModeIter != VPUIP::supportedInterpModeMap.end(), "Unsupported interpolate mode {0}",
                      mode);
    builder.add_interpolationMode(interpolateModeIter->second);

    const auto coordModeIter = VPUIP::coordTransformModeMap.find(coord_mode);
    VPUX_THROW_UNLESS(coordModeIter != VPUIP::coordTransformModeMap.end(),
                      "Unsupported coordinate transformation mode {0}", coord_mode);
    builder.add_coordTransformMode(coordModeIter->second);

    const auto nearestModeIter = VPUIP::nearestModeMap.find(nearest_mode);
    VPUX_THROW_UNLESS(nearestModeIter != VPUIP::nearestModeMap.end(), "Unsupported nearest mode {0}", nearest_mode);
    builder.add_nearestMode(nearestModeIter->second);

    builder.add_align_corners(coord_mode == IE::InterpolateCoordMode::ALIGN_CORNERS);
    builder.add_antialias(antialias);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_InterpolateParams});
}

void applyInterpPads(MutableArrayRef<int64_t> outShape, ArrayRef<int64_t> padsBegin, ArrayRef<int64_t> padsEnd) {
    // pads might be zero initialized
    if (padsBegin.size() != padsEnd.size() || padsBegin.size() != outShape.size()) {
        return;
    }
    // naive implementation only apply pads to calculated output shape
    for (auto d : outShape | indexed) {
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
    const auto axes_val = axes.getValue();
    auto inferedShape = to_small_vector(origShape);

    if (calcMode == IE::InterpolateCalcMode::SIZES) {
        const auto sizes_val = sizes.getValue();

        if (sizes_val.size() != axes_val.size()) {
            return errorAt(loc,
                           "Num of elements in sizes tensor: {0} should be equal to number of indices in axes: {1}",
                           sizes_val.size(), axes_val.size());
        }
        auto sizesIter = sizes_val.begin();

        for (const auto& i : axes_val) {
            log.trace("Interp sizes - axis: {0}", i);
            inferedShape[i] = *sizesIter++;
        }
    } else if (calcMode == IE::InterpolateCalcMode::SCALES) {
        const auto scales_val = scales.getValue();

        if (scales_val.size() != axes_val.size()) {
            return errorAt(loc,
                           "Num of elements in scales tensor: {0} should be equal to number of indices in axes: {1}",
                           scales_val.size(), axes_val.size());
        }

        auto scalesIter = scales_val.begin();

        for (const auto& i : axes_val) {
            log.trace("Interp scales - axis: {0}", i);
            inferedShape[i] = static_cast<int64_t>(floor((*scalesIter++) * origShape[i]));
        }

    } else
        return errorAt(loc, "Doesn't support shape_calculation_mode: {0}", calcMode);

    // meaning pads provided in attributes
    if (mlir::succeeded(padsBegin) && mlir::succeeded(padsEnd)) {
        applyInterpPads(inferedShape, padsBegin.getValue(), padsEnd.getValue());
    }

    log.trace("Interp propagate shape: output = {0}", inferedShape);

    return inferedShape;
}

InputTiling vpux::VPU::InterpolateOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto origAxes = extractIntVector(getLoc(), axes(), axes_attrAttr());
    VPUX_THROW_UNLESS(mlir::succeeded(origAxes), "InterpolateOp::backInferTileInfo failed to extract axes");

    // Compute scale-factors based on full I/O resolution ratio
    const auto iShape = input().getType().cast<vpux::NDTypeInterface>().getShape().raw();
    const auto oShape = output().getType().cast<vpux::NDTypeInterface>().getShape().raw();
    const auto axesVal = origAxes.getValue();
    vpux::Scales fwdScales;
    for (size_t i = 0; i < axesVal.size(); i++) {
        fwdScales.push_back(static_cast<double>(oShape[axesVal[i]]) / iShape[axesVal[i]]);
    }

    SmallVector<double> backwardScale;
    for (auto fwScale : fwdScales) {
        backwardScale.push_back(1. / fwScale);
    }

    auto backwardInferShape = [&](ShapeRef shape, MutableArrayRef<int64_t> beginPads,
                                  MutableArrayRef<int64_t> endPads) {
        ArrayRef<int64_t> shapeArray = shape.raw();
        // TODO: E#36318 how to deal with calc-mode = size if scales missed - recalc them somewhere:
        auto shapeCalcMode = IE::InterpolateCalcMode::SCALES;

        auto backwardInferedShape =
                propagateShape(getLoc(), origAxes, shapeArray, {beginPads}, {endPads}, shapeCalcMode,
                               extractIntVector(getLoc(), sizes(), sizes_attr().getValueOr<mlir::ArrayAttr>({})),
                               {backwardScale}, log);
        VPUX_THROW_UNLESS(mlir::succeeded(backwardInferedShape),
                          "InterpolateOp::backInferTileInfo failed to propagate Shape-back");

        // need to run forward propagate in order to adjust pads
        auto forwardInferedShape = propagateShape(
                getLoc(), origAxes, backwardInferedShape.getValue(), {beginPads}, {endPads}, shapeCalcMode,
                extractIntVector(getLoc(), sizes(), sizes_attr().getValueOr<mlir::ArrayAttr>({})), {fwdScales}, log);

        VPUX_THROW_UNLESS(mlir::succeeded(forwardInferedShape),
                          "InterpolateOp::backInferTileInfo failed to propagate Shape-forward");

        // TODO: E#36319 we counting only endpads - begin pad might matter for offsets not for dims
        if (endPads.size() == shapeArray.size()) {
            for (auto shapeOrig : shapeArray | indexed) {
                endPads[shapeOrig.index()] = shapeOrig.value() - forwardInferedShape.getValue()[shapeOrig.index()];
            }
        }

        return backwardInferedShape;
    };

    SmallVector<int64_t> beginPads(4, 0);
    SmallVector<int64_t> endPads(4, 0);

    auto inferedInputTile = backwardInferShape(outputTile.shape, beginPads, endPads);
    if (mlir::failed(inferedInputTile)) {
        return TilingInfo(ArrayRef<TileInfo>());
    }

    auto inferedInputOffset = backwardInferShape(outputTile.offsets, {}, {});
    if (mlir::failed(inferedInputOffset)) {
        return TilingInfo(ArrayRef<TileInfo>());
    }

    TileInfo inputTile{inferedInputTile.getValue().size()};

    inputTile.shape = Shape(inferedInputTile.getValue());
    inputTile.offsets = Shape(inferedInputOffset.getValue());

    auto convertShape = [](::mlir::Value plainShape) {
        auto origShape = getShape(plainShape);
        TileInfo tileShape{origShape.size()};
        tileShape.shape = getShape(plainShape).toValues();
        return tileShape;
    };

    SmallVector<TileInfo> tiles(1, inputTile);

    if (auto size = sizes()) {
        tiles.push_back(convertShape(size));
    }
    if (auto scale = scales()) {
        tiles.push_back(convertShape(scale));
    }
    if (auto axis = axes()) {
        tiles.push_back(convertShape(axis));
    }
    auto iTiling = InputTiling{tiles};

    iTiling.pads = {0, endPads[2], 0, endPads[3]};

    return iTiling;
}

void vpux::VPU::InterpolateOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& /*outTile*/) {
    if (!inputTiling.pads.hasValue()) {
        return;
    }
    mlir::Builder builder(*this);

    SmallVector<int64_t> endPads = {0, 0, 0, 0};
    SmallVector<int64_t> beginPads = {0, 0, 0, 0};

    endPads[2] = inputTiling.pads.getValue().right;
    endPads[3] = inputTiling.pads.getValue().bottom;

    auto newEndPads = builder.getI64ArrayAttr(endPads);
    auto newBeginPads = builder.getI64ArrayAttr(beginPads);

    // forcing scales calculation mode
    auto scalesAttr = vpux::IE::InterpolateCalcModeAttr::get(this->getContext(), IE::InterpolateCalcMode::SCALES);

    auto newAttrs =
            IE::InterpolateAttr::get(attr().mode(), scalesAttr, attr().coord_mode(), attr().nearest_mode(),
                                     attr().antialias(), newBeginPads, newEndPads, attr().cube_coeff(), getContext());

    // set pads begin + end attrs
    attrAttr(newAttrs);
}
