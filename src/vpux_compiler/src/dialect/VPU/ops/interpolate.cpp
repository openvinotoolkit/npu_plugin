//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

namespace {

mlir::FailureOr<SmallVector<int64_t>> extractIntVector(mlir::Location loc, const mlir::Value value,
                                                       const Optional<mlir::ArrayAttr>& attr) {
    if (attr.hasValue() && attr.getValue() != nullptr) {
        return parseIntArrayAttr<int64_t>(attr.getValue());
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

mlir::FailureOr<SmallVector<double>> extractFPVector(mlir::Location loc, const mlir::Value value,
                                                     const Optional<mlir::ArrayAttr>& attr) {
    if (attr.hasValue() && attr.getValue() != nullptr) {
        return parseFPArrayAttr<double>(attr.getValue());
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

mlir::FailureOr<SmallVector<int64_t>> calcOutputShapes(mlir::Location loc, VPU::InterpolateOpAdaptor interpolate) {
    const auto axes = extractIntVector(loc, interpolate.axes(), interpolate.axes_attr());
    const auto axesVal = axes.getValue();

    const auto inType = interpolate.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inType.getShape().raw();
    SmallVector<int64_t> outShape;

    for (size_t i = 0; i < inputShape.size(); ++i) {
        outShape.emplace_back(inputShape[i]);
    }

    auto calcMode = interpolate.attr().getShapeCalcMode().getValue();

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

    } else if (calcMode == IE::InterpolateCalcMode::SCALES) {
        const auto scales = extractFPVector(loc, interpolate.scales(), interpolate.scales_attr());
        const auto scales_val = scales.getValue();

        if (scales_val.size() != axesVal.size()) {
            return errorAt(loc,
                           "Num of elements in scales tensor: {0} should be equal to number of indices in axes: {1}",
                           scales_val.size(), axesVal.size());
        }

        auto scalesIter = scales_val.begin();

        auto inputShapeArray = to_small_vector(inputShape);
        auto initInputDim = interpolate.initial_input_dims_attr().hasValue()
                                    ? parseIntArrayAttr<int64_t>(interpolate.initial_input_dims_attr().getValue())
                                    : inputShapeArray;
        for (const auto& i : axesVal) {
            outShape[i] = static_cast<int64_t>(floor((*scalesIter++) * inputShape[i]));
        }

    } else
        return errorAt(loc, "Doesn't support shape_calculation_mode: {0}", calcMode);

    auto padsBegin = parseIntArrayAttr<int64_t>(interpolate.attr().getPadsBegin());
    auto padsEnd = parseIntArrayAttr<int64_t>(interpolate.attr().getPadsEnd());
    applyInterpPads(outShape, padsBegin, padsEnd);

    return outShape;
}

}  // namespace

mlir::LogicalResult vpux::VPU::InterpolateOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               mlir::Optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

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

void vpux::VPU::InterpolateOp::inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info) {
    auto inputShape = op->getOperand(0).getType().cast<vpux::NDTypeInterface>().getShape().raw();
    VPUX_THROW_UNLESS(inputShape.size() == 4, "Interpolate input shape expected to have 4 dimensions, but has {0}",
                      inputShape.size());

    // Select NCHW layout due to performance reasons
    // [Track number: E#25302]
    auto channels = inputShape[Dims4D::Act::C.ind()];
    const auto antialias = mlir::cast<IE::InterpolateOp>(op).attr().getAntialias().getValue();
    if (channels == 1 || antialias) {
        VPU::inferLayoutInfoSameInOutSpecificDimsOrder(info, {DimsOrder::NCHW});
    } else {
        VPU::inferLayoutInfoSameInOutSpecificDimsOrder(info, {DimsOrder::NCHW, DimsOrder::NHWC});
    }
}

//
// ClusteredOpInterface
//

bool vpux::VPU::InterpolateOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }
    const auto coordMode = attr().getCoordMode().getValue();
    const auto mode = attr().getMode().getValue();
    // E#67003, note that currenly only enable multi cluster when mode is linear_onnx and coord mode is half pixel
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped && mode == IE::InterpolateMode::LINEAR_ONNX &&
        (coordMode == IE::InterpolateCoordMode::HALF_PIXEL || coordMode == IE::InterpolateCoordMode::ALIGN_CORNERS ||
         coordMode == IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL)) {
        auto inputShape = getShape(input());
        return inputShape[Dims4D::Act::H] > 1;
    }
    return false;
}

void vpux::VPU::InterpolateOp::build(
        ::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, ::mlir::Value input,
        /*optional*/ ::mlir::Value sizes, /*optional*/ ::mlir::Value scales, /*optional*/ ::mlir::Value axes,
        /*optional*/ ::mlir::ArrayAttr sizes_attr, /*optional*/ ::mlir::ArrayAttr scales_attr,
        /*optional*/ ::mlir::ArrayAttr axes_attr, /*optional*/ ::mlir::ArrayAttr tile_offset_attr,
        /*optional*/ ::mlir::ArrayAttr initial_input_dims_attr, /*optional*/ ::mlir::ArrayAttr initial_output_dims_attr,
        vpux::IE::InterpolateAttr attr) {
    build(odsBuilder, odsState, input, sizes, scales, axes, sizes_attr, scales_attr, axes_attr, tile_offset_attr,
          initial_input_dims_attr, initial_output_dims_attr, nullptr, nullptr, nullptr, attr);
}

//
// SWOpInterface
//

bool vpux::VPU::InterpolateOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() >= 2 && buffers.size() <= 5,
                      "InterpolateOp requires 1 input, 3 optional attribution inputs and 1 output, but the "
                      "number of buffer is {0}",
                      buffers.size());

    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::InterpolateOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::InterpolateOp::supportCycleCostCalculation() {
    return false;
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::InterpolateOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::InterpolateParamsBuilder builder(writer);

    const auto mode = attr().getMode().getValue();
    const auto coord_mode = attr().getCoordMode().getValue();
    const auto nearest_mode = attr().getNearestMode().getValue();
    const auto antialias = attr().getAntialias().getValue();

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

    } else {
        return errorAt(loc, "Doesn't support shape_calculation_mode: {0}", calcMode);
    }

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

    auto iShape = initial_input_dims_attr().hasValue()
                          ? parseIntArrayAttr<int64_t>(initial_input_dims_attr().getValue())
                          : to_small_vector(getShape(input()));
    auto oShape = initial_output_dims_attr().hasValue()
                          ? parseIntArrayAttr<int64_t>(initial_output_dims_attr().getValue())
                          : to_small_vector(getShape(output()));
    auto initialInputOffsets = initial_input_offset_attr().hasValue()
                                       ? parseIntArrayAttr<int64_t>(initial_input_offset_attr().getValue())
                                       : SmallVector<int64_t>(getShape(input()).size(), 0);

    auto initialOutputOffsets = initial_output_offset_attr().hasValue()
                                        ? parseIntArrayAttr<int64_t>(initial_output_offset_attr().getValue())
                                        : SmallVector<int64_t>(getShape(output()).size(), 0);

    mlir::Builder builder(*this);
    if (!initial_input_dims_attr().hasValue()) {
        auto newInitialInputDims = builder.getI64ArrayAttr(iShape);
        initial_input_dims_attrAttr(newInitialInputDims);
    }
    if (!initial_output_dims_attr().hasValue()) {
        auto newInitialOutputDims = builder.getI64ArrayAttr(oShape);
        initial_output_dims_attrAttr(newInitialOutputDims);
    }

    SmallVector<double> tileOffset(iShape.size(), 0.f);
    auto newTileOffset = builder.getF64ArrayAttr(tileOffset);
    tile_offset_attrAttr(newTileOffset);

    const auto axesVal = origAxes.getValue();
    vpux::Scales fwdScales;
    // Compute scale-factors based on full I/O resolution ratio
    SmallVector<double> backwardScale;
    for (size_t i = 0; i < axesVal.size(); i++) {
        backwardScale.push_back(static_cast<double>(iShape[axesVal[i]]) / oShape[axesVal[i]]);
    }

    SmallVector<int64_t> beginPads(iShape.size(), 0);
    SmallVector<int64_t> endPads(iShape.size(), 0);

    mlir::FailureOr<SmallVector<int64_t>> inferedInputTile;
    auto coordMode = attr().getCoordMode().getValue();
    auto inTiles = vpux::backInferInterpolateTile(outputTile, iShape, oShape, initialInputOffsets, initialOutputOffsets,
                                                  coordMode, log);
    auto newInputOffset = to_small_vector(inTiles.tiles[0].offsets);

    // Recalculate the backward scale based on the new input/output shape
    for (size_t i = 0; i < axesVal.size(); i++) {
        fwdScales.push_back(static_cast<double>(outputTile.shape[Dim(axesVal[i])]) /
                            inTiles.tiles[0].shape[Dim(axesVal[i])]);
    }

    auto shapeCalcMode = IE::InterpolateCalcMode::SCALES;
    auto forwardInferedShape = propagateShape(
            getLoc(), origAxes, inTiles.tiles[0].shape.raw(), {beginPads}, {endPads}, shapeCalcMode,
            extractIntVector(getLoc(), sizes(), sizes_attr().value_or<mlir::ArrayAttr>({})), {fwdScales}, log);

    VPUX_THROW_UNLESS(mlir::succeeded(forwardInferedShape),
                      "InterpolateOp::backInferTileInfo failed to propagate Shape-forward");

    // TODO: E#36319 we counting only endpads - begin pad might matter for offsets not for dims
    auto shapeArray = to_small_vector(outputTile.shape);
    if (endPads.size() == shapeArray.size()) {
        for (auto& shapeOrig : shapeArray | indexed) {
            endPads[shapeOrig.index()] = shapeOrig.value() - forwardInferedShape.getValue()[shapeOrig.index()];
        }
    }
    auto convertShape = [](::mlir::Value plainShape) {
        auto origShape = getShape(plainShape);
        TileInfo tileShape{origShape.size()};
        tileShape.shape = getShape(plainShape).toValues();
        return tileShape;
    };

    if (auto size = sizes()) {
        inTiles.tiles.push_back(convertShape(size));
    }
    if (auto scale = scales()) {
        inTiles.tiles.push_back(convertShape(scale));
    }
    if (auto axis = axes()) {
        inTiles.tiles.push_back(convertShape(axis));
    }
    inTiles.pads = {0, endPads[2], 0, endPads[3]};
    return inTiles;
}

void vpux::VPU::InterpolateOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& outTile) {
    if (!inputTiling.pads.hasValue()) {
        return;
    }
    mlir::Builder builder(*this);

    TileInfo inputTile = inputTiling.tiles.begin()[0];
    const auto origInputDims = extractIntVector(getLoc(), axes(), axes_attrAttr());
    const auto initialInputDims = parseIntArrayAttr<int64_t>(initial_input_dims_attrAttr());
    const auto initialOutputDims = parseIntArrayAttr<int64_t>(initial_output_dims_attrAttr());
    const auto inputOrder = DimsOrder::fromValue(input());

    const auto initialInputOffset = builder.getI64ArrayAttr(to_small_vector(inputTiling.tiles[0].offsets));
    const auto initialOutputOffset = builder.getI64ArrayAttr(to_small_vector(outTile.offsets));
    initial_input_offset_attrAttr(initialInputOffset);
    initial_output_offset_attrAttr(initialOutputOffset);

    const auto numDims = initialInputDims.size();

    SmallVector<double> tileOffset(numDims, 0.f);
    const auto coordMode = attr().getCoordMode().getValue();
    if (coordMode == IE::InterpolateCoordMode::ALIGN_CORNERS) {
        const auto oShape = output().getType().cast<vpux::NDTypeInterface>().getShape().raw();

        int indiceH, indiceW;
        if (inputOrder == DimsOrder::NCHW) {
            indiceH = 2;
            indiceW = 3;
        } else if (inputOrder == DimsOrder::NHWC) {
            indiceH = 2;
            indiceW = 1;
        } else {
            return;
        }

        const auto IH = initialInputDims[indiceH];
        const auto IW = initialInputDims[indiceW];

        const auto OH = oShape[indiceH];
        const auto OW = oShape[indiceW];
        if (IH > OH || IW > OW) {
            double H_offset = (outTile.offsets.begin()[indiceH] * static_cast<double>(IH - 1) / (OH - 1)) -
                              inputTile.offsets.begin()[indiceH];
            double W_offset = (outTile.offsets.begin()[indiceW] * static_cast<double>(IW - 1) / (OW - 1)) -
                              inputTile.offsets.begin()[indiceW];
            VPUX_THROW_UNLESS(H_offset >= 0.0 && W_offset >= 0.0, "Invalid tile calculation");

            tileOffset[indiceH] = H_offset;
            tileOffset[indiceW] = W_offset;
        }
    }

    auto newTileOffset = builder.getF64ArrayAttr(tileOffset);
    tile_offset_attrAttr(newTileOffset);

    SmallVector<int64_t> endPads(numDims, 0);
    SmallVector<int64_t> beginPads(numDims, 0);

    endPads[2] = inputTiling.pads.getValue().right;
    endPads[3] = inputTiling.pads.getValue().bottom;

    auto newEndPads = builder.getI64ArrayAttr(endPads);
    auto newBeginPads = builder.getI64ArrayAttr(beginPads);

    // forcing scales calculation mode
    auto calcModeAttr = vpux::IE::InterpolateCalcModeAttr::get(this->getContext(), IE::InterpolateCalcMode::SCALES);

    auto newAttrs = IE::InterpolateAttr::get(getContext(), attr().getMode(), calcModeAttr, attr().getCoordMode(),
                                             attr().getNearestMode(), attr().getAntialias(), newBeginPads, newEndPads,
                                             attr().getCubeCoeff());

    auto axesValue = extractIntVector(getLoc(), axes(), axes_attr().value_or<mlir::ArrayAttr>({})).getValue();
    auto scale = SmallVector<double>(axesValue.size(), 1);
    // Recompute SCALE attribute based on new input output tiling
    for (auto& axis : axesValue | indexed) {
        const auto axisDim = Dim(axis.value());
        scale[axis.index()] = static_cast<double>(outTile.shape[axisDim]) / inputTiling.tiles[0].shape[axisDim];
    }

    // set pads begin + end attrs
    attrAttr(newAttrs);
    scales_attrAttr(builder.getF64ArrayAttr(scale));
}

// Generates a list of tiles of the output tensor that satisfy the CMX memory constrains and don't alter the original
// scaling factors. When generating the tiles the following characteristics are taken into consideration:
// 1. The data layout, in order to find an optimal manner of transferring the data between CMX and DDR
//    NCHW -> the tiling dim order is C, H, W
//    NHWC -> the tiling dim order is H, W, C
// 2. Scaling factor, based on value of the scaling factor is in one axis, the following decisions are taken:
//    If the greatest common factor between in dim and out dim is 1 then remove axis from tiling axis
//    If the greatest common factor > 1 then choose the smallest common factor that satisfies constrains
//    If in and out dim are equal (scaling factor == 1) then iteratively increase the number of tiles on this axis since
//    the constrains are satisfied or the number of tiles is equal with minimum value between in and out dim
OutputTiling vpux::VPU::InterpolateOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto op = this->getOperation();
    auto origOp = mlir::dyn_cast<VPU::InterpolateOp>(op);
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());
    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    op->getName());
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for SW currently, for op {0} at '{1}'", op->getName(),
                    op->getLoc());

    const auto inType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto inShapeArray = inType.getShape().raw();
    const auto outType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    const auto outShape = outType.getShape();
    const auto outShapeArray = outType.getShape().raw();
    const auto axes = extractIntVector(op->getLoc(), origOp.axes(), origOp.axes_attr());
    const auto axesVal = axes.getValue();
    Shape nTilesOnDim(outShapeArray.size(), 1);

    auto getTileDimOrder = [&]() {
        VPUX_THROW_UNLESS(outType.getDimsOrder() == DimsOrder::NCHW || outType.getDimsOrder() == DimsOrder::NHWC,
                          "Interpolate Op only support NCHW and NHWC layout, but got '{0}'", outType.getDimsOrder());

        auto dimOrder = outType.getDimsOrder().toPermutation();
        // Get greatest common factor between two values
        auto getGreaterCommonFactor = [&](int64_t firstVal, int64_t secondVal) {
            while (firstVal != secondVal) {
                if (firstVal > secondVal) {
                    firstVal = firstVal - secondVal;
                } else {
                    secondVal = secondVal - firstVal;
                }
            }
            return firstVal;
        };

        for (auto& axis : axesVal | indexed) {
            const auto axisDim = Dim(axis.value());
            // Check if greatest common factor between current in and out dim is 1 and if yes remove the axis from the
            // list of tiling dims
            if (getGreaterCommonFactor(inShapeArray[axisDim.ind()], outShapeArray[axisDim.ind()]) == 1) {
                llvm::erase_if(dimOrder, [axisDim](Dim dim) {
                    return (dim == axisDim);
                });
            }
        }

        log.nest(2).debug("getTilingStrategy: dimOrder = {0}", dimOrder);
        return dimOrder;
    };

    const auto tileDimOrder = getTileDimOrder();

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;

    const auto isSupportedTileSize = [op, &tilingInfo, outShape, log](ShapeRef nTilesOnDim,
                                                                      TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outShape);
        return tilingInfo.isSupportedTiling(tiles, tilingMode, log);
    };
    // Compute next common factor of two integer values greater or equal then currFactor
    // if it doesn't exist getNextCommonFactor returns 0
    auto getNextCommonFactor = [&](int64_t firstVal, int64_t secondVal, int64_t currFactor) -> int64_t {
        auto minVal = std::min(firstVal, secondVal);
        while (currFactor < minVal) {
            currFactor++;
            if (firstVal % currFactor == 0 && secondVal % currFactor == 0) {
                return currFactor;
            }
        }
        return 0;
    };

    // Get an feasible isolated tiling strategy
    while (!isSupportedTileSize(nTilesOnDim, tilingMode)) {
        while (tileDimIter < tileDimOrder.end() &&
               nTilesOnDim[dimToTile] >=
                       std::min(inShapeArray[(*tileDimIter).ind()], outShapeArray[(*tileDimIter).ind()])) {
            dimToTile = *(++tileDimIter);
        }
        VPUX_THROW_WHEN(tileDimIter == tileDimOrder.end(), "Failed to tile {0} at '{1}'", op->getName(), op->getLoc());
        // Check if current tiling axis is an interpolation axis and if yes increase the number of tiles to
        // next common factor between in and out dim on current axis
        auto axesIt = llvm::find(axesVal, (*tileDimIter).ind());
        if (axesIt != axesVal.end() && inShapeArray[*axesIt] != outShapeArray[*axesIt]) {
            // If getNextCommonFactor returns > 0 then updates the number of tiles in current axis to returned value
            // else move to next axis considered for tiling
            auto nextFactor =
                    getNextCommonFactor(inShapeArray[*axesIt], outShapeArray[*axesIt], nTilesOnDim[dimToTile]);
            if (nextFactor > 0) {
                nTilesOnDim[Dim((*axesIt))] = nextFactor;
            } else {
                dimToTile = *(++tileDimIter);
            }
        } else {
            ++nTilesOnDim[dimToTile];
        }
    }

    log.trace("Isolated tiling strategy: {0}", nTilesOnDim);
    return fillDividedTiles(op, nTilesOnDim, outShape);
}
