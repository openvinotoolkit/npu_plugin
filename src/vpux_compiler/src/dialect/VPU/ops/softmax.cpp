//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::SoftMaxOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                           mlir::Optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::SoftMaxOpAdaptor softMax(operands, attrs);
    if (mlir::failed(softMax.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = softMax.input().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::SoftMaxOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::SoftmaxParamsBuilder builder(writer);
    builder.add_axis(checked_cast<uint32_t>(axisInd()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_SoftmaxParams});
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::SoftMaxOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    return TilingInfo(outputTile);
}

void vpux::VPU::SoftMaxOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

// Based on getSWLayerTilingStrategy logic, need to avoid tiling on Softmax dimension.

OutputTiling vpux::VPU::SoftMaxOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto op = getOperation();
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());
    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    op->getName());
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for SW currently, for op {0} at '{1}'", op->getName(),
                    op->getLoc());

    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();

    Shape nTilesOnDim(outputShape.size(), 1);

    const auto tileDimOrder = getTileDimOrder(op, tilingMode, log);
    log.nest(2).trace("Tile Dim order is {0}", tileDimOrder);

    auto axis = axisIndAttr().getValue().getSExtValue();
    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;
    if (dimToTile == Dim(axis)) {
        dimToTile = *(++tileDimIter);
    }

    const auto isSupportedTileSize = [op, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                         TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
        return tilingInfo.isSupportedTiling(tiles, tilingMode, log);
    };

    const auto& maxNumTiles = tilingBuilder.getMaxNumTiles();
    const auto isDimLeftToTile = [&](ShapeRef tileShape) -> bool {
        return tileShape[dimToTile] < maxNumTiles[dimToTile.ind()];
    };

    // Get an feasible isolated tiling strategy
    while (!isSupportedTileSize(nTilesOnDim, tilingMode)) {
        while ((tileDimIter < tileDimOrder.end()) && (!isDimLeftToTile(nTilesOnDim))) {
            dimToTile = *(++tileDimIter);
            if (dimToTile == Dim(axis)) {
                dimToTile = *(++tileDimIter);
            }
            if (tileDimIter == tileDimOrder.end()) {
                VPUX_THROW_WHEN(tilingMode == TilingMode::ISOLATED, "Failed to tile {0} at '{1}'", op->getName(),
                                op->getLoc());
            }
        }
        ++nTilesOnDim[dimToTile];
    }

    log.trace("Isolated tiling strategy: {0}", nTilesOnDim);
    return fillDividedTiles(op, nTilesOnDim, outputShape);
}

//
// SWOpInterface
//

Dim getHighestDim(vpux::VPU::SoftMaxOp sofmaxOp) {
    const auto input = sofmaxOp.input();
    const auto inputType = input.getType().cast<vpux::NDTypeInterface>();
    const auto inOrder = inputType.getDimsOrder();
    const auto inShape = inputType.getShape();
    for (auto i : irange(inOrder.numDims())) {
        auto dim = inOrder.dimAt(i);
        if (inShape[dim] > 1) {
            return dim;
        }
    }
    return inOrder.dimAt(0);
};

bool vpux::VPU::SoftMaxOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy) {
    const auto inputType = input().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inputType.getShape();
    const auto highestDim = getHighestDim(*this);

    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }

    // MC strategy should be segmented on highest dimension to prevent DDR2DDR copy in act Shave kernel tiling.
    // This limitation can be removed when stride access is supported.
    // TODO:[E76529]Softmax kernel support stride access.

    // Split input/output by H dim when axisInd is not point to H and H is the highest dimension
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight && axisInd() != Dims4D::Act::H.ind() &&
        highestDim == Dims4D::Act::H && inShape[Dims4D::Act::H] > 1) {
        return true;
    }

    // Split input/output by C dim when axisInd is not point to C and C is the highest dimension
    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel && axisInd() != Dims4D::Act::C.ind() &&
        highestDim == Dims4D::Act::C && inShape[Dims4D::Act::C] > 1) {
        return true;
    }

    return false;
}

void vpux::VPU::SoftMaxOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, ::mlir::Value input,
                                 ::mlir::IntegerAttr axisInd) {
    build(odsBuilder, odsState, input.getType(), input, axisInd, {});
}

bool vpux::VPU::SoftMaxOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 2, "SoftMaxOp requires 1 input and 1 output, but the number of buffer is {0}",
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

bool vpux::VPU::SoftMaxOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

// Cost model might assign SOK for SoftMax in below:
// VPU.SoftMax(%arg0) {axisInd = 3} : tensor<1x8x4x76xf16, {order = #NHWC}> -> tensor<1x8x4x76xf16, {order = #NHWC}>
// Cost model strategy is not used because MC strategy should be segmented on highest dimension to prevent DDR2DDR copy
// in act Shave kernel tiling. This limitation can be removed when stride access is supported.
// TODO:[E76529]Softmax kernel support stride access.

bool vpux::VPU::SoftMaxOp::supportCycleCostCalculation() {
    return false;
}
