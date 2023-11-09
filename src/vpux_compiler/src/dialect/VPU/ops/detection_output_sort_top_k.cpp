//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

//
// inferReturnTypes
//

mlir::LogicalResult VPU::DetectionOutputSortTopKOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DetectionOutputSortTopKOpAdaptor sortTopK(operands, attrs);
    if (mlir::failed(sortTopK.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = sortTopK.class_predictions().getType().cast<NDTypeInterface>();
    const auto inputShape = inputType.getShape();

    const auto batchSize = inputShape[Dim(0)];
    const auto paddedOne = inputShape[Dim(1)];  // filler to have 4D tensor
    const auto numClasses = inputShape[Dim(2)];
    const auto numPriors = inputShape[Dim(3)];

    VPUX_THROW_UNLESS(paddedOne == 1, "DetectionOutput ClassPredictions tensor has unexpected shape");

    const auto topK = sortTopK.top_k();

    const auto outTopKConfidenceShape = SmallVector<int64_t>{batchSize, numClasses, topK};
    const auto outIndicesShape = SmallVector<int64_t>{batchSize, numClasses, numPriors};
    const auto outSizesShape = SmallVector<int64_t>{batchSize, numClasses};

    const auto outTopKConfidenceType = inputType.changeShape(Shape(outTopKConfidenceShape));

    const auto outIndicesElemType = mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
    const auto outIndicesType = mlir::RankedTensorType::get(outIndicesShape, outIndicesElemType);
    const auto outSizesType = mlir::RankedTensorType::get(outSizesShape, outIndicesElemType);

    inferredReturnTypes.push_back(outTopKConfidenceType);
    inferredReturnTypes.push_back(outIndicesType);
    inferredReturnTypes.push_back(outSizesType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask VPU::DetectionOutputSortTopKOp::serialize(EMU::BlobWriter&) {
    VPUX_THROW("VPU::DetectionOutputSortTopKOp is not supported by EMU");
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::DetectionOutputSortTopKOp::backInferTileInfo(const vpux::TileInfo& firstOutputTileInfo,
                                                                    vpux::Logger /*log*/) {
    const auto outputShape = firstOutputTileInfo.shape;
    VPUX_THROW_UNLESS(outputShape.size() == 3, "Expected 3D output shape to be tiled");

    const auto indicesType = out_indices().getType().cast<vpux::NDTypeInterface>();
    const auto indicesShape = indicesType.getShape();
    const auto numPriors = indicesShape.back();

    const auto outputOffsets = firstOutputTileInfo.offsets;

    const auto batchesDims = outputShape[Dim(0)];
    const auto batchesOffsets = outputOffsets[Dim(0)];

    const auto classesDims = outputShape[Dim(1)];
    const auto classesOffsets = outputOffsets[Dim(1)];

    const auto inputRank = 4;
    auto inputTile = TileInfo(inputRank);
    inputTile.shape = Shape{batchesDims, 1, classesDims, numPriors};
    inputTile.offsets = Shape{batchesOffsets, 0, classesOffsets, 0};

    return InputTiling{inputTile};
}

void vpux::VPU::DetectionOutputSortTopKOp::adjustAttrs(const TilingInfo&, const TileInfo& outputTile) {
    const auto outputOffsets = outputTile.offsets;
    VPUX_THROW_UNLESS(outputOffsets.size() == 3,
                      "Expected 3D shape for the first output of DetectionOutputSortTopK layer, got {0}",
                      outputOffsets.size());

    const auto classOffset = outputOffsets[Dim(1)];
    const auto shiftedBackgroundId = background_id() - classOffset;
    const auto newBackgroundIdAttr = getIntAttr(getContext(), shiftedBackgroundId);

    background_idAttr(newBackgroundIdAttr);
}

OutputTiling vpux::VPU::DetectionOutputSortTopKOp::getOutputTiling(const vpux::TileInfo& firstOutputTile,
                                                                   vpux::Logger /*log*/) {
    // Output 0 top_k_confidence    [ 1, numClasses, numBoxes ]
    // Output 1 indices             [ 1, numClasses, numPriors ]
    // Output 2 sizes               [ 1, numClasses ]
    const auto shapeBatch = firstOutputTile.shape[Dim(0)];
    const auto shapeClasses = firstOutputTile.shape[Dim(1)];
    const auto offsetBatch = firstOutputTile.offsets[Dim(0)];
    const auto offsetClasses = firstOutputTile.offsets[Dim(1)];

    const auto indicesType = out_indices().getType().cast<vpux::NDTypeInterface>();
    const auto indicesShape = indicesType.getShape();
    const auto indicesWidth = indicesShape.back();

    auto indicesTile = TileInfo(indicesShape.size());
    indicesTile.shape = Shape{shapeBatch, shapeClasses, indicesWidth};
    indicesTile.offsets = Shape{offsetBatch, offsetClasses, 0};

    const auto sizesShapeSize = 2;
    auto sizesTile = TileInfo(sizesShapeSize);
    sizesTile.shape = Shape{shapeBatch, shapeClasses};
    sizesTile.offsets = Shape{offsetBatch, offsetClasses};

    return OutputTiling{firstOutputTile, std::move(indicesTile), std::move(sizesTile)};
}

DimArr getTilingOrder(mlir::Operation* op, TilingMode tilingMode, ShapeRef tilingBounds, Logger log) {
    auto tileDimOrder = getTileDimOrder(op, tilingMode, log);

    const auto inBounds = [rank = checked_cast<int32_t>(tilingBounds.size())](const Dim dim) {
        return dim.ind() < rank;
    };

    VPUX_THROW_UNLESS(llvm::all_of(tileDimOrder, inBounds),
                      "'{0}' at '{1}' has incorrect tiling order '{2}' for defined tiling space '{3}'", op->getName(),
                      op->getLoc(), tileDimOrder, tilingBounds);

    return tileDimOrder;
}

llvm::Optional<Shape> getOutputTilingStrategy(mlir::Operation* op, mlir::Value output, TilingMode tilingMode,
                                              Logger log) {
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());

    const auto tilingBounds = getShape(output);
    auto totalTilesPerDim = Shape(tilingBounds.size(), 1);

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim, TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, tilingBounds);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.getValue(), tilingMode, log);
    };

    const auto tilingOrder = getTilingOrder(op, tilingMode, tilingBounds, log);
    for (const auto& dimToTile : tilingOrder) {
        auto& tilesOnCurrentDim = totalTilesPerDim[dimToTile];
        for (; tilesOnCurrentDim < tilingBounds[dimToTile]; ++tilesOnCurrentDim) {
            if (isSupportedTileSize(totalTilesPerDim, tilingMode)) {
                return totalTilesPerDim;
            }
        }
    }

    return llvm::None;
}

mlir::FailureOr<OutputTiling> vpux::VPU::DetectionOutputSortTopKOp::getTilingStrategy(TilingMode tilingMode,
                                                                                      Logger log) {
    auto* const op = getOperation();

    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for SW currently, for op {0} at '{1}'", op->getName(),
                    op->getLoc());

    const auto firstOutput = op->getResult(0);

    const auto tilingStrategy = getOutputTilingStrategy(op, firstOutput, tilingMode, log);
    VPUX_THROW_UNLESS(tilingStrategy.hasValue(), "Failed to tile {0} at '{1}'", op->getName(), op->getLoc());

    log.trace("Isolated tiling strategy: {0}", tilingStrategy);
    return vpux::fillDividedTiles(op, tilingStrategy.getValue(), getShape(firstOutput));
}
