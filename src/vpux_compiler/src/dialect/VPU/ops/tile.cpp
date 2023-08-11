//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

namespace {

mlir::SmallVector<int64_t> calcTileOutputShape(mlir::Value input, llvm::SmallVector<int64_t> repeatsVals) {
    // If number of elements in *"repeats"* is more than shape of *"data"*, then *"data"* will be promoted to
    // "*repeats*" by prepending new axes, e.g. let's shape of *"data"* is equal to (2, 3) and *"repeats"* is equal to
    // [2, 2, 2], then shape of *"data"* will be promoted to (1, 2, 3) and result shape will be (2, 4, 6).
    //
    // If number of elements in *"repeats"* is less than shape of *"data"*, then *"repeats"* will be promoted to
    // "*data*" by prepending 1's to it, e.g. let's shape of *"data"* is equal to (4, 2, 3) and *"repeats"* is equal to
    // [2, 2], then *"repeats"* will be promoted to [1, 2, 2] and result shape will be (4, 4, 6)

    const auto inType = input.getType().cast<vpux::NDTypeInterface>();
    auto outShape = to_small_vector(inType.getShape());

    while (repeatsVals.size() > outShape.size()) {
        outShape.insert(outShape.begin(), 1);
    }

    auto out_shape_iter = std::prev(outShape.end());
    auto repeats_iter = std::prev(repeatsVals.end());
    for (; out_shape_iter != std::prev(outShape.begin()) && repeats_iter != std::prev(repeatsVals.begin());
         --out_shape_iter, --repeats_iter) {
        *out_shape_iter *= *repeats_iter;
    }
    return outShape;
}

}  // namespace

mlir::LogicalResult vpux::VPU::TileOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                        mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                        mlir::RegionRange /*regions*/,
                                                        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::TileOpAdaptor tile(operands, attrs);
    if (mlir::failed(tile.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = tile.input().getType().cast<vpux::NDTypeInterface>();
    auto repeats_vector = parseIntArrayAttr<int64_t>(tile.repeats_values());

    auto outShape = calcTileOutputShape(tile.input(), repeats_vector);

    const auto outType = inType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::VPU::TileOp::fold(ArrayRef<mlir::Attribute>) {
    if (input().getType() == output().getType()) {
        return input();
    }

    return nullptr;
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::TileOp::serialize(EMU::BlobWriter& /*writer*/) {
    VPUX_THROW("Unreacheable code, since all tile ops are converted to VPU::PerAxisTileOp");
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::TileOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    auto original_input_shape = input().getType().cast<vpux::NDTypeInterface>().getShape().raw();
    auto tiled_output_shape = outputTile.shape.raw();
    auto tiled_output_offsets = outputTile.offsets.raw();
    auto tiled_output_axis = outputTile.axis.raw();
    auto repeats_values = parseIntArrayAttr<int64_t>(repeats_valuesAttr());
    Shape suggested_new_input_shape, suggested_offS, suggested_axis;
    const auto isZero = [](int64_t number) {
        return (number == 0 ? 1 : number);
    };

    // adjust and calc input shape depending on tiled output
    for (size_t i = 0; i < original_input_shape.size(); i++) {
        auto repeats_dimension = tiled_output_shape[i] % original_input_shape[i];
        if (repeats_dimension != 0) {
            // need to tile input and repeats too
            int64_t new_repeats_d = isZero(repeats_values[i] - 1);
            while (tiled_output_shape[i] % new_repeats_d != 0)
                new_repeats_d--;
            suggested_new_input_shape.insert(suggested_new_input_shape.end(), tiled_output_shape[i] / new_repeats_d);
        } else {
            // no need to tile from inputs
            suggested_new_input_shape.insert(suggested_new_input_shape.end(), original_input_shape[i]);
        }
        // set axis and offsets
        suggested_offS.insert(suggested_offS.end(), tiled_output_offsets[i] % original_input_shape[i]);
        suggested_axis.insert(suggested_axis.end(), tiled_output_axis[i]);
    }
    TileInfo suggestedInput(suggested_new_input_shape, suggested_offS, suggested_axis);

    return TilingInfo{{suggestedInput}};
}

void vpux::VPU::TileOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& outputTile) {
    auto inputTilingTiles = inputTiling.tiles[0];
    auto outputShape = outputTile.shape.raw();
    auto suggestedInShape = inputTilingTiles.shape.raw();

    // repeats values is eq with output shape divided by input shape
    SmallVector<int64_t> new_repeats;
    for (size_t i = 0; i < repeats_values().getValue().size(); i++) {
        new_repeats.push_back(outputShape[i] / suggestedInShape[i]);
    }

    auto newRepeatsAttr = getIntArrayAttr(getContext(), new_repeats);
    repeats_valuesAttr(newRepeatsAttr);
}

OutputTiling vpux::VPU::TileOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto op = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for SW currently, for op {0} at '{1}'", op->getName(),
                    op->getLoc());

    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());

    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    op->getName());

    const auto inputType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape();
    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();

    Shape nTilesOnDim(outputShape.size(), 1);

    const auto tileDimOrder = getTileDimOrder(op, tilingMode, log);
    log.trace("Tile Dim order is {0}", tileDimOrder);

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;

    const auto isSupportedTileSize = [op, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                         TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
        return tilingInfo.isSupportedTiling(tiles, tilingMode, log);
    };

    const auto& maxNumTiles = tilingBuilder.getMaxNumTiles();
    const auto isDimLeftToTile = [&](ShapeRef tileShape) -> bool {
        return (tileShape[dimToTile] < maxNumTiles[dimToTile.ind()]) ||
               (tileShape[dimToTile] < outputShape[dimToTile] / inputShape[dimToTile]);
    };

    const auto isLegalRepeat = [&](ShapeRef tileShape) -> bool {
        return (inputShape[dimToTile] * tileShape[dimToTile] == outputShape[dimToTile]) || (inputShape[dimToTile] == 1);
    };

    // Get an feasible isolated tiling strategy
    while (!isSupportedTileSize(nTilesOnDim, tilingMode) || !isLegalRepeat(nTilesOnDim)) {
        while (((tileDimIter < tileDimOrder.end()) && (!isDimLeftToTile(nTilesOnDim))) ||
               (nTilesOnDim[dimToTile] == outputShape[dimToTile] / inputShape[dimToTile])) {
            dimToTile = *(++tileDimIter);
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
