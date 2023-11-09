//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::NormalizeL2Op::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               mlir::Optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::NormalizeL2OpAdaptor normalizeL2(operands, attrs);
    if (mlir::failed(normalizeL2.verify(loc))) {
        return mlir::failure();
    }

    auto axes = IE::constInputToData(loc, normalizeL2.axes());
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    const auto inType = normalizeL2.data().getType().cast<vpux::NDTypeInterface>();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::NormalizeL2Op::serialize(EMU::BlobWriter&) {
    VPUX_THROW("NormalizeL2Op implemented just on 37xx.");
}

//
// verify
//

mlir::LogicalResult vpux::VPU::NormalizeL2Op::verify() {
    const auto inRank = data().getType().cast<vpux::NDTypeInterface>().getRank();
    auto axesVec = parseIntArrayAttr<int64_t>(vpux::IE::getIntArrayAttrValue(axes()));

    for (auto& axis : axesVec) {
        if (axis < 0) {
            axis += inRank;
        }
    }

    bool isAllUnique = std::unique(axesVec.begin(), axesVec.end()) == axesVec.end();
    if (!isAllUnique) {
        return errorAt(*this, "Axes values should be unique");
    }

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::NormalizeL2Op::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    TileInfo axesShape(getShape(axes()));
    return InputTiling{{outputTile, axesShape}};
}

void vpux::VPU::NormalizeL2Op::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::NormalizeL2Op::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto op = this->getOperation();
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

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;

    const auto isSupportedTileSize = [op, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                         TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    const auto& maxNumTiles = tilingBuilder.getMaxNumTiles();
    const auto isDimLeftToTile = [&](ShapeRef tileShape) -> bool {
        return tileShape[dimToTile] < maxNumTiles[dimToTile.ind()];
    };

    auto axesConst = axes().getDefiningOp<Const::DeclareOp>().getContent();
    auto axesValues = to_small_vector(axesConst.getValues<int64_t>());

    const auto isSpecifiedAxis = [&](const vpux::Dim* tiledDim) -> bool {
        // cases when there is no chance to skip dims, all is specified => default tiling mode
        if (axesValues.size() == outputShape.size()) {
            return false;
        } else {
            // cases when there is chance to tile other dims than specified
            for (int64_t axis : axesValues) {
                if (axis == static_cast<int64_t>(tiledDim->ind())) {
                    return true;
                }
            }
        }
        return false;
    };

    while (!isSupportedTileSize(nTilesOnDim, tilingMode)) {
        while (((tileDimIter < tileDimOrder.end()) && (!isDimLeftToTile(nTilesOnDim))) ||
               isSpecifiedAxis(tileDimIter)) {
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
