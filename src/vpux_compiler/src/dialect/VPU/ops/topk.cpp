//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::TopKOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                        mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                        mlir::RegionRange /*regions*/,
                                                        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::TopKOpAdaptor topK(operands, attrs);
    if (mlir::failed(topK.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = topK.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inType.getShape().raw();

    auto reshapeK = topK.k().getDefiningOp<VPU::ReshapeOp>();
    auto kConst = (reshapeK != nullptr) ? reshapeK.input().getDefiningOp<Const::DeclareOp>()
                                        : topK.k().getDefiningOp<Const::DeclareOp>();
    if (kConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for k");
    }

    const auto kContent = kConst.content();
    if (!kContent.isSplat()) {
        return errorAt(loc, "K input must be scalar");
    }

    SmallVector<int64_t> outShape;
    for (size_t i = 0; i < inputShape.size(); ++i) {
        outShape.push_back(inputShape[i]);
    }
    int64_t axis = topK.axis();
    const auto inRank = inType.getRank();
    if (axis < 0) {
        axis += inRank;
    }
    outShape[axis] = kContent.getSplatValue<int64_t>();

    const auto outType = inType.changeShape(Shape(outShape));

    inferredReturnTypes.push_back(outType);

    const auto outType1 = outType.changeElemType(topK.element_type());
    inferredReturnTypes.push_back(outType1);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::TopKOp::serialize(EMU::BlobWriter& /*writer*/) {
    VPUX_THROW("TopK op without regions is not implemented in low level dialects.");
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::TopKOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    SmallVector<TileInfo> inputTiles;
    auto curTile = outputTile;
    const auto inShape = getShape(input());
    const auto kAxis = Dim(axis());
    curTile.shape[kAxis] = inShape[kAxis];
    inputTiles.push_back(curTile);
    const auto kShape = getShape(k());
    auto kTile = TileInfo(kShape);
    inputTiles.push_back(kTile);

    return TilingInfo{inputTiles};
}

void vpux::VPU::TopKOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // Do nothing
}

OutputTiling vpux::VPU::TopKOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto baseOp = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for TopK currently, for op {0} at '{1}'", baseOp->getName(),
                    getLoc());
    auto axis = this->axis();
    auto tileDim = 0;
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(baseOp);
    const auto outputType = baseOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();
    Shape nTilesOnDim(outputShape.size(), 1);
    const auto isSupportedTileSize = [baseOp, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                             TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(baseOp, nTilesOnDim, outputShape);
        return tilingInfo.isSupportedTiling(tiles, tilingMode, log);
    };

    while (!isSupportedTileSize(nTilesOnDim, tilingMode)) {
        VPUX_THROW_WHEN(tileDim >= static_cast<int>(outputShape.size()), "Failed to tile {0} at '{1}'",
                        baseOp->getName(), baseOp->getLoc());

        if (tileDim == axis) {
            ++tileDim;
        } else {
            if (nTilesOnDim[Dim(tileDim)] >= outputShape[Dim(tileDim)]) {
                ++tileDim;
            } else {
                ++nTilesOnDim[Dim(tileDim)];
            }
        }
    }

    log.trace("Isolated tiling strategy: {0}", nTilesOnDim);
    auto origTiles = fillDividedTiles(baseOp, nTilesOnDim, outputShape);
    return origTiles;
}
