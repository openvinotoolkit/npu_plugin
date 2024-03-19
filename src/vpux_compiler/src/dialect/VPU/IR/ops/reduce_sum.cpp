//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;
mlir::LogicalResult vpux::VPU::ReduceSumOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             std::optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ReduceSumOpAdaptor reduceSum(operands, attrs);
    if (mlir::failed(reduceSum.verify(loc))) {
        return mlir::failure();
    }

    const auto input = reduceSum.getInput();
    const auto keepDims = reduceSum.getKeepDims();

    auto axesValue = parseIntArrayAttr<int64_t>(reduceSum.getAxesValue());

    return VPU::inferReduceReturnTypes(loc, input, keepDims, axesValue, inferredReturnTypes);
}

//
// fold
//

mlir::OpFoldResult vpux::VPU::ReduceSumOp::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    return nullptr;
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::ReduceSumOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    SmallVector<TileInfo> inputTiles;

    auto inShape = getInput().getType().cast<vpux::NDTypeInterface>().getShape().raw();
    auto axesValue = parseIntArrayAttr<int64_t>(getAxesValue());

    auto tiledOutputAxis = outputTile.axis.raw();
    auto tiledOutputShape = outputTile.shape.raw();
    auto tiledOutputOffsets = outputTile.offsets.raw();

    // Adding tiling case when keep dims is false and the axes are reduced from outputShape
    if (getKeepDims() == false) {
        Shape newInput, newAxis, newOffset;
        std::copy(tiledOutputShape.begin(), tiledOutputShape.end(), std::back_inserter(newInput));
        std::copy(tiledOutputAxis.begin(), tiledOutputAxis.end(), std::back_inserter(newAxis));
        std::copy(tiledOutputOffsets.begin(), tiledOutputOffsets.end(), std::back_inserter(newOffset));

        for (auto axesInd : axesValue) {
            // Adjusting the new input based on tiled output
            newInput.insert(newInput.begin() + axesInd, inShape[axesInd]);
            newAxis.insert(newAxis.begin() + axesInd, 1);
            newOffset.insert(newOffset.begin() + axesInd, 0);
        }

        TileInfo inTile(newInput, newOffset, newAxis);

        return TilingInfo{{std::move(inTile)}};
    }

    auto inTile = outputTile;
    for (auto axesInd : axesValue) {
        inTile.shape[Dim(axesInd)] = inShape[axesInd];
    }

    return TilingInfo{{std::move(inTile)}};
}

void vpux::VPU::ReduceSumOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::ReduceSumOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto baseOp = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for ReduceSum currently, for op {0} at '{1}'", baseOp->getName(),
                    getLoc());

    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(baseOp);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface",
                    baseOp->getName());
    const auto outputType = baseOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();
    const auto inType = baseOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape();

    const auto axesValue = getAxesValue();
    Shape nTilesOnDim(outputShape.size(), 1);

    const auto checkAxes = [axesValue](int64_t tileDim) -> bool {
        auto axesArray = parseIntArrayAttr<int64_t>(axesValue);

        for (auto axesInd : axesArray) {
            if (tileDim == axesInd) {
                return true;
            }
        }
        return false;
    };

    const auto isSupportedTileSize = [baseOp, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                             TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(baseOp, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    int64_t tileDim = 0;

    while (!isSupportedTileSize(nTilesOnDim, tilingMode)) {
        VPUX_THROW_WHEN(
                getAxesValue().size() == inShape.size(),
                "Tiling for ReduceSum is not supported when all axes are reduced, got axes {0} and input size {1}",
                getAxesValue().size(), inShape.size());
        if ((getKeepDims() == true) && checkAxes(tileDim)) {
            ++tileDim;
        } else {
            if (nTilesOnDim[Dim(tileDim)] >= outputShape[Dim(tileDim)]) {
                ++tileDim;
            } else {
                ++nTilesOnDim[Dim(tileDim)];
            }
        }
    }

    auto origTiles = fillDividedTiles(baseOp, nTilesOnDim, outputShape);
    return origTiles;
}
