//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;
namespace {
llvm::SmallVector<int64_t> extractAxes(mlir::Location loc, mlir::Value axes) {
    auto axesVec = IE::constInputToData(loc, axes);
    VPUX_THROW_UNLESS(mlir::succeeded(axesVec), "Failed to extract axes");
    auto axesValue = axesVec.getValue();

    return axesValue;
}

bool checkAxes(int64_t tileDim, mlir::Location loc, mlir::Value axes) {
    auto axesValue = extractAxes(loc, axes);

    for (auto axesInd : axesValue) {
        if (tileDim == axesInd) {
            return true;
        }
    }
    return false;
};
}  // namespace

mlir::LogicalResult vpux::VPU::ReduceSumOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             mlir::Optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ReduceSumOpAdaptor reduceSum(operands, attrs);
    if (mlir::failed(reduceSum.verify(loc))) {
        return mlir::failure();
    }

    const auto input = reduceSum.input();
    const auto keepDims = reduceSum.keep_dims();

    auto axesValue = extractAxes(loc, reduceSum.axes());

    return VPU::inferReduceReturnTypes(loc, input, keepDims, axesValue, inferredReturnTypes);
}

//
// inferLayoutInfo
//

void vpux::VPU::ReduceSumOp::inferLayoutInfo(mlir::Operation* op, vpux::IE::LayerLayoutInfo& info) {
    vpux::IE::inferReduceLayoutInfo(op, info);
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::ReduceSumOp::serialize(EMU::BlobWriter& writer) {
    EMU::BlobWriter::String type;
    type = writer.createString("sum");

    MVCNN::ReduceParamsBuilder builder(writer);
    builder.add_keep_dims(checked_cast<bool>(keep_dims()));
    builder.add_operation(type);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ReduceParams});
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::ReduceSumOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    SmallVector<TileInfo> inputTiles;
    auto curTile = outputTile;
    auto axesValue = extractAxes(getLoc(), axes());
    const auto inShape = getShape(input());

    for (auto ind : axesValue) {
        curTile.shape[Dim(ind)] = inShape[Dim(ind)];
    }

    inputTiles.push_back(curTile);

    const auto axesShape = getShape(axes());
    auto axesTile = TileInfo(axesShape);

    inputTiles.push_back(axesTile);

    return TilingInfo{inputTiles};
}

void vpux::VPU::ReduceSumOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

OutputTiling vpux::VPU::ReduceSumOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto baseOp = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for ReduceSum currently, for op {0} at '{1}'", baseOp->getName(),
                    getLoc());

    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(baseOp);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface",
                    baseOp->getName());
    const auto outputType = baseOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();

    Shape nTilesOnDim(outputShape.size(), 1);

    const auto isSupportedTileSize = [baseOp, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                             TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(baseOp, nTilesOnDim, outputShape);
        return tilingInfo.isSupportedTiling(tiles, tilingMode, log);
    };

    int64_t tileDim = 0;

    while (!isSupportedTileSize(nTilesOnDim, tilingMode)) {
        if (checkAxes(tileDim, getLoc(), axes())) {
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
