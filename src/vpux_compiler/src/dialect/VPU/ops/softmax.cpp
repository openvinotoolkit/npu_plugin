//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::SoftMaxOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                           mlir::Optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

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

OutputTiling vpux::VPU::SoftMaxOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto baseOp = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for SoftMax currently, for op {0} at '{1}'", baseOp->getName(),
                    getLoc());
    auto axis = this->axisIndAttr().getValue().getSExtValue();
    int64_t tileDim = 0;
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

    while (!isSupportedTileSize(nTilesOnDim, tilingMode)) {
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
