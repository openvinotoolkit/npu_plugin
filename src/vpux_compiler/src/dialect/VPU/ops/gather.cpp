//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

namespace {

mlir::FailureOr<int64_t> extractAxis(mlir::Location loc, VPU::GatherOpAdaptor gather) {
    if (gather.axis() != nullptr) {
        auto axisConst = gather.axis().getDefiningOp<Const::DeclareOp>();
        if (axisConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for axis");
        }

        const auto axisContent = axisConst.content();
        if (!axisContent.isSplat()) {
            return errorAt(loc, "Axis value must be a scalar");
        }

        int64_t axisInd = axisContent.getSplatValue<int64_t>();

        if (axisInd < 0) {
            const auto inType = gather.input().getType().cast<vpux::NDTypeInterface>();
            const auto inRank = inType.getRank();
            axisInd += inRank;
            VPUX_THROW_UNLESS(axisInd >= 0 && axisInd < inRank, "Wrong Gather axis {0}", axisInd);
        }

        return axisInd;
    } else if (gather.axis_value().hasValue()) {
        return gather.axis_value().getValue();
    } else {
        return errorAt(loc, "Axis was not provided");
    }
}

}  // namespace

mlir::LogicalResult vpux::VPU::GatherOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                          mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                          mlir::RegionRange /*regions*/,
                                                          mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::GatherOpAdaptor gather(operands, attrs);
    if (mlir::failed(gather.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = gather.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inType.getShape().raw();
    const auto indicesShape = gather.indices().getType().cast<vpux::NDTypeInterface>().getShape().raw();

    const auto axis = extractAxis(loc, gather);
    if (mlir::failed(axis)) {
        return mlir::failure();
    }

    SmallVector<int64_t> outShape;

    // calculate output shape
    int64_t batchDims = gather.batch_dims();
    int64_t axisVal = checked_cast<int64_t>(*axis);
    int64_t outRank = inputShape.size() + indicesShape.size() - 1 - batchDims;
    int64_t indicesRank = indicesShape.size();
    int64_t i = 0;

    for (; i < batchDims; i++) {
        outShape.push_back(inputShape[i] & indicesShape[i]);
    }
    for (; i < axisVal; i++) {
        outShape.push_back(inputShape[i]);
    }
    for (; i < axisVal + indicesRank - batchDims; i++) {
        outShape.push_back(indicesShape[batchDims - axisVal + i]);
    }
    for (; i < outRank; i++) {
        outShape.push_back(inputShape[batchDims + 1 - indicesRank + i]);
    }

    const auto outType = inType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

void vpux::VPU::GatherOp::inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
    IE::fillDefaultLayoutInfo(info);
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::GatherOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::GatherParamsBuilder builder(writer);
    builder.add_axis(checked_cast<uint32_t>(axis_value().getValue()));
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_GatherParams});
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::GatherOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    const auto origInputShape = getShape(input());
    const auto origIndicesShape = getShape(indices());
    bool hasAxisTensor = false;

    int64_t axisValue = 0;

    if (axis_valueAttr() != nullptr) {
        axisValue = axis_valueAttr().dyn_cast_or_null<mlir::IntegerAttr>().getValue().getSExtValue();
    }
    if (axis() != nullptr) {
        auto axisConst = axis().getDefiningOp<Const::DeclareOp>();
        VPUX_THROW_UNLESS(axisConst != nullptr, "Only constant input is supported for axis");
        const auto axisContent = axisConst.content();
        VPUX_THROW_UNLESS(axisContent.isSplat(), "Axis value must be a scalar");
        axisValue = axisContent.getSplatValue<int64_t>();
        hasAxisTensor = true;
    }
    int64_t batchDims = 0;
    if (batch_dimsAttr() != nullptr) {
        batchDims = batch_dimsAttr().dyn_cast_or_null<mlir::IntegerAttr>().getValue().getSExtValue();
    }

    TileInfo inputTile(origInputShape);
    TileInfo indicesTile(origIndicesShape);
    TileInfo axisTile(ShapeRef({1}));

    auto inputRank = origInputShape.size();
    auto indicesRank = origIndicesShape.size();

    for (int64_t i = 0; i < static_cast<signed>(inputRank); ++i) {
        if (i < axisValue) {
            inputTile.shape[Dim(i)] = outputTile.shape[Dim(i)];
            inputTile.offsets[Dim(i)] = outputTile.offsets[Dim(i)];
        } else if (i == axisValue) {
            continue;
        } else {
            inputTile.shape[Dim(i)] = outputTile.shape[Dim(i + indicesRank - batchDims - 1)];
            inputTile.offsets[Dim(i)] = outputTile.offsets[Dim(i + indicesRank - batchDims - 1)];
        }
    }
    if (hasAxisTensor) {
        return InputTiling{{inputTile, indicesTile, axisTile}};
    } else {
        return InputTiling{{inputTile, indicesTile}};
    }
}

void vpux::VPU::GatherOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

OutputTiling vpux::VPU::GatherOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto baseOp = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for Gather currently, for op {0} at '{1}'", baseOp->getName(),
                    getLoc());
    int64_t axisValue = 0;

    if (axis_valueAttr() != nullptr) {
        axisValue = axis_valueAttr().dyn_cast_or_null<mlir::IntegerAttr>().getValue().getSExtValue();
    }
    if (axis() != nullptr) {
        auto axisConst = axis().getDefiningOp<Const::DeclareOp>();
        VPUX_THROW_UNLESS(axisConst != nullptr, "Only constant input is supported for axis");
        const auto axisContent = axisConst.content();
        VPUX_THROW_UNLESS(axisContent.isSplat(), "Axis value must be a scalar");
        axisValue = axisContent.getSplatValue<int64_t>();
    }

    const auto indicesType = indices().getType().cast<vpux::NDTypeInterface>();
    const auto indicesShape = indicesType.getShape();

    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(baseOp);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface",
                    baseOp->getName());
    const auto outputType = baseOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();
    Shape nTilesOnDimforGather(outputShape.size(), 1);
    const auto isSupportedTileSize = [baseOp, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                             TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(baseOp, nTilesOnDim, outputShape);
        return tilingInfo.isSupportedTiling(tiles, tilingMode, log);
    };

    int64_t batchDims = 0;
    if (batch_dimsAttr() != nullptr) {
        batchDims = batch_dimsAttr().dyn_cast_or_null<mlir::IntegerAttr>().getValue().getSExtValue();
    }
    int64_t tileDim = batchDims;
    while (!isSupportedTileSize(nTilesOnDimforGather, tilingMode)) {
        if (tileDim == axisValue) {
            tileDim += (indicesShape.size() - batchDims);
        } else if (nTilesOnDimforGather[Dim(tileDim)] >= outputShape[Dim(tileDim)]) {
            ++tileDim;
        } else {
            ++nTilesOnDimforGather[Dim(tileDim)];
        }
    }
    log.trace("Isolated tiling strategy: {0}", nTilesOnDimforGather);
    return fillDividedTiles(baseOp, nTilesOnDimforGather, outputShape);
}
