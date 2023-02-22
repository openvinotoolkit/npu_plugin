//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::SpaceToDepthOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::SpaceToDepthOpAdaptor spd(operands, attrs);
    if (mlir::failed(spd.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = spd.input().getType().cast<vpux::NDTypeInterface>();

    const auto elementType = inputType.getElementType();
    if (!(elementType.isF16() || elementType.isF32() || elementType.isUnsignedInteger(8) ||
          elementType.isa<mlir::quant::QuantizedType>())) {
        return errorAt(loc, "SpaceToDepth only supports FP16, FP32, U8 or Quantized input data type");
    }

    const auto inputShape = inputType.getShape().raw();
    const auto block_size = spd.block_size();

    if (inputShape.size() < 3) {
        return errorAt(loc, "Input tensor rank must be greater than 2. Got {0}D tensor", inputShape.size());
    }

    if (block_size <= 0) {
        return errorAt(loc, "Invalid block size {0}, should be greater than zero", block_size);
    }

    static const auto N = Dims4D::Act::N;
    static const auto C = Dims4D::Act::C;
    static const auto H = Dims4D::Act::H;
    static const auto W = Dims4D::Act::W;

    if (inputShape[H.ind()] % block_size || inputShape[W.ind()] % block_size) {
        return errorAt(loc, "Invalid block_size {0} , height {1} and width {2} must be divisible by block_size",
                       block_size, inputShape[H.ind()], inputShape[W.ind()]);
    }

    const auto outN = inputShape[N.ind()];
    const auto outC = inputShape[C.ind()] * block_size * block_size;
    const auto outH = inputShape[H.ind()] / block_size;
    const auto outW = inputShape[W.ind()] / block_size;

    SmallVector<int64_t> outShape{outN, outC, outH, outW};

    const auto outType = inputType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::SpaceToDepthOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::SpaceToDepthParamsBuilder builder(writer);

    const auto blockSize = checked_cast<int32_t>(block_size());
    builder.add_blockSize(blockSize);

    const auto spdMode = VPUIP::convertVPUXSpaceToDepthMode2MVCNN(mode());
    builder.add_mode(spdMode);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_SpaceToDepthParams});
}
//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::SpaceToDepthOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    const auto origInputShape = getShape(input());

    int64_t blockSize = 0;

    VPUX_THROW_UNLESS(block_sizeAttr() != nullptr, "Got NULL block_size");
    blockSize = block_sizeAttr().dyn_cast<mlir::IntegerAttr>().getValue().getSExtValue();

    TileInfo inputTile(origInputShape);

    inputTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    inputTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C] / (blockSize * blockSize);
    inputTile.shape[Dims4D::Act::W] = outputTile.shape[Dims4D::Act::W] * blockSize;
    inputTile.shape[Dims4D::Act::H] = outputTile.shape[Dims4D::Act::H] * blockSize;

    inputTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];
    inputTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C] / (blockSize * blockSize);
    inputTile.offsets[Dims4D::Act::W] = outputTile.offsets[Dims4D::Act::W] * blockSize;
    inputTile.offsets[Dims4D::Act::H] = outputTile.offsets[Dims4D::Act::H] * blockSize;

    return InputTiling{inputTile};
}

void vpux::VPU::SpaceToDepthOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

OutputTiling vpux::VPU::SpaceToDepthOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto op = this->getOperation();
    auto origOp = mlir::dyn_cast<VPU::SpaceToDepthOp>(op);
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);

    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();

    int64_t blockSize = 0;

    VPUX_THROW_UNLESS(block_sizeAttr() != nullptr, "Got NULL block_size");
    blockSize = block_sizeAttr().dyn_cast<mlir::IntegerAttr>().getValue().getSExtValue();

    Shape nTilesOnDimforSpaceToDepth(outputShape.size(), 1);
    tilingMode = TilingMode::ISOLATED;
    const auto tilingModeToCheck = tilingMode;

    SmallVector<Dim> tileDimOrder;

    if (origOp.mode() == IE::SpaceToDepthMode::BLOCKS_FIRST) {
        tileDimOrder = {Dims4D::Act::W, Dims4D::Act::H};
    } else if (origOp.mode() == IE::SpaceToDepthMode::DEPTH_FIRST) {
        tileDimOrder = {Dims4D::Act::W, Dims4D::Act::H, Dims4D::Act::C};
    } else {
        VPUX_THROW("Unknown SpaceToDepthMode: {0}. BLOCKS_FIRST and DEPTH_FIRST methods are supported only",
                   origOp.mode());
    }

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;

    const auto isSupportedTileSize = [op, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                         TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
        return tilingInfo.isSupportedTiling(tiles, tilingMode, log);
    };

    int64_t maxTile = 1;

    while (tileDimIter < tileDimOrder.end()) {
        if (dimToTile == Dims4D::Act::C) {
            while (((maxTile * blockSize * blockSize) <= outputShape[dimToTile]) &&
                   (!isSupportedTileSize(nTilesOnDimforSpaceToDepth, tilingModeToCheck))) {
                if (outputShape[dimToTile] % (maxTile * blockSize * blockSize) == 0) {
                    nTilesOnDimforSpaceToDepth[dimToTile] = maxTile;
                    maxTile++;
                } else {
                    maxTile++;
                }
            }
            dimToTile = *(++tileDimIter);
            maxTile = 1;
        } else if (dimToTile == Dims4D::Act::H || dimToTile == Dims4D::Act::W) {
            while (!isSupportedTileSize(nTilesOnDimforSpaceToDepth, tilingModeToCheck)) {
                if (nTilesOnDimforSpaceToDepth[dimToTile] >= outputShape[dimToTile]) {
                    break;
                } else {
                    ++nTilesOnDimforSpaceToDepth[dimToTile];
                }
            }
            dimToTile = *(++tileDimIter);
        } else {
            VPUX_THROW("Unsupported dim to tile: {0}", dimToTile);
        }
    }

    auto origTiles = fillDividedTiles(op, nTilesOnDimforSpaceToDepth, outputShape);
    return origTiles;
}
