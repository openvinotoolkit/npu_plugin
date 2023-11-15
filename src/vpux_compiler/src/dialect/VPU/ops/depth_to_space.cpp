//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

bool isDistributedOutputTypeOverlapped(VPU::ClusteredOpInterface op, int64_t numClusters) {
    auto resultType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto numClustersAttr = mlir::IntegerAttr::get(getInt64Type(op->getContext()), numClusters);
    auto opStrategy = op.getMultiClusterStrategy().value();
    auto outputDistributedType = getDistributedOutputTypeFromOp(op, resultType, numClustersAttr, opStrategy);
    if (auto distributedTensor = outputDistributedType.dyn_cast<VPU::DistributedTensorType>()) {
        return distributedTensor.getDistribution().getMode().getValue() == VPU::DistributionMode::OVERLAPPED;
    }
    return false;
}

bool isCompatibleWithMultiClusterNNDMA(VPU::DepthToSpaceOp op, vpux::ShapeRef nTilesOnDim) {
    if (op.mode() != IE::DepthToSpaceMode::BLOCKS_FIRST) {
        return false;
    }
    const auto inputType = op.input().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = op.output().getType().cast<vpux::NDTypeInterface>();
    const auto inOrder = inputType.getDimsOrder();
    const auto outOrder = outputType.getDimsOrder();
    if (inOrder != DimsOrder::NHWC || outOrder != DimsOrder::NHWC) {
        return false;
    }
    const auto inputShape = inputType.getShape();
    if (inputShape[Dims4D::Act::H] > VPUIP::DMA_MAX_NUMBER_PLANES) {
        // TODO: split more DMAs when the numPlanes is larger than 256 [Track number: E#57027]
        return false;
    }
    // Check previous op
    auto prevOp = op->getOperand(0).getDefiningOp<VPU::ClusteredOpInterface>();
    if (prevOp == nullptr) {
        return false;
    }
    auto prevOpStrategyAttr = prevOp.getMultiClusterStrategy();
    if (!prevOpStrategyAttr.has_value() || prevOpStrategyAttr.value() != VPU::MultiClusterStrategy::SplitOverHeight) {
        return false;
    }
    auto module = prevOp->getParentOfType<mlir::ModuleOp>();
    auto nceResOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::NCE);
    auto numClusters = nceResOp.count();

    if (isDistributedOutputTypeOverlapped(prevOp, numClusters)) {
        return false;
    }
    // Check next ops
    for (auto nextOp : op->getUsers()) {
        while (VPU::isPureViewOp(nextOp)) {
            if (!nextOp->hasOneUse()) {
                return false;
            }
            nextOp = *nextOp->getUsers().begin();
        }
        auto nceOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nextOp);
        if (nceOp == nullptr) {
            return false;
        }

        auto strategyAttr = nceOp.getMultiClusterStrategy();
        if (!strategyAttr.has_value()) {
            return false;
        }
        auto strategy = strategyAttr.value();
        if (strategy != VPU::MultiClusterStrategy::SplitOverHeight && strategy != VPU::MultiClusterStrategy::HKSwitch) {
            return false;
        }
        if (isDistributedOutputTypeOverlapped(nceOp, numClusters)) {
            return false;
        }
    }

    // Only support SOH and when numTiles is smaller than numClusters
    if (nTilesOnDim[Dims4D::Act::H] > numClusters) {
        return false;
    }
    // No tile on other axis
    if (nTilesOnDim[Dims4D::Act::C] != 1 || nTilesOnDim[Dims4D::Act::W] != 1) {
        return false;
    }
    return true;
}

mlir::LogicalResult vpux::VPU::DepthToSpaceOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DepthToSpaceOpAdaptor depthToSpace(operands, attrs);
    if (mlir::failed(depthToSpace.verify(loc))) {
        return mlir::failure();
    }

    const auto inShape = getShape(depthToSpace.input());
    const auto inType = depthToSpace.input().getType().cast<vpux::NDTypeInterface>();
    const auto block_size = depthToSpace.block_size();

    const auto elemType = inType.getElementType();
    if (!(elemType.isF16() || elemType.isF32() || elemType.isUnsignedInteger(8) ||
          elemType.isa<mlir::quant::QuantizedType>())) {
        return errorAt(loc, "DepthToSpace only support FP16, FP32, U8 data type");
    }

    if (inShape.size() < 3) {
        return errorAt(loc, "Invalid input tensor shape, dimension must be greater than 2.");
    }

    if (block_size <= 0) {
        return errorAt(loc, "Invalid block size {0}, should be greater than zero", block_size);
    }

    if (inShape[Dims4D::Act::C] % (block_size * block_size) != 0) {
        return errorAt(loc, "Invalid block size {0}, which is not divisible by input shape {1}", block_size,
                       inShape[Dims4D::Act::C]);
    }

    size_t W_out = inShape[Dims4D::Act::W] * block_size;
    size_t H_out = inShape[Dims4D::Act::H] * block_size;
    size_t C_out = inShape[Dims4D::Act::C] / (block_size * block_size);
    size_t N_out = inShape[Dims4D::Act::N];

    SmallVector<int64_t> outShape{checked_cast<int64_t>(N_out), checked_cast<int64_t>(C_out),
                                  checked_cast<int64_t>(H_out), checked_cast<int64_t>(W_out)};

    const auto outType = inType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::DepthToSpaceOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::DepthToSpaceParamsBuilder builder(writer);

    const auto blockSize = checked_cast<int32_t>(block_size());
    builder.add_blockSize(blockSize);

    builder.add_mode(vpux::VPUIP::convertVPUXDepthToSpaceMode2MVCNN(mode()));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_DepthToSpaceParams});
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::DepthToSpaceOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    const auto origInputShape = getShape(input());

    int64_t blockSize = 0;
    if (block_sizeAttr() != nullptr) {
        blockSize = block_sizeAttr().getValue().getSExtValue();
    }
    VPUX_THROW_WHEN(blockSize == 0, "BlockSize is zero and used as a divisor");

    TileInfo inputTile(origInputShape);
    inputTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    inputTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C] * (blockSize * blockSize);
    inputTile.shape[Dims4D::Act::W] = outputTile.shape[Dims4D::Act::W] / blockSize;
    inputTile.shape[Dims4D::Act::H] = outputTile.shape[Dims4D::Act::H] / blockSize;

    inputTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];
    inputTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C] * (blockSize * blockSize);
    inputTile.offsets[Dims4D::Act::W] = outputTile.offsets[Dims4D::Act::W] / blockSize;
    inputTile.offsets[Dims4D::Act::H] = outputTile.offsets[Dims4D::Act::H] / blockSize;

    return InputTiling{inputTile};
}

void vpux::VPU::DepthToSpaceOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::DepthToSpaceOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto op = this->getOperation();
    auto origOp = mlir::dyn_cast<VPU::DepthToSpaceOp>(op);
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);

    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();

    int64_t blockSize = 0;
    if (origOp.block_sizeAttr() != nullptr) {
        blockSize = origOp.block_sizeAttr().getValue().getSExtValue();
    }
    VPUX_THROW_WHEN(blockSize == 0, "BlockSize is zero and used as a divisor");

    Shape nTilesOnDimforDepthToSpace(outputShape.size(), 1);
    tilingMode = TilingMode::ISOLATED;
    const auto tilingModeToCheck = tilingMode;

    auto getTileDimOrder = [&]() {
        VPUX_THROW_UNLESS(outputType.getDimsOrder() == DimsOrder::NCHW || outputType.getDimsOrder() == DimsOrder::NHWC,
                          "DepthToSpace Op only support NCHW and NHWC layout, but got '{0}'",
                          outputType.getDimsOrder());

        // It is better to tile DepthToSpace Op at the highest dimension
        // to avoid stride concat that is inefficient
        if (origOp.mode() == IE::DepthToSpaceMode::DEPTH_FIRST) {
            return outputType.getDimsOrder() == DimsOrder::NHWC
                           ? SmallVector<Dim>{Dims4D::Act::H, Dims4D::Act::W, Dims4D::Act::C}
                           : SmallVector<Dim>{Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W};
        }

        // It is illegal to tile DepthToSpace Op at channel when it is the BLOCKS_FIRST mode
        // If that, the output will be a discontinuous memory buffer and will cause accuracy issue
        if (origOp.mode() == IE::DepthToSpaceMode::BLOCKS_FIRST) {
            return SmallVector<Dim>{Dims4D::Act::H, Dims4D::Act::W};
        }

        VPUX_THROW("Unknown DepthToSpaceMode. BLOCKS_FIRST and DEPTH_FIRST methods are supported only");
    };

    auto tileDimOrder = getTileDimOrder();

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

    int64_t maxTile = 1;

    while (tileDimIter < tileDimOrder.end()) {
        if (dimToTile == Dims4D::Act::H || dimToTile == Dims4D::Act::W) {
            while (((maxTile * blockSize) <= outputShape[dimToTile]) &&
                   (!isSupportedTileSize(nTilesOnDimforDepthToSpace, tilingModeToCheck))) {
                if (outputShape[dimToTile] % (maxTile * blockSize) == 0) {
                    nTilesOnDimforDepthToSpace[dimToTile] = maxTile;
                    maxTile++;
                } else {
                    maxTile++;
                }
            }
            dimToTile = *(++tileDimIter);
            maxTile = 1;
        } else if (dimToTile == Dims4D::Act::C) {
            while (!isSupportedTileSize(nTilesOnDimforDepthToSpace, tilingModeToCheck)) {
                if (nTilesOnDimforDepthToSpace[dimToTile] >= outputShape[dimToTile]) {
                    break;
                } else {
                    ++nTilesOnDimforDepthToSpace[dimToTile];
                }
            }
            dimToTile = *(++tileDimIter);
        }
    }

    // Explicit tiling not needed, op will be converted to multicluster DMA
    if (isCompatibleWithMultiClusterNNDMA(origOp, nTilesOnDimforDepthToSpace)) {
        nTilesOnDimforDepthToSpace = vpux::Shape(outputShape.size(), 1);
    }

    auto origTiles = fillDividedTiles(op, nTilesOnDimforDepthToSpace, outputShape);
    return origTiles;
}
