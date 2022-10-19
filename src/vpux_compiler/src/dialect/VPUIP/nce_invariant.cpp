//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/Operation.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// verifyConvChannels
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyConvChannels(mlir::Location loc, vpux::NDTypeInterface inputType,
                                                                  vpux::NDTypeInterface filterType, Logger log) {
    log.setName("NCEInvariant");

    if (filterType.getRank() != 4) {
        log.trace("[{0}] Filter has unsupported rank: {1}", loc, filterType.getRank());
        return mlir::failure();
    }

    const auto filterShape = filterType.getShape();

    const auto OC = filterShape[Dims4D::Filter::OC];
    if (OC % VPU::NCEInvariant::getAlignment(filterType.getElementType()) != 0) {
        log.trace("[{0}] Convolution output channels are not aligned", loc);
        return mlir::failure();
    }

    if (inputType.getDimsOrder() == DimsOrder::NHWC) {
        const auto IC = filterShape[Dims4D::Filter::IC];
        if (IC % VPU::NCEInvariant::getAlignment(filterType.getElementType()) != 0) {
            log.trace("[{0}] ZMajor Convolution input channels are not aligned", loc);
            return mlir::failure();
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::ConvolutionOp origOp, Logger log) {
    return verifyConvChannels(origOp->getLoc(), origOp.input().getType().cast<vpux::NDTypeInterface>(),
                              origOp.filter().getType().cast<vpux::NDTypeInterface>(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(VPU::NCEConvolutionOp, Logger) {
    // VPU.NCE operations guarantees that invariants
    return mlir::success();
}

//
// verifyPoolChannels
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPoolChannels(mlir::Location loc, vpux::NDTypeInterface inputType,
                                                                  Logger log) {
    log.setName("NCEInvariant");

    if (inputType.getRank() != 4) {
        log.trace("[{0}] Input has unsupported rank: {1}", loc, inputType.getRank());
        return mlir::failure();
    }

    const auto inputShape = inputType.getShape();
    const auto IC = inputShape[Dims4D::Act::C];

    if (IC % VPU::NCEInvariant::getAlignment(inputType.getElementType()) != 0) {
        log.trace("[{0}] Pooling channels are not aligned", loc);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::MaxPoolOp origOp, Logger log) {
    return verifyPoolChannels(origOp->getLoc(), origOp.input().getType().cast<vpux::NDTypeInterface>(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(VPU::NCEMaxPoolOp, Logger) {
    // VPU.NCE operations guarantees that invariants
    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::AvgPoolOp origOp, Logger log) {
    return verifyPoolChannels(origOp->getLoc(), origOp.input().getType().cast<vpux::NDTypeInterface>(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(VPU::NCEAveragePoolOp, Logger) {
    // VPU.NCE operations guarantees that invariants
    return mlir::success();
}

//
// verifyEltwiseChannels
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyEltwiseChannels(mlir::Location loc,
                                                                     vpux::NDTypeInterface firstInputType,
                                                                     vpux::NDTypeInterface secondInputType,
                                                                     Logger log) {
    log.setName("NCEInvariant");
    if (firstInputType.getRank() != 4) {
        log.trace("[{0}] Eltwise input1 shape does not have 4 dimensions. Not supported.", loc);
        return mlir::failure();
    }

    if (secondInputType.getRank() != 4) {
        log.trace("[{0}] Eltwise input2 shape does not have 4 dimensions. Not supported.", loc);
        return mlir::failure();
    }

    const auto firstInputShape = firstInputType.getShape();
    const auto secondInputShape = secondInputType.getShape();
    const auto firstIC = firstInputShape[Dims4D::Act::C];
    const auto secondIC = secondInputShape[Dims4D::Act::C];

    if (firstIC % VPU::NCEInvariant::getAlignment(firstInputType.getElementType()) != 0) {
        log.trace("[{0}] Eltwise input1 channels are not aligned", loc);
        return mlir::failure();
    }

    if (secondIC % VPU::NCEInvariant::getAlignment(secondInputType.getElementType()) != 0) {
        log.trace("[{0}] Eltwise input2 channels are not aligned", loc);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::AddOp origOp, Logger log) {
    auto input1Type = origOp.input1().getType().cast<vpux::NDTypeInterface>();
    auto input2Type = origOp.input2().getType().cast<vpux::NDTypeInterface>();
    return verifyEltwiseChannels(origOp->getLoc(), input1Type, input2Type, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::MultiplyOp origOp, Logger log) {
    auto input1Type = origOp.input1().getType().cast<vpux::NDTypeInterface>();
    auto input2Type = origOp.input2().getType().cast<vpux::NDTypeInterface>();
    return verifyEltwiseChannels(origOp->getLoc(), input1Type, input2Type, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::SubtractOp origOp, Logger log) {
    auto input1Type = origOp.input1().getType().cast<vpux::NDTypeInterface>();
    auto input2Type = origOp.input2().getType().cast<vpux::NDTypeInterface>();
    return verifyEltwiseChannels(origOp->getLoc(), input1Type, input2Type, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::AndOp origOp, Logger log) {
    auto input1Type = origOp.input1().getType().cast<vpux::NDTypeInterface>();
    auto input2Type = origOp.input2().getType().cast<vpux::NDTypeInterface>();
    return verifyEltwiseChannels(origOp->getLoc(), input1Type, input2Type, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(VPU::NCEEltwiseOp, Logger) {
    // VPU.NCE operations guarantees that invariants
    return mlir::success();
}

//
// verifyGroupConvChannels
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyGroupConvChannels(mlir::Location loc,
                                                                       vpux::NDTypeInterface inputType,
                                                                       vpux::NDTypeInterface filterType, Logger log) {
    log.setName("NCEInvariant");

    if (inputType.getRank() != 4) {
        log.trace("[{0}] Input has unsupported rank: {1}", loc, inputType.getRank());
        return mlir::failure();
    }

    if (filterType.getRank() != 4) {
        log.trace("[{0}] Filter has unsupported rank: {1}", loc, filterType.getRank());
        return mlir::failure();
    }

    const auto filterShape = filterType.getShape();
    const auto filtersPerInChan = filterShape[Dims4D::Filter::IC];
    if (filtersPerInChan != 1) {
        log.trace("[{0}] Group Convolution with more than one filter per channel is not supported", loc);
        return mlir::failure();
    }

    const auto inputShape = inputType.getShape();
    const auto inputChan = inputShape[Dims4D::Act::C];
    const auto OC = filterShape[Dims4D::Filter::OC];
    if (OC != inputChan) {
        log.trace("[{0}] Group Convolution has {1} groups, expected {2}", loc, OC, inputChan);
        return mlir::failure();
    }

    if (OC % VPU::NCEInvariant::getAlignment(filterType.getElementType()) != 0) {
        log.trace("[{0}] Group Convolution output channels are not aligned", loc);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::GroupConvolutionOp origOp, Logger log) {
    return verifyGroupConvChannels(origOp->getLoc(), origOp.input().getType().cast<vpux::NDTypeInterface>(),
                                   origOp.filter().getType().cast<vpux::NDTypeInterface>(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(VPU::NCEDepthConvolutionOp, Logger) {
    // VPU.NCE operations guarantees that invariants
    return mlir::success();
}

//
// verifyConvCMX
//

namespace {

Byte getRequiredCMXSize(ArrayRef<vpux::NDTypeInterface> operands) {
    Byte requiredCMX(0);

    for (const auto& operand : operands) {
        requiredCMX += operand.getTotalAllocSize();
    }

    return requiredCMX;
}

Byte getRequiredCMXSizeForNCEOps(ArrayRef<vpux::NDTypeInterface> operands, int64_t numChannels) {
    auto requiredCMX = getRequiredCMXSize(operands);

    requiredCMX += numChannels * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC * 4_Byte;

    return requiredCMX;
}

}  // namespace

Byte getCMXSizeForTiling(mlir::ModuleOp module) {
    return vpux::VPU::getTotalCMXSize(module);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyConvCMX(mlir::Location loc, mlir::ModuleOp module,
                                                             vpux::NDTypeInterface inputType,
                                                             vpux::NDTypeInterface filterType,
                                                             vpux::NDTypeInterface outputType,
                                                             mlir::ArrayAttr kernelStrides, Logger log) {
    log.setName("NCEInvariant");

    const auto filterShape = filterType.getShape();
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto IC = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto alignment = VPU::NCEInvariant::getAlignment(outputType.getElementType());
    if (OC % alignment != 0) {
        log.debug("[{0}] Output channels count of depthwise convolution must be a multiple of {1}, got {2}", loc,
                  alignment, OC);
        return mlir::failure();
    }

    const auto inOrder = inputType.getDimsOrder();

    Byte requiredCMX;
    if (inOrder == DimsOrder::NHWC) {
        requiredCMX = getRequiredCMXSizeForNCEOps({inputType, filterType, outputType}, OC);
    } else if (inOrder == DimsOrder::NCHW) {
        const auto remainder = (IC * KY * KX) % alignment;
        VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);

        const auto padding = (remainder > 0) ? (alignment - remainder) : 0;

        const auto alignedWeightShape = SmallVector<int64_t>{OC, 1, 1, IC * KY * KX + padding};
        const auto alignedFilterType = mlir::RankedTensorType::get(alignedWeightShape, filterType.getElementType());

        const auto kernelSize = Shape{KY, KX};
        const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(kernelStrides));

        const auto activationWindowSize = VPU::NCESparsity::getActivationWindowSize(
                VPU::NCESparsity::Mode::CM_CONV, kernelSize, kernelStridesVals[Dims4D::Strides::X],
                inputType.getElementType(), IC);

        requiredCMX = getRequiredCMXSizeForNCEOps({inputType, alignedFilterType, outputType}, OC) +
                      activationWindowSize * 1_Byte;
    } else {
        log.debug("[{0}] Unsupported input layout '{1}'", loc, inOrder);
        return mlir::failure();
    }

    const auto cmxSize = getCMXSizeForTiling(module);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for Convolution, available '{1}', required '{2}'", loc, cmxSize,
                  requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

//
// verifyPoolCMX
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPoolCMX(mlir::Location loc, mlir::ModuleOp module,
                                                             vpux::NDTypeInterface inputType,
                                                             vpux::NDTypeInterface outputType,
                                                             mlir::ArrayAttr kernelSize, mlir::ArrayAttr kernelStrides,
                                                             Logger log) {
    log.setName("NCEInvariant");

    VPUX_THROW_UNLESS(kernelSize.size() == 2, "Unsupported kernel size: {0}", kernelSize.size());
    VPUX_THROW_UNLESS(kernelStrides.size() == 2, "Unsupported strides size: {0}", kernelSize.size());

    const auto outputShape = outputType.getShape();
    const auto OC = outputShape[Dims4D::Act::C];

    const auto kernelSizeVals = Shape(parseIntArrayAttr<int64_t>(kernelSize));
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(kernelStrides));

    const auto activationWindowSize = VPU::NCESparsity::getActivationWindowSize(
            VPU::NCESparsity::Mode::POOL, kernelSizeVals, kernelStridesVals[Dims4D::Strides::X],
            inputType.getElementType(), 1);

    const auto requiredCMX = getRequiredCMXSizeForNCEOps({inputType, outputType}, OC) + activationWindowSize * 1_Byte;

    const auto cmxSize = getCMXSizeForTiling(module);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for Pooling, available '{1}', required '{2}'", loc, cmxSize,
                  requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

//
// verifyEltwiseCMX
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyEltwiseCMX(mlir::Location loc, mlir::ModuleOp module,
                                                                vpux::NDTypeInterface firstInputType,
                                                                vpux::NDTypeInterface secondInputType,
                                                                vpux::NDTypeInterface outputType, Logger log) {
    log.setName("NCEInvariant");

    const auto requiredCMX = getRequiredCMXSizeForNCEOps({firstInputType, secondInputType, outputType}, 0);

    const auto cmxSize = getCMXSizeForTiling(module);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for Eltwise, available '{1}', required '{2}'", loc, cmxSize,
                  requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

//
// verifyGroupConvCMX
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyGroupConvCMX(mlir::Location loc, mlir::ModuleOp module,
                                                                  vpux::NDTypeInterface inputType,
                                                                  vpux::NDTypeInterface filterType,
                                                                  vpux::NDTypeInterface outputType,
                                                                  mlir::ArrayAttr kernelStrides, Logger log) {
    log.setName("NCEInvariant");

    VPUX_THROW_UNLESS(kernelStrides.size() == 2, "Unsupported strides size: {0}", kernelStrides.size());

    const auto filterShape = filterType.getShape();
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto filtersPerInChan = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto alignment = VPU::NCEInvariant::getAlignment(outputType.getElementType());
    if (OC % alignment != 0) {
        log.debug("[{0}] Output channels count of depthwise convolution must be a multiple of {1}, got {2}", loc,
                  alignment, OC);
        return mlir::failure();
    }

    const auto remainder = (filtersPerInChan * KY * KX) % alignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);

    const auto padding = (remainder > 0) ? (alignment - remainder) : 0;
    const auto alignedWeightShape = SmallVector<int64_t>{OC, 1, 1, filtersPerInChan * KY * KX + padding};
    const auto alignedFilterType = mlir::RankedTensorType::get(alignedWeightShape, filterType.getElementType());

    const Shape kernelSizeVals{KY, KX};
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(kernelStrides));

    const auto activationWindowSize = VPU::NCESparsity::getActivationWindowSize(
            VPU::NCESparsity::Mode::DW_CONV, kernelSizeVals, kernelStridesVals[Dims4D::Strides::X],
            inputType.getElementType(), 1);

    const auto requiredCMX =
            getRequiredCMXSizeForNCEOps({inputType, alignedFilterType, outputType}, OC) + activationWindowSize * 1_Byte;

    const auto cmxSize = getCMXSizeForTiling(module);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for Depthwise Convolution, available '{1}', required '{2}'", loc,
                  cmxSize, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

// verifyPipeliningCMX

bool isNestedTiling(vpux::OutputTiling tiling) {
    return tiling[0].axis[Dims4D::Act::C] > 1 && tiling[0].axis[Dims4D::Act::H] > 1;
}

vpux::NDTypeInterface getAlignedFilterType(const SmallVector<vpux::NDTypeInterface>& tileTypes) {
    const auto outputTileType = tileTypes[2];
    const auto filterTileType = tileTypes[1];
    const auto filterTileShape = filterTileType.getShape();
    const auto OC = filterTileShape[Dims4D::Filter::OC];
    const auto IC = filterTileShape[Dims4D::Filter::IC];
    const auto KY = filterTileShape[Dims4D::Filter::KY];
    const auto KX = filterTileShape[Dims4D::Filter::KX];

    const auto alignment = VPU::NCEInvariant::getAlignment(outputTileType.getElementType());
    const auto remainder = (IC * KY * KX) % alignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);

    const auto padding = (remainder > 0) ? (alignment - remainder) : 0;

    const auto alignedWeightShape = SmallVector<int64_t>{OC, 1, 1, IC * KY * KX + padding};
    const auto alignedFilterType = mlir::RankedTensorType::get(alignedWeightShape, filterTileType.getElementType());
    return alignedFilterType;
}

// Convolution

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::ConvolutionOp origOp, const TileInfo& outTile) {
    const auto origBiasShape = origOp.bias() != nullptr ? getShape(origOp.bias()) : ShapeRef();
    const auto origPadding = PadInfo(origOp.pads_begin(), origOp.pads_end());

    auto tileConf = vpux::backInferConvTile(outTile, getShape(origOp.input()), getShape(origOp.filter()), origBiasShape,
                                            origOp.strides(), origPadding);

    SmallVector<vpux::NDTypeInterface> tileTypes;

    tileTypes.push_back(origOp.input().getType().cast<vpux::NDTypeInterface>().extractDenseTile(
            tileConf.tiles[0].offsets, tileConf.tiles[0].shape));
    tileTypes.push_back(origOp.filter().getType().cast<vpux::NDTypeInterface>().extractDenseTile(
            tileConf.tiles[1].offsets, tileConf.tiles[1].shape));
    tileTypes.push_back(
            origOp.getType().cast<vpux::NDTypeInterface>().extractDenseTile(outTile.offsets, outTile.shape));

    return tileTypes;
}

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::NCEConvolutionOp origOp, const TileInfo& outTile) {
    const auto inputTiles = origOp.backInferTileInfo(outTile, Logger::global()).tiles;
    auto inputTileType = origOp.input().getType().cast<vpux::NDTypeInterface>().extractDenseTile(inputTiles[0].offsets,
                                                                                                 inputTiles[0].shape);
    auto filterTileType = origOp.filter().getType().cast<vpux::NDTypeInterface>().extractDenseTile(
            inputTiles[1].offsets, inputTiles[1].shape);
    auto outputTileType =
            origOp.getType().cast<vpux::NDTypeInterface>().extractDenseTile(outTile.offsets, outTile.shape);

    if (origOp->hasAttr(VPU::multiClusterStrategy)) {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(nceOp == nullptr, "Op {0} has multiClusterStrategy but is not an NCEOp", origOp->getLoc());
        auto numClusters = VPU::getOptimalNumClusters(
                nceOp, outputTileType.getShape()[Dims4D::Act::C],
                nceOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
        return {VPU::getDistributedActivationTypeFromOp(nceOp, inputTileType, numClusters),
                VPU::getDistributedFilterTypeFromOp(nceOp, filterTileType, numClusters),
                VPU::getDistributedOutputTypeFromOp(nceOp, outputTileType, numClusters)};
    }

    return {inputTileType, filterTileType, outputTileType};
}

SmallVector<vpux::NDTypeInterface> getRequiredOperandsForPipelining(VPU::ConvolutionOp origOp,
                                                                    vpux::OutputTiling tiling) {
    // The tiling strategy follows last-tile-not-biggest
    // So just check the first two tiles are enough to make sure prefetchable
    auto curTile = tiling[0];
    auto nextTile = tiling[1];
    bool isWeightPrefetch = curTile.axis[Dims4D::Act::C] > 1;

    const auto& curTileTypes = getTileTypes(origOp, curTile);
    const auto& nextTileTypes = getTileTypes(origOp, nextTile);

    SmallVector<vpux::NDTypeInterface> requiredOperands{curTileTypes[0], getAlignedFilterType(curTileTypes),
                                                        curTileTypes[2]};
    if (isWeightPrefetch) {
        requiredOperands.push_back(getAlignedFilterType(nextTileTypes));
    } else {
        requiredOperands.push_back(nextTileTypes[0]);
    }
    return requiredOperands;
}

SmallVector<vpux::NDTypeInterface> getRequiredOperandsForPipelining(VPU::NCEConvolutionOp origOp,
                                                                    vpux::OutputTiling tiling) {
    // The tiling strategy follows last-tile-not-biggest
    // So just check the first two tiles are enough to make sure prefetchable
    auto curTile = tiling[0];
    auto nextTile = tiling[1];
    bool isWeightPrefetch = curTile.axis[Dims4D::Act::C] > 1;

    const auto& curTileTypes = getTileTypes(origOp, curTile);
    const auto& nextTileTypes = getTileTypes(origOp, nextTile);

    return {curTileTypes[0], curTileTypes[1], curTileTypes[2], isWeightPrefetch ? nextTileTypes[1] : nextTileTypes[0]};
}

int64_t getRequiredChannelSizeForPipelining(VPU::ConvolutionOp origOp, vpux::OutputTiling tiling) {
    auto curFilterShape = getTileTypes(origOp, tiling[0])[1].getShape();
    auto nextFilterShape = getTileTypes(origOp, tiling[1])[1].getShape();
    return curFilterShape[Dims4D::Filter::OC] + nextFilterShape[Dims4D::Filter::OC];
}

int64_t getRequiredChannelSizeForPipelining(VPU::NCEConvolutionOp origOp, vpux::OutputTiling tiling) {
    auto curFilterShape = getTileTypes(origOp, tiling[0])[1].getShape();
    auto nextFilterShape = getTileTypes(origOp, tiling[1])[1].getShape();
    return curFilterShape[Dims4D::Filter::OC] + nextFilterShape[Dims4D::Filter::OC];
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::ConvolutionOp origOp, vpux::OutputTiling tiling,
                                                                   Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() <= 1) {
        return mlir::failure();
    }
    if (isNestedTiling(tiling)) {
        return mlir::failure();
    }

    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);

    Byte requiredCMX = Byte(0);

    requiredCMX = getRequiredCMXSize(getRequiredOperandsForPipelining(origOp, tiling));
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}'",
                  origOp->getLoc(), cmxSize, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::NCEConvolutionOp origOp,
                                                                   vpux::OutputTiling tiling, Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() <= 1) {
        return mlir::failure();
    }
    if (isNestedTiling(tiling)) {
        return mlir::failure();
    }

    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);

    Byte requiredCMX = Byte(0);

    requiredCMX = getRequiredCMXSizeForNCEOps(getRequiredOperandsForPipelining(origOp, tiling),
                                              getRequiredChannelSizeForPipelining(origOp, tiling));
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}'",
                  origOp->getLoc(), cmxSize, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

// MaxPool

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::MaxPoolOp origOp, const TileInfo& outTile) {
    const auto origPadding = PadInfo(origOp.pads_begin(), origOp.pads_end());

    auto tileConf = vpux::backInferPoolTile(outTile, getShape(origOp.input()), origOp.kernel_size(), origOp.strides(),
                                            origPadding);

    SmallVector<vpux::NDTypeInterface> tileTypes;

    tileTypes.push_back(origOp.input().getType().cast<vpux::NDTypeInterface>().extractDenseTile(
            tileConf.tiles[0].offsets, tileConf.tiles[0].shape));
    tileTypes.push_back(
            origOp.getType().cast<vpux::NDTypeInterface>().extractDenseTile(outTile.offsets, outTile.shape));

    return tileTypes;
}

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::NCEMaxPoolOp origOp, const TileInfo& outTile) {
    const auto inputTiles = origOp.backInferTileInfo(outTile, Logger::global()).tiles;
    auto inputTileType = origOp.input().getType().cast<vpux::NDTypeInterface>().extractDenseTile(inputTiles[0].offsets,
                                                                                                 inputTiles[0].shape);
    auto outputTileType =
            origOp.getType().cast<vpux::NDTypeInterface>().extractDenseTile(outTile.offsets, outTile.shape);

    if (origOp->hasAttr(VPU::multiClusterStrategy)) {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(nceOp == nullptr, "Op {0} has multiClusterStrategy but is not an NCEOp", origOp->getLoc());
        auto numClusters = VPU::getOptimalNumClusters(
                nceOp, outputTileType.getShape()[Dims4D::Act::C],
                nceOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
        return {VPU::getDistributedActivationTypeFromOp(nceOp, inputTileType, numClusters),
                VPU::getDistributedOutputTypeFromOp(nceOp, outputTileType, numClusters)};
    }

    return {inputTileType, outputTileType};
}

// AveragePool

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::NCEAveragePoolOp origOp, const TileInfo& outTile) {
    const auto inputTiles = origOp.backInferTileInfo(outTile, Logger::global()).tiles;
    auto inputTileType = origOp.input().getType().cast<vpux::NDTypeInterface>().extractDenseTile(inputTiles[0].offsets,
                                                                                                 inputTiles[0].shape);
    auto outputTileType =
            origOp.getType().cast<vpux::NDTypeInterface>().extractDenseTile(outTile.offsets, outTile.shape);

    if (origOp->hasAttr(VPU::multiClusterStrategy)) {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(nceOp == nullptr, "Op {0} has multiClusterStrategy but is not an NCEOp", origOp->getLoc());
        auto numClusters = VPU::getOptimalNumClusters(
                nceOp, outputTileType.getShape()[Dims4D::Act::C],
                nceOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
        return {VPU::getDistributedActivationTypeFromOp(nceOp, inputTileType, numClusters),
                VPU::getDistributedOutputTypeFromOp(nceOp, outputTileType, numClusters)};
    }

    return {inputTileType, outputTileType};
}

SmallVector<vpux::NDTypeInterface> getRequiredOperandsForPipelining(VPU::MaxPoolOp origOp, vpux::OutputTiling tiling) {
    // The tiling strategy follows last-tile-not-biggest
    // So just check the first two tiles are enough to make sure prefetchable
    auto curTile = tiling[0];
    auto nextTile = tiling[1];

    const auto& curTileTypes = getTileTypes(origOp, curTile);
    const auto& nextTileTypes = getTileTypes(origOp, nextTile);
    SmallVector<vpux::NDTypeInterface> requiredOperands{curTileTypes[0], curTileTypes[1], nextTileTypes[0]};
    return requiredOperands;
}

SmallVector<vpux::NDTypeInterface> getRequiredOperandsForPipelining(VPU::NCEMaxPoolOp origOp,
                                                                    vpux::OutputTiling tiling) {
    // The tiling strategy follows last-tile-not-biggest
    // So just check the first two tiles are enough to make sure prefetchable
    auto curTile = tiling[0];
    auto nextTile = tiling[1];

    const auto& curTileTypes = getTileTypes(origOp, curTile);
    const auto& nextTileTypes = getTileTypes(origOp, nextTile);
    return {curTileTypes[0], curTileTypes[1], nextTileTypes[0]};
}

SmallVector<vpux::NDTypeInterface> getRequiredOperandsForPipelining(VPU::NCEAveragePoolOp origOp,
                                                                    vpux::OutputTiling tiling) {
    // The tiling strategy follows last-tile-not-biggest
    // So just check the first two tiles are enough to make sure prefetchable
    auto curTile = tiling[0];
    auto nextTile = tiling[1];

    const auto& curTileTypes = getTileTypes(origOp, curTile);
    const auto& nextTileTypes = getTileTypes(origOp, nextTile);
    return {curTileTypes[0], curTileTypes[1], nextTileTypes[0]};
}

int64_t getRequiredChannelSizeForPipelining(VPU::MaxPoolOp origOp, vpux::OutputTiling tiling) {
    auto curInputShape = getTileTypes(origOp, tiling[0])[0].getShape();
    auto nextInputShape = getTileTypes(origOp, tiling[1])[0].getShape();
    return curInputShape[Dims4D::Act::C] + nextInputShape[Dims4D::Act::C];
}

int64_t getRequiredChannelSizeForPipelining(VPU::NCEMaxPoolOp origOp, vpux::OutputTiling tiling) {
    auto curInputShape = getTileTypes(origOp, tiling[0])[0].getShape();
    auto nextInputShape = getTileTypes(origOp, tiling[1])[0].getShape();
    return curInputShape[Dims4D::Act::C] + nextInputShape[Dims4D::Act::C];
}

int64_t getRequiredChannelSizeForPipelining(VPU::NCEAveragePoolOp origOp, vpux::OutputTiling tiling) {
    auto curInputShape = getTileTypes(origOp, tiling[0])[0].getShape();
    auto nextInputShape = getTileTypes(origOp, tiling[1])[0].getShape();
    return curInputShape[Dims4D::Act::C] + nextInputShape[Dims4D::Act::C];
}

Byte getRequiredActWindowForPipelining(VPU::MaxPoolOp origOp) {
    const auto kernelSizeVals = Shape(parseIntArrayAttr<int64_t>(origOp.kernel_sizeAttr()));
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(origOp.stridesAttr()));

    //  Consider tiling does not change the element type
    const auto inType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto activationWindowSizePerTile = VPU::NCESparsity::getActivationWindowSize(
            VPU::NCESparsity::Mode::POOL, kernelSizeVals, kernelStridesVals[Dims4D::Strides::X],
            inType.getElementType(), 1);
    return Byte(activationWindowSizePerTile * 2);
}

Byte getRequiredActWindowForPipelining(VPU::NCEMaxPoolOp origOp) {
    const auto kernelSizeVals = Shape(parseIntArrayAttr<int64_t>(origOp.kernel_sizeAttr()));
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(origOp.stridesAttr()));

    //  Consider tiling does not change the element type
    const auto inType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto activationWindowSizePerTile = VPU::NCESparsity::getActivationWindowSize(
            VPU::NCESparsity::Mode::POOL, kernelSizeVals, kernelStridesVals[Dims4D::Strides::X],
            inType.getElementType(), 1);
    return Byte(activationWindowSizePerTile * 2);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::MaxPoolOp origOp, vpux::OutputTiling tiling,
                                                                   Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() <= 1) {
        return mlir::failure();
    }
    if (isNestedTiling(tiling)) {
        return mlir::failure();
    }

    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);

    Byte requiredCMX = Byte(0);

    requiredCMX = getRequiredCMXSize(getRequiredOperandsForPipelining(origOp, tiling));
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}'",
                  origOp->getLoc(), cmxSize, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::NCEMaxPoolOp origOp, vpux::OutputTiling tiling,
                                                                   Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() <= 1) {
        return mlir::failure();
    }
    if (isNestedTiling(tiling)) {
        return mlir::failure();
    }

    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);

    Byte requiredCMX = Byte(0);

    requiredCMX = getRequiredCMXSizeForNCEOps(getRequiredOperandsForPipelining(origOp, tiling),
                                              getRequiredChannelSizeForPipelining(origOp, tiling)) +
                  getRequiredActWindowForPipelining(origOp);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}'",
                  origOp->getLoc(), cmxSize, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::NCEAveragePoolOp origOp,
                                                                   vpux::OutputTiling tiling, Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() <= 1) {
        return mlir::failure();
    }
    if (isNestedTiling(tiling)) {
        return mlir::failure();
    }

    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);

    Byte requiredCMX = Byte(0);
    requiredCMX = getRequiredCMXSizeForNCEOps(getRequiredOperandsForPipelining(origOp, tiling),
                                              getRequiredChannelSizeForPipelining(origOp, tiling));
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}'",
                  origOp->getLoc(), cmxSize, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

// GroupConvolution

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::GroupConvolutionOp origOp, const TileInfo& outTile) {
    const auto origBiasShape = origOp.bias() != nullptr ? getShape(origOp.bias()) : ShapeRef();
    const auto origPadding = PadInfo(origOp.pads_begin(), origOp.pads_end());

    auto tileConf = vpux::backInferGroupConvTile(outTile, getShape(origOp.input()), getShape(origOp.filter()),
                                                 origBiasShape, origOp.strides(), origPadding);

    SmallVector<vpux::NDTypeInterface> tileTypes;

    tileTypes.push_back(origOp.input().getType().cast<vpux::NDTypeInterface>().extractDenseTile(
            tileConf.tiles[0].offsets, tileConf.tiles[0].shape));
    tileTypes.push_back(origOp.filter().getType().cast<vpux::NDTypeInterface>().extractDenseTile(
            tileConf.tiles[1].offsets, tileConf.tiles[1].shape));
    tileTypes.push_back(
            origOp.getType().cast<vpux::NDTypeInterface>().extractDenseTile(outTile.offsets, outTile.shape));

    return tileTypes;
}

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::NCEDepthConvolutionOp origOp, const TileInfo& outTile) {
    const auto inputTiles = origOp.backInferTileInfo(outTile, Logger::global()).tiles;
    auto inputTileType = origOp.input().getType().cast<vpux::NDTypeInterface>().extractDenseTile(inputTiles[0].offsets,
                                                                                                 inputTiles[0].shape);
    auto filterTileType = origOp.filter().getType().cast<vpux::NDTypeInterface>().extractDenseTile(
            inputTiles[1].offsets, inputTiles[1].shape);
    auto outputTileType =
            origOp.getType().cast<vpux::NDTypeInterface>().extractDenseTile(outTile.offsets, outTile.shape);

    if (origOp->hasAttr(VPU::multiClusterStrategy)) {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(nceOp == nullptr, "Op {0} has multiClusterStrategy but is not an NCEOp", origOp->getLoc());
        auto numClusters = VPU::getOptimalNumClusters(
                nceOp, outputTileType.getShape()[Dims4D::Act::C],
                nceOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
        return {VPU::getDistributedActivationTypeFromOp(nceOp, inputTileType, numClusters),
                VPU::getDistributedFilterTypeFromOp(nceOp, filterTileType, numClusters),
                VPU::getDistributedOutputTypeFromOp(nceOp, outputTileType, numClusters)};
    }

    return {inputTileType, filterTileType, outputTileType};
}

SmallVector<vpux::NDTypeInterface> getRequiredOperandsForPipelining(VPU::GroupConvolutionOp origOp,
                                                                    vpux::OutputTiling tiling) {
    // The tiling strategy follows last-tile-not-biggest
    // So just check the first two tiles are enough to make sure prefetchable
    auto curTile = tiling[0];
    auto nextTile = tiling[1];
    bool isWeightPrefetch = curTile.axis[Dims4D::Act::C] > 1;

    const auto& curTileTypes = getTileTypes(origOp, curTile);
    const auto& nextTileTypes = getTileTypes(origOp, nextTile);
    SmallVector<vpux::NDTypeInterface> requiredOperands{curTileTypes[0], getAlignedFilterType(curTileTypes),
                                                        curTileTypes[2]};
    if (isWeightPrefetch) {
        requiredOperands.push_back(getAlignedFilterType(nextTileTypes));
    } else {
        requiredOperands.push_back(nextTileTypes[0]);
    }
    return requiredOperands;
}

SmallVector<vpux::NDTypeInterface> getRequiredOperandsForPipelining(VPU::NCEDepthConvolutionOp origOp,
                                                                    vpux::OutputTiling tiling) {
    // The tiling strategy follows last-tile-not-biggest
    // So just check the first two tiles are enough to make sure prefetchable
    auto curTile = tiling[0];
    auto nextTile = tiling[1];
    bool isWeightPrefetch = curTile.axis[Dims4D::Act::C] > 1;

    const auto& curTileTypes = getTileTypes(origOp, curTile);
    const auto& nextTileTypes = getTileTypes(origOp, nextTile);

    return {curTileTypes[0], curTileTypes[1], curTileTypes[2], isWeightPrefetch ? nextTileTypes[1] : nextTileTypes[0]};
}

int64_t getRequiredChannelSizeForPipelining(VPU::GroupConvolutionOp origOp, vpux::OutputTiling tiling) {
    auto curFilterShape = getTileTypes(origOp, tiling[0])[1].getShape();
    auto nextFilterShape = getTileTypes(origOp, tiling[1])[1].getShape();
    return curFilterShape[Dims4D::Filter::OC] + nextFilterShape[Dims4D::Filter::OC];
}

int64_t getRequiredChannelSizeForPipelining(VPU::NCEDepthConvolutionOp origOp, vpux::OutputTiling tiling) {
    auto curFilterShape = getTileTypes(origOp, tiling[0])[1].getShape();
    auto nextFilterShape = getTileTypes(origOp, tiling[1])[1].getShape();
    return curFilterShape[Dims4D::Filter::OC] + nextFilterShape[Dims4D::Filter::OC];
}

Byte getRequiredActWindowForPipelining(VPU::GroupConvolutionOp origOp) {
    const auto inType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto filterType = origOp.filter().getType().cast<vpux::NDTypeInterface>();

    const Shape kernelSizeVals{filterType.getShape()[Dims4D::Filter::KY], filterType.getShape()[Dims4D::Filter::KX]};
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(origOp.stridesAttr()));

    const auto activationWindowSizePerTile = VPU::NCESparsity::getActivationWindowSize(
            VPU::NCESparsity::Mode::DW_CONV, kernelSizeVals, kernelStridesVals[Dims4D::Strides::X],
            inType.getElementType(), 1);

    return Byte(activationWindowSizePerTile * 2);
}

Byte getRequiredActWindowForPipelining(VPU::NCEDepthConvolutionOp origOp) {
    const auto inType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto filterType = origOp.filter().getType().cast<vpux::NDTypeInterface>();

    const Shape kernelSizeVals{filterType.getShape()[Dims4D::Filter::KY], filterType.getShape()[Dims4D::Filter::KX]};
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(origOp.stridesAttr()));

    const auto activationWindowSizePerTile = VPU::NCESparsity::getActivationWindowSize(
            VPU::NCESparsity::Mode::DW_CONV, kernelSizeVals, kernelStridesVals[Dims4D::Strides::X],
            inType.getElementType(), 1);

    return Byte(activationWindowSizePerTile * 2);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::GroupConvolutionOp origOp,
                                                                   vpux::OutputTiling tiling, Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() <= 1) {
        return mlir::failure();
    }
    if (isNestedTiling(tiling)) {
        return mlir::failure();
    }

    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);

    Byte requiredCMX = Byte(0);

    requiredCMX = getRequiredCMXSize(getRequiredOperandsForPipelining(origOp, tiling));
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}'",
                  origOp->getLoc(), cmxSize, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::NCEDepthConvolutionOp origOp,
                                                                   vpux::OutputTiling tiling, Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() <= 1) {
        return mlir::failure();
    }
    if (isNestedTiling(tiling)) {
        return mlir::failure();
    }

    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);

    Byte requiredCMX = Byte(0);

    requiredCMX = getRequiredCMXSizeForNCEOps(getRequiredOperandsForPipelining(origOp, tiling),
                                              getRequiredChannelSizeForPipelining(origOp, tiling)) +
                  getRequiredActWindowForPipelining(origOp);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}'",
                  origOp->getLoc(), cmxSize, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

//
// verifyEltwisePipeliningCMX
//

SmallVector<vpux::NDTypeInterface> getTileTypes(mlir::Operation* op, const TileInfo& outTile) {
    if (auto convOp = mlir::dyn_cast<VPU::ConvolutionOp>(op)) {
        return getTileTypes(convOp, outTile);
    }
    if (auto convOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(op)) {
        return getTileTypes(convOp, outTile);
    }
    if (auto poolOp = mlir::dyn_cast<VPU::MaxPoolOp>(op)) {
        return getTileTypes(poolOp, outTile);
    }
    if (auto poolOp = mlir::dyn_cast<VPU::NCEMaxPoolOp>(op)) {
        return getTileTypes(poolOp, outTile);
    }
    if (auto poolOp = mlir::dyn_cast<VPU::NCEAveragePoolOp>(op)) {
        return getTileTypes(poolOp, outTile);
    }
    if (auto groupConvOp = mlir::dyn_cast<VPU::GroupConvolutionOp>(op)) {
        return getTileTypes(groupConvOp, outTile);
    }
    if (auto depthConvOp = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(op)) {
        return getTileTypes(depthConvOp, outTile);
    }

    auto tileConf = vpux::backInferEltwiseTile(op, outTile);

    SmallVector<vpux::NDTypeInterface> tileTypes;

    tileTypes.push_back(op->getOperand(0).getType().cast<vpux::NDTypeInterface>().extractDenseTile(
            tileConf.tiles[0].offsets, tileConf.tiles[0].shape));
    tileTypes.push_back(op->getOperand(1).getType().cast<vpux::NDTypeInterface>().extractDenseTile(
            tileConf.tiles[1].offsets, tileConf.tiles[1].shape));
    tileTypes.push_back(
            op->getResult(0).getType().cast<vpux::NDTypeInterface>().extractDenseTile(outTile.offsets, outTile.shape));

    return tileTypes;
}

SmallVector<vpux::NDTypeInterface> getRequiredOperandsForPipelining(mlir::Operation* op, vpux::OutputTiling tiling) {
    // The tiling strategy follows last-tile-not-biggest
    // So just check the first two tiles are enough to make sure prefetchable
    auto curTile = tiling[0];
    auto nextTile = tiling[1];

    const auto& curTileTypes = getTileTypes(op, curTile);
    const auto& nextTileTypes = getTileTypes(op, nextTile);

    return SmallVector<vpux::NDTypeInterface>{curTileTypes[0], curTileTypes[1], curTileTypes[2], nextTileTypes[0],
                                              nextTileTypes[1]};
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyEltwisePipeliningCMX(mlir::Operation* op,
                                                                          vpux::OutputTiling tiling, Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() <= 1) {
        return mlir::failure();
    }
    if (isNestedTiling(tiling)) {
        return mlir::failure();
    }

    auto module = op->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);

    Byte requiredCMX = Byte(0);

    requiredCMX = getRequiredCMXSizeForNCEOps({getRequiredOperandsForPipelining(op, tiling)}, 0);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}'", op->getLoc(),
                  cmxSize, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::AddOp origOp, vpux::OutputTiling tiling,
                                                                   Logger log) {
    return verifyEltwisePipeliningCMX(origOp.getOperation(), tiling, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::MultiplyOp origOp, vpux::OutputTiling tiling,
                                                                   Logger log) {
    return verifyEltwisePipeliningCMX(origOp.getOperation(), tiling, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::SubtractOp origOp, vpux::OutputTiling tiling,
                                                                   Logger log) {
    return verifyEltwisePipeliningCMX(origOp.getOperation(), tiling, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::AndOp origOp, vpux::OutputTiling tiling,
                                                                   Logger log) {
    return verifyEltwisePipeliningCMX(origOp.getOperation(), tiling, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::NCEEltwiseOp origOp, vpux::OutputTiling tiling,
                                                                   Logger log) {
    return verifyEltwisePipeliningCMX(origOp.getOperation(), tiling, log);
}

//
// verifyPrefetchCMX
//

Byte getRequiredCMXForWeight(VPU::ConvolutionOp convOp, const vpux::TileInfo& tiling) {
    auto tileTypes = getTileTypes(convOp.getOperation(), tiling);
    const auto lastFilterTileType = tileTypes[1];
    return getRequiredCMXSize({lastFilterTileType});
}

Byte getRequiredCMXForWeight(VPU::NCEConvolutionOp convOp, const vpux::TileInfo& tiling) {
    auto tileTypes = getTileTypes(convOp.getOperation(), tiling);
    const auto lastFilterTileType = tileTypes[1];
    const auto outputTileType = tileTypes[2];
    const auto OC = outputTileType.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps({lastFilterTileType}, OC);
}

Byte getRequiredCMX(VPU::ConvolutionOp convOp, const vpux::TileInfo& tiling) {
    auto tileTypes = getTileTypes(convOp.getOperation(), tiling);
    const auto lastInputTileType = tileTypes[0];
    const auto lastFilterTileType = tileTypes[1];
    const auto lastOutputTileType = tileTypes[2];
    return getRequiredCMXSize({lastInputTileType, lastFilterTileType, lastOutputTileType});
}

Byte getRequiredCMX(VPU::NCEConvolutionOp convOp, const vpux::TileInfo& tiling) {
    auto tileTypes = getTileTypes(convOp.getOperation(), tiling);
    const auto lastInputTileType = tileTypes[0];
    const auto lastFilterTileType = tileTypes[1];
    const auto lastOutputTileType = tileTypes[2];
    const auto OC = lastOutputTileType.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps({lastInputTileType, lastFilterTileType, lastOutputTileType}, OC);
}

Byte getRequiredCMXForWeight(VPU::GroupConvolutionOp gConvOp, const vpux::TileInfo& tiling) {
    auto tileTypes = getTileTypes(gConvOp.getOperation(), tiling);
    const auto filterTileType = tileTypes[1];
    return getRequiredCMXSize({filterTileType});
}

Byte getRequiredCMXForWeight(VPU::NCEDepthConvolutionOp gConvOp, const vpux::TileInfo& tiling) {
    auto tileTypes = getTileTypes(gConvOp.getOperation(), tiling);
    const auto filterTileShape = tileTypes[1];
    const auto outputTileType = tileTypes[2];
    const auto OC = outputTileType.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps({filterTileShape}, OC);
}

Byte getRequiredCMX(VPU::GroupConvolutionOp gConvOp, const vpux::TileInfo& tiling) {
    auto tileTypes = getTileTypes(gConvOp.getOperation(), tiling);
    const auto inputTileType = tileTypes[0];
    return getRequiredCMXSize({inputTileType, inputTileType}) + getRequiredCMXForWeight(gConvOp, tiling);
}

Byte getRequiredCMX(VPU::NCEDepthConvolutionOp dConvOp, const vpux::TileInfo& tiling) {
    auto tileTypes = getTileTypes(dConvOp.getOperation(), tiling);
    const auto inputTileType = tileTypes[0];
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(dConvOp.rawFilterShape()));
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];
    const Shape kernelSizeVals{KY, KX};
    auto kernelStrides = dConvOp.strides();
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(kernelStrides));

    const auto activationWindowSize = VPU::NCESparsity::getActivationWindowSize(
            VPU::NCESparsity::Mode::CM_CONV, kernelSizeVals, kernelStridesVals[Dims4D::Strides::X],
            inputTileType.getElementType(), 1);

    return getRequiredCMXSizeForNCEOps({inputTileType, inputTileType}, 0) + activationWindowSize * 1_Byte +
           getRequiredCMXForWeight(dConvOp, tiling);
}

Byte getRequiredCMXForWeight(VPU::MaxPoolOp /*op*/, const vpux::TileInfo& /*tiling*/) {
    return Byte(0);
}

Byte getRequiredCMXForWeight(VPU::NCEMaxPoolOp /*op*/, const vpux::TileInfo& /*tiling*/) {
    return Byte(0);
}

Byte getRequiredCMXForWeight(VPU::NCEAveragePoolOp /*op*/, const vpux::TileInfo& /*tiling*/) {
    return Byte(0);
}

Byte getRequiredCMX(VPU::MaxPoolOp poolOp, const vpux::TileInfo& tiling) {
    auto tileTypes = getTileTypes(poolOp.getOperation(), tiling);
    auto inputType = tileTypes[0];
    auto outputType = tileTypes[1];
    return getRequiredCMXSize({inputType, outputType});
}

Byte getRequiredCMX(VPU::NCEMaxPoolOp poolOp, const vpux::TileInfo& tiling) {
    auto tileTypes = getTileTypes(poolOp.getOperation(), tiling);
    auto inputType = tileTypes[0];
    auto outputType = tileTypes[1];
    auto kernelSize = poolOp.kernel_size();
    auto kernelStrides = poolOp.strides();
    const auto inputShape = inputType.getShape();
    const auto IC = inputShape[Dims4D::Act::C];

    const auto kernelSizeVals = Shape(parseIntArrayAttr<int64_t>(kernelSize));
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(kernelStrides));

    const auto activationWindowSize = VPU::NCESparsity::getActivationWindowSize(
            VPU::NCESparsity::Mode::POOL, kernelSizeVals, kernelStridesVals[Dims4D::Strides::X],
            inputType.getElementType(), 1);

    return getRequiredCMXSizeForNCEOps({inputType, outputType}, IC) + activationWindowSize * 1_Byte;
}

Byte getRequiredCMX(VPU::NCEAveragePoolOp poolOp, const vpux::TileInfo& tiling) {
    auto tileTypes = getTileTypes(poolOp.getOperation(), tiling);
    auto inputType = tileTypes[0];
    auto outputType = tileTypes[1];
    auto kernelSize = poolOp.kernel_size();
    auto kernelStrides = poolOp.strides();
    const auto inputShape = inputType.getShape();
    const auto IC = inputShape[Dims4D::Act::C];

    const auto kernelSizeVals = Shape(parseIntArrayAttr<int64_t>(kernelSize));
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(kernelStrides));

    return getRequiredCMXSizeForNCEOps({inputType, outputType}, IC);
}

Byte getEltwiseRequiredCMX(mlir::Operation* op, const vpux::TileInfo& tiling) {
    auto tileTypes = getTileTypes(op, tiling);
    auto firstInputType = tileTypes[0];
    auto secondInputType = tileTypes[1];
    auto outputType = tileTypes[2];
    return getRequiredCMXSize({firstInputType, secondInputType, outputType});
}

Byte getRequiredCMX(VPU::AddOp op, const vpux::TileInfo& tiling) {
    return getEltwiseRequiredCMX(op.getOperation(), tiling);
}

Byte getRequiredCMXForWeight(VPU::AddOp /*op*/, const vpux::TileInfo& /*tiling*/) {
    return Byte(0);
}

Byte getRequiredCMX(VPU::MultiplyOp op, const vpux::TileInfo& tiling) {
    return getEltwiseRequiredCMX(op.getOperation(), tiling);
}

Byte getRequiredCMXForWeight(VPU::MultiplyOp /*op*/, const vpux::TileInfo& /*tiling*/) {
    return Byte(0);
}

Byte getRequiredCMX(VPU::SubtractOp op, const vpux::TileInfo& tiling) {
    return getEltwiseRequiredCMX(op.getOperation(), tiling);
}

Byte getRequiredCMXForWeight(VPU::SubtractOp /*op*/, const vpux::TileInfo& /*tiling*/) {
    return Byte(0);
}

Byte getRequiredCMX(VPU::AndOp op, const vpux::TileInfo& tiling) {
    return getEltwiseRequiredCMX(op.getOperation(), tiling);
}

Byte getRequiredCMXForWeight(VPU::AndOp /*op*/, const vpux::TileInfo& /*tiling*/) {
    return Byte(0);
}

Byte getRequiredCMX(VPU::NCEEltwiseOp op, const vpux::TileInfo& tiling) {
    return getEltwiseRequiredCMX(op.getOperation(), tiling);
}

Byte getRequiredCMXForWeight(VPU::NCEEltwiseOp /*op*/, const vpux::TileInfo& /*tiling*/) {
    return Byte(0);
}

Byte getRequiredCMXForWeight(mlir::Operation* op, const vpux::TileInfo& tiling) {
    return llvm::TypeSwitch<mlir::Operation*, Byte>(op)
            .Case<VPU::ConvolutionOp>([&](VPU::ConvolutionOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling);
            })
            .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling);
            })
            .Case<VPU::MaxPoolOp>([&](VPU::MaxPoolOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling);
            })
            .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling);
            })
            .Case<VPU::NCEAveragePoolOp>([&](VPU::NCEAveragePoolOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling);
            })
            .Case<VPU::AddOp>([&](VPU::AddOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling);
            })
            .Case<VPU::MultiplyOp>([&](VPU::MultiplyOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling);
            })
            .Case<VPU::SubtractOp>([&](VPU::SubtractOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling);
            })
            .Case<VPU::AndOp>([&](VPU::AndOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling);
            })
            .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling);
            })
            .Case<VPU::GroupConvolutionOp>([&](VPU::GroupConvolutionOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling);
            })
            .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling);
            })
            .Default([](mlir::Operation* unknownOp) -> Byte {
                VPUX_THROW("Operation CMX check '{0}' at '{1}' is not implemented", unknownOp->getName(),
                           unknownOp->getLoc());
            });
}

Byte getRequiredCMX(mlir::Operation* op, const vpux::TileInfo& tiling) {
    return llvm::TypeSwitch<mlir::Operation*, Byte>(op)
            .Case<VPU::ConvolutionOp>([&](VPU::ConvolutionOp origOp) {
                return getRequiredCMX(origOp, tiling);
            })
            .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp origOp) {
                return getRequiredCMX(origOp, tiling);
            })
            .Case<VPU::MaxPoolOp>([&](VPU::MaxPoolOp origOp) {
                return getRequiredCMX(origOp, tiling);
            })
            .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp origOp) {
                return getRequiredCMX(origOp, tiling);
            })
            .Case<VPU::NCEAveragePoolOp>([&](VPU::NCEAveragePoolOp origOp) {
                return getRequiredCMX(origOp, tiling);
            })
            .Case<VPU::AddOp>([&](VPU::AddOp origOp) {
                return getRequiredCMX(origOp, tiling);
            })
            .Case<VPU::MultiplyOp>([&](VPU::MultiplyOp origOp) {
                return getRequiredCMX(origOp, tiling);
            })
            .Case<VPU::SubtractOp>([&](VPU::SubtractOp origOp) {
                return getRequiredCMX(origOp, tiling);
            })
            .Case<VPU::AndOp>([&](VPU::AndOp origOp) {
                return getRequiredCMX(origOp, tiling);
            })
            .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp origOp) {
                return getRequiredCMX(origOp, tiling);
            })
            .Case<VPU::GroupConvolutionOp>([&](VPU::GroupConvolutionOp origOp) {
                return getRequiredCMX(origOp, tiling);
            })
            .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp origOp) {
                return getRequiredCMX(origOp, tiling);
            })
            .Default([](mlir::Operation* unknownOp) -> Byte {
                VPUX_THROW("Operation CMX check '{0}' at '{1}' is not implemented", unknownOp->getName(),
                           unknownOp->getLoc());
            });
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPrefetchCMX(mlir::Operation* op, vpux::OutputTiling tiling,
                                                                 mlir::Operation* parentOp,
                                                                 vpux::OutputTiling parentTiling, Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() < 1 || parentTiling.size() < 1) {
        return mlir::failure();
    }
    auto module = op->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);

    // Calculate the CMX memory required by the last tile of parent Op
    auto lastParentTile = parentTiling[parentTiling.size() - 1];
    auto cmxRequiredByParent = getRequiredCMX(parentOp, lastParentTile);

    // Calculate the CMX memory required by the first tile of current op to prefetch
    auto firstPrefetchTile = tiling[tiling.size() - 1];
    auto cmxRequiredToPrefetch = getRequiredCMXForWeight(op, firstPrefetchTile);
    auto cmxWithFragmentationRatio = Byte(
            static_cast<int64_t>(std::ceil(static_cast<double>(cmxSize.count()) * VPU::FRAGMENTATION_AVOID_RATIO)));

    if (cmxRequiredByParent + cmxRequiredToPrefetch > cmxWithFragmentationRatio) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}', required by "
                  "parent {3}",
                  op->getLoc(), cmxWithFragmentationRatio, cmxRequiredByParent + cmxRequiredToPrefetch,
                  cmxRequiredByParent);
        return mlir::failure();
    }

    return mlir::success();
}

//
// verifyKernel
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(mlir::Location loc, int64_t KY, int64_t KX, int64_t SY,
                                                            int64_t SX, int64_t padTop, int64_t padBottom,
                                                            int64_t padLeft, int64_t padRight, VPU::ArchKind arch,
                                                            Logger log) {
    log.setName("NCEInvariant");

    static const int32_t NCE_MAX_STRIDE_SIZE = 8;

    if (KY > VPU::NCEInvariant::MAX_KERNEL_SIZE || KY <= 0) {
        log.trace("[{0}] Unsupported kernel height dimension '{1}', must be in range [1, {2}]", loc, KY,
                  VPU::NCEInvariant::MAX_KERNEL_SIZE);
        return mlir::failure();
    }
    if (KX > VPU::NCEInvariant::MAX_KERNEL_SIZE || KX <= 0) {
        log.trace("[{0}] Unsupported kernel width dimension '{1}', must be in range [1, {2}]", loc, KX,
                  VPU::NCEInvariant::MAX_KERNEL_SIZE);
        return mlir::failure();
    }

    if (SX != SY && arch != VPU::ArchKind::VPUX37XX) {
        log.trace("[{0}] Asymmetric strides are not supported", loc);
        return mlir::failure();
    }
    if (SY > NCE_MAX_STRIDE_SIZE || SY <= 0) {
        log.trace("[{0}] Unsupported stride height dimension '{1}', must be in range [1, {2}]", loc, SY,
                  NCE_MAX_STRIDE_SIZE);
        return mlir::failure();
    }
    if (SX > NCE_MAX_STRIDE_SIZE || SX <= 0) {
        log.trace("[{0}] Unsupported stride width dimension '{1}', must be in range [1, {2}]", loc, SX,
                  NCE_MAX_STRIDE_SIZE);
        return mlir::failure();
    }

    if (padTop < 0 || (padTop > 1 && padTop > KY / 2)) {
        log.trace("[{0}] Unsupported padding '{1}', must be in range [0, {2}]", loc, padTop, KY / 2);
        return mlir::failure();
    }
    if (padBottom < 0 || (padBottom > 1 && padBottom > KY / 2)) {
        log.trace("[{0}] Unsupported padding '{1}', must be in range [0, {2}]", loc, padBottom, KY / 2);
        return mlir::failure();
    }
    if (padLeft < 0 || (padLeft > 1 && padLeft > KX / 2)) {
        log.trace("[{0}] Unsupported padding '{1}', must be in range [0, {2}]", loc, padLeft, KX / 2);
        return mlir::failure();
    }
    if (padRight < 0 || (padRight > 1 && padRight > KX / 2)) {
        log.trace("[{0}] Unsupported padding '{1}', must be in range [0, {2}]", loc, padRight, KX / 2);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IE::ConvolutionOp origOp, Logger log) {
    log.setName("NCEInvariant");

    if (origOp.input().getType().cast<vpux::NDTypeInterface>().getRank() != 4) {
        return mlir::failure();
    }

    const auto dilations = parseIntArrayAttr<int64_t>(origOp.dilations());
    if (dilations[0] != 1 || dilations[1] != 1) {
        log.trace("[{0}] Unsupported kernel dilations '{1}'", origOp->getLoc(), dilations);
        return mlir::failure();
    }

    const auto filterShape = getShape(origOp.filter());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    const auto padTop = padsBegin[0];
    const auto padBottom = padsEnd[0];
    const auto padLeft = padsBegin[1];
    const auto padRight = padsEnd[1];

    const auto arch = VPU::getArch(origOp->getParentOfType<mlir::ModuleOp>());
    return verifyKernel(origOp->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, arch, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(VPU::NCEConvolutionOp, Logger) {
    // VPU.NCE operations guarantees that invariants
    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IE::MaxPoolOp origOp, Logger log) {
    log.setName("NCEInvariant");

    if (origOp.input().getType().cast<vpux::NDTypeInterface>().getRank() != 4) {
        return mlir::failure();
    }

    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    if (kernelSize[0] != kernelSize[1]) {
        log.trace("[{0}] Asymmetric kernel is not supported", origOp->getLoc());
        return mlir::failure();
    }
    const auto KY = kernelSize[0];
    const auto KX = kernelSize[1];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    const auto padTop = padsBegin[0];
    const auto padBottom = padsEnd[0];
    const auto padLeft = padsBegin[1];
    const auto padRight = padsEnd[1];

    const auto arch = VPU::getArch(origOp->getParentOfType<mlir::ModuleOp>());
    return verifyKernel(origOp->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, arch, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(VPU::NCEMaxPoolOp, Logger) {
    // VPU.NCE operations guarantees that invariants
    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IE::AvgPoolOp origOp, Logger log) {
    log.setName("NCEInvariant");

    if (origOp.input().getType().cast<vpux::NDTypeInterface>().getRank() != 4) {
        return mlir::failure();
    }

    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    if (kernelSize[0] != kernelSize[1]) {
        log.trace("[{0}] Asymmetric kernel is not supported", origOp->getLoc());
        return mlir::failure();
    }
    const auto KY = kernelSize[0];
    const auto KX = kernelSize[1];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    const auto padTop = padsBegin[0];
    const auto padBottom = padsEnd[0];
    const auto padLeft = padsBegin[1];
    const auto padRight = padsEnd[1];

    const auto arch = VPU::getArch(origOp->getParentOfType<mlir::ModuleOp>());
    return verifyKernel(origOp->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, arch, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(VPU::NCEAveragePoolOp, Logger) {
    // VPU.NCE operations guarantees that invariants
    return mlir::success();
}

//
// verifyEltwiseKernel
//

static mlir::LogicalResult verifyEltwiseKernel(vpux::NDTypeInterface input1, vpux::NDTypeInterface input2,
                                               vpux::NDTypeInterface output, const bool allowDifferentScales = false,
                                               const bool allowDifferentZp = true) {
    // Eltwise add is expected to have the same shapes for all operands
    if (input1.getRank() != 4 || input2.getRank() != 4 || output.getRank() != 4) {
        return mlir::failure();
    }

    if (input1.getShape() != input2.getShape())
        return mlir::failure();

    // Output type can differ from input type. In case of quantization
    // this can be different quant scale value.
    // Input types can also differ when both of them are quantized. E.g. scale value for Eltwise Multiply
    auto input1ElemType = input1.getElementType();
    auto input2ElemType = input2.getElementType();

    if (!input1ElemType.isa<mlir::quant::QuantizedType>() && !input2ElemType.isa<mlir::quant::QuantizedType>()) {
        if (input1ElemType != input2ElemType) {
            return mlir::failure();
        }
    } else if (input1ElemType.isa<mlir::quant::UniformQuantizedType>() &&
               input2ElemType.isa<mlir::quant::UniformQuantizedType>()) {
        auto qInput1 = input1ElemType.cast<mlir::quant::UniformQuantizedType>();
        auto qInput2 = input2ElemType.cast<mlir::quant::UniformQuantizedType>();

        if (qInput1.getExpressedType() != qInput2.getExpressedType() ||
            qInput1.getStorageType() != qInput2.getStorageType() || qInput1.isSigned() != qInput2.isSigned()) {
            return mlir::failure();
        }

        if (!allowDifferentZp && qInput1.getZeroPoint() != qInput2.getZeroPoint())
            return mlir::failure();

        if (!allowDifferentScales && qInput1.getScale() != qInput2.getScale())
            return mlir::failure();
    } else {
        VPUX_THROW("Unsupported inputs type. in1='{0}', in2='{1}'", input1ElemType, input2ElemType);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IE::AddOp origOp, Logger) {
    auto input1Type = origOp.input1().getType().cast<vpux::NDTypeInterface>();
    auto input2Type = origOp.input2().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();

    const auto arch = VPU::getArch(origOp->getParentOfType<mlir::ModuleOp>());

    return verifyEltwiseKernel(input1Type, input2Type, outputType, supportsPerInputEltwiseScale(arch));
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IE::MultiplyOp origOp, Logger) {
    auto input1Type = origOp.input1().getType().cast<vpux::NDTypeInterface>();
    auto input2Type = origOp.input2().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    return verifyEltwiseKernel(input1Type, input2Type, outputType, true);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IE::SubtractOp origOp, Logger) {
    auto input1Type = origOp.input1().getType().cast<vpux::NDTypeInterface>();
    auto input2Type = origOp.input2().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    return verifyEltwiseKernel(input1Type, input2Type, outputType);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IE::AndOp origOp, Logger) {
    auto input1Type = origOp.input1().getType().cast<vpux::NDTypeInterface>();
    auto input2Type = origOp.input2().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    return verifyEltwiseKernel(input1Type, input2Type, outputType);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(VPU::NCEEltwiseOp, Logger) {
    // VPU.NCE operations guarantees that invariants
    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IE::GroupConvolutionOp origOp, Logger log) {
    log.setName("NCEInvariant");

    if (origOp.input().getType().cast<vpux::NDTypeInterface>().getRank() != 4) {
        return mlir::failure();
    }
    if (origOp.filter().getType().cast<vpux::NDTypeInterface>().getRank() != 4) {
        return mlir::failure();
    }

    const auto dilations = parseIntArrayAttr<int64_t>(origOp.dilations());
    if (dilations[0] != 1 || dilations[1] != 1) {
        log.trace("[{0}] Unsupported kernel dilations '{1}'", origOp->getLoc(), dilations);
        return mlir::failure();
    }

    const auto filterShape = getShape(origOp.filter());
    const auto filtersPerInChan = filterShape[Dims4D::Filter::IC];
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    if (!origOp.groups().hasValue()) {
        log.trace("[{0}] Grouped convolution does not have groups", origOp->getLoc());
        return mlir::failure();
    }
    if (origOp.groups().getValue() != OC) {
        log.trace("[{0}] Unsupported group size: '{1}' expected '{2}'", origOp->getLoc(), origOp.groups(), OC);
        return mlir::failure();
    }
    if (filtersPerInChan != 1) {
        log.trace("[{0}] Group Convolution with more than one filter per channel is not supported", origOp->getLoc());
        return mlir::failure();
    }

    const auto inputShape = getShape(origOp.input());
    const auto IC = inputShape[Dims4D::Act::C];
    if (OC != IC) {
        log.trace("[{0}] Group Convolution has {1} groups, expected {2}", origOp->getLoc(), OC, IC);
        return mlir::failure();
    }

    const auto kernelStrides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    const auto padTop = padsBegin[0];
    const auto padBottom = padsEnd[0];
    const auto padLeft = padsBegin[1];
    const auto padRight = padsEnd[1];

    const auto arch = VPU::getArch(origOp->getParentOfType<mlir::ModuleOp>());
    return verifyKernel(origOp->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, arch, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(VPU::NCEDepthConvolutionOp, Logger) {
    // VPU.NCE operations guarantees that invariants
    return mlir::success();
}
