//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;
using namespace VPUIP;

namespace {

bool doesSwKernelSupportTiling(VPUIP::SwKernelOp swKernelOp, vpux::Logger log) {
    if (swKernelOp.getOutputs().size() != 1) {
        // SwKernel is tiled by dividing output into several small outputs. So tiling is disabled for SwKernel with
        // mulit outputs. The pass doesn't know how to divide correctly on all outputs.
        log.trace("SW kernel op has more than one output at '{0}'", swKernelOp->getLoc());
        return false;
    }

    if (!isSwKernelTilingSupported(swKernelOp)) {
        return false;
    }

    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (kernelEntryName == "singleShaveMVN") {
        auto kernelArgsRange = [](VPUIP::SwKernelOp swKernelOp) {
            SmallVector<mlir::Attribute> attrStorage;

            for (auto&& kernelRun : swKernelOp.body().getOps<VPUIP::SwKernelRun>()) {
                if (kernelRun.attrs().hasValue()) {
                    const mlir::ArrayAttr arrayAttrs = kernelRun.attrs().getValue();
                    const auto& attrs = arrayAttrs.getValue();
                    for (const auto& attr : attrs) {
                        attrStorage.push_back(attr);
                    }
                }
            }
            return attrStorage;
        };

        auto taskArgs = kernelArgsRange(swKernelOp);
        const auto acrossChannels = taskArgs[0].dyn_cast<mlir::BoolAttr>();
        return !acrossChannels.getValue();
    }

    return true;
}

Dim getSwKernelTileDim(VPUIP::SwKernelOp swKernelOp) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (kernelEntryName == "singleShaveMVN") {
        // MVN only supports tiling on C
        return Dims4D::Act::C;
    } else {
        // get highest dim by default
        const auto output = swKernelOp->getResult(0);
        const auto outputType = output.getType().cast<vpux::NDTypeInterface>();
        const auto outOrder = outputType.getDimsOrder();
        const auto outShape = outputType.getShape();
        for (auto i : irange(outOrder.numDims())) {
            auto dim = outOrder.dimAt(i);
            if (outShape[dim] > 1) {
                return dim;
            }
        }
        return outOrder.dimAt(0);
    }
}

OutputTiling getSwKernelOutputTiling(VPUIP::SwKernelOp swKernelOp, int64_t maxNumTiles, vpux::Logger log) {
    const auto output = swKernelOp->getResult(0);
    const auto outputType = output.getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();
    VPUX_THROW_UNLESS(outputShape.size() == 4, "Unsupported operation '{0}' at '{1}', it has non 4D result",
                      swKernelOp->getName(), swKernelOp->getLoc());

    Shape nTilesOnDim(outputShape.size(), 1);
    const auto tileDim = getSwKernelTileDim(swKernelOp);
    log.trace("Tile Dim is {0}", tileDim);
    nTilesOnDim[tileDim] = std::min(maxNumTiles, outputShape[tileDim]);
    return fillDividedTiles(nTilesOnDim, outputShape, None);
}

InputTiling backInferSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile) {
    SmallVector<TileInfo> inputTiles;
    for (auto origInput : swKernelOp->getOperands()) {
        const auto curShape = getShape(origInput);
        VPUX_THROW_UNLESS(curShape.size() == outputTile.shape.size(),
                          "Can't tile SwKernel operation '{0}' at '{1}', which has operands with different rank",
                          swKernelOp->getName(), swKernelOp->getLoc());

        // Handle broadcasted inputs
        auto curTile = outputTile;
        for (auto ind : irange(curShape.size())) {
            const auto d = Dim(ind);
            if (curShape[d] == 1) {
                curTile.shape[d] = 1;
                curTile.offsets[d] = 0;
            }
        }

        inputTiles.push_back(curTile);
    }
    return TilingInfo{inputTiles};
}

SmallVector<mlir::Value> getOuterMappingOperand(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange innerOperands) {
    auto clusterTilingOp = swKernelOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
    auto isClusterTilingApplied = clusterTilingOp != nullptr;
    SmallVector<mlir::Value> outerOperands;
    for (auto operand : innerOperands) {
        if (!isClusterTilingApplied) {
            outerOperands.push_back(operand);
        } else {
            auto blockArg = operand.dyn_cast<mlir::BlockArgument>();
            VPUX_THROW_WHEN(blockArg == nullptr, "Matching argument was not identified");
            auto outerOperand = clusterTilingOp->getOperand(blockArg.getArgNumber());
            outerOperands.push_back(outerOperand);
        }
    }
    return outerOperands;
}

VPUIP::NCEClusterTilingOp createNewTilingCopyOp(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Type outType,
                                                ArrayRef<mlir::Value> operands) {
    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
    };
    return rewriter.create<VPUIP::NCEClusterTilingOp>(loc, outType, operands, bodyBuilder);
}

//
// SwKernelRewriterBase
//

class SwKernelRewriterBase : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    SwKernelRewriterBase(mlir::MLIRContext* ctx, int64_t shaveCount, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _shaveCount(shaveCount), _log(log) {
        setDebugName("SwKernelRewriterBase");
    }
    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp SwKernelOp, mlir::PatternRewriter& rewriter) const final;

    virtual bool checkTilePattern(VPUIP::SwKernelOp swKernelOp) const = 0;
    virtual SmallVector<mlir::Value> createNewInputs(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange operands,
                                                     ArrayRef<TileInfo> inTiles, bool stridedDataAccessSupport,
                                                     mlir::PatternRewriter& rewriter) const = 0;
    virtual SmallVector<mlir::Value> createNewOutBuffs(mlir::ValueRange operands, ArrayRef<TileInfo> inTiles,
                                                       bool stridedDataAccessSupport,
                                                       mlir::PatternRewriter& rewriter) const = 0;
    virtual void createNewSwKernelAndConcatOp(VPUIP::SwKernelOp swKernelOp, ArrayRef<mlir::Value> newInputs,
                                              ArrayRef<mlir::Value> newOutBufs, const OutputTiling& outTiles,
                                              bool stridedDataAccessSupport,

                                              mlir::PatternRewriter& rewriter) const = 0;

protected:
    int64_t _shaveCount;
    Logger _log;
};

/*
 Tile SwKernel within a cluster. Note that copy op is inserted to provide continuous buffer for each tile of SwKernel

     |          |                      |
Copy(DDR2CMX) Alloc               /            \
     \        /             SubView          Alloc
      SwKernel                   |              |
    (SwKernelRun)    =>     Copy(DDR2CMX)       |
         |                       \             /
    Copy(CMX2DDR)            SwKernel(Multi SwKerneRun)
                                      |
                                    Concat
*/
mlir::LogicalResult SwKernelRewriterBase::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                          mlir::PatternRewriter& rewriter) const {
    auto swKernelRun = swKernelOp.body().getOps<VPUIP::SwKernelRun>();
    if (std::distance(swKernelRun.begin(), swKernelRun.end()) > 1) {
        // swKernelOp has already been tiled
        return mlir::failure();
    }

    if (!doesSwKernelSupportTiling(swKernelOp, _log)) {
        // swKernelOp doesn't support tiling on mulit shaves
        return mlir::failure();
    }

    if (!checkTilePattern(swKernelOp)) {
        return mlir::failure();
    }

    _log.trace("process swKernelOp at {0}", swKernelOp->getLoc());

    SmallVector<mlir::Value> newInputs;
    SmallVector<mlir::Value> newOutBuffs;
    auto outTiles = getSwKernelOutputTiling(swKernelOp, _shaveCount, _log);
    if (outTiles.size() == 1) {
        return mlir::failure();
    }
    auto stridedDataAccessSupport = isStridedDataAccessSupported(swKernelOp);

    for (auto outTile : outTiles) {
        auto inTiles = backInferSwKernelInputTile(swKernelOp, outTile);
        auto inputs = getOuterMappingOperand(swKernelOp, swKernelOp.inputs());
        auto outBuffs = getOuterMappingOperand(swKernelOp, swKernelOp.output_buffs());

        SmallVector<TileInfo> inputTiles(inTiles.tiles.begin(), inTiles.tiles.begin() + inputs.size());
        SmallVector<TileInfo> outputTiles(inTiles.tiles.begin() + inputs.size(), inTiles.tiles.end());

        newInputs.append(createNewInputs(swKernelOp, inputs, inputTiles, stridedDataAccessSupport, rewriter));
        newOutBuffs.append(createNewOutBuffs(outBuffs, outputTiles, stridedDataAccessSupport, rewriter));
    }

    createNewSwKernelAndConcatOp(swKernelOp, newInputs, newOutBuffs, outTiles, stridedDataAccessSupport, rewriter);
    return mlir::success();
}

//
// SwKernelRewriter
//

class SwKernelRewriter final : public SwKernelRewriterBase {
public:
    SwKernelRewriter(mlir::MLIRContext* ctx, int64_t shaveCout, Logger log): SwKernelRewriterBase(ctx, shaveCout, log) {
        setDebugName("SwKernelRewriter");
    }

    bool checkTilePattern(VPUIP::SwKernelOp swKernelOp) const override;
    SmallVector<mlir::Value> createNewInputs(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange operands,
                                             ArrayRef<TileInfo> inTiles, bool stridedDataAccessSupport,
                                             mlir::PatternRewriter& rewriter) const override;
    SmallVector<mlir::Value> createNewOutBuffs(mlir::ValueRange operands, ArrayRef<TileInfo> inTiles,
                                               bool stridedDataAccessSupport,
                                               mlir::PatternRewriter& rewriter) const override;

    void createNewSwKernelAndConcatOp(VPUIP::SwKernelOp swKernelOp, ArrayRef<mlir::Value> newInputs,
                                      ArrayRef<mlir::Value> newOutBufs, const OutputTiling& outTiles,
                                      bool stridedDataAccessSupport,

                                      mlir::PatternRewriter& rewriter) const override;
};

bool SwKernelRewriter::checkTilePattern(VPUIP::SwKernelOp swKernelOp) const {
    if (mlir::isa<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp())) {
        return false;
    }

    if (isStridedDataAccessSupported(swKernelOp)) {
        return true;
    }

    // Strided data access is not supported, need insert extra copy ops for inputs
    Byte requiredCMX(0);
    for (auto input : swKernelOp.inputs()) {
        auto inputType = input.getType().dyn_cast<vpux::NDTypeInterface>();
        requiredCMX += 2 * inputType.getTotalAllocSize();
    }
    if (requiredCMX > VPU::getTotalCMXSize(swKernelOp)) {
        return false;
    }

    // Strided data access is not supported, need insert extra copy ops for output
    requiredCMX = Byte(0);
    for (auto output : swKernelOp.output_buffs()) {
        auto outputType = output.getType().dyn_cast<vpux::NDTypeInterface>();
        requiredCMX += 2 * outputType.getTotalAllocSize();
    }

    return requiredCMX <= VPU::getTotalCMXSize(swKernelOp);
}

SmallVector<mlir::Value> SwKernelRewriter::createNewInputs(VPUIP::SwKernelOp, mlir::ValueRange operands,
                                                           ArrayRef<TileInfo> inTiles, bool stridedDataAccessSupport,
                                                           mlir::PatternRewriter& rewriter) const {
    SmallVector<mlir::Value> newInputs;

    for (const auto& p : operands | indexed) {
        const auto& index = p.index();
        const auto& operand = p.value();
        const auto& offset = inTiles[index].offsets;
        const auto& tiledShape = inTiles[index].shape;

        // handle swkernel's input copy
        if (stridedDataAccessSupport) {
            auto inputSubview = rewriter.create<VPUIP::SubViewOp>(operand.getLoc(), operand, offset, tiledShape);
            newInputs.push_back(inputSubview);
        } else {
            auto inputSubview = rewriter.create<VPUIP::SubViewOp>(operand.getLoc(), operand, offset, tiledShape);
            auto allocType = operand.getType().dyn_cast<vpux::NDTypeInterface>();
            auto newAllocType = allocType.changeShape(tiledShape);
            auto newInputAllocOp =
                    rewriter.create<mlir::memref::AllocOp>(operand.getLoc(), newAllocType.cast<mlir::MemRefType>());
            auto newCopyOp = rewriter.create<VPUIP::CopyOp>(operand.getLoc(), inputSubview.result(), newInputAllocOp);
            newInputs.push_back(newCopyOp);
        }
    }
    return newInputs;
}

SmallVector<mlir::Value> SwKernelRewriter::createNewOutBuffs(mlir::ValueRange operands, ArrayRef<TileInfo> inTiles,
                                                             bool stridedDataAccessSupport,
                                                             mlir::PatternRewriter& rewriter) const {
    SmallVector<mlir::Value> newOutBuffs;
    for (const auto& p : operands | indexed) {
        const auto& index = p.index();
        const auto& operand = p.value();
        const auto& tiledShape = inTiles[index].shape;
        const auto& offset = inTiles[index].offsets;
        // handle swkernel's output buf
        if (stridedDataAccessSupport) {
            auto outputSubview = rewriter.create<VPUIP::SubViewOp>(operand.getLoc(), operand, offset, tiledShape);
            newOutBuffs.push_back(outputSubview);
        } else {
            auto allocType = operand.getType().dyn_cast<vpux::NDTypeInterface>();
            auto newAllocType = allocType.changeShape(tiledShape);
            auto newAllocOp =
                    rewriter.create<mlir::memref::AllocOp>(operand.getLoc(), newAllocType.cast<mlir::MemRefType>());
            newOutBuffs.push_back(newAllocOp);
        }
    }
    return newOutBuffs;
}

void SwKernelRewriter::createNewSwKernelAndConcatOp(VPUIP::SwKernelOp swKernelOp, ArrayRef<mlir::Value> newInputs,
                                                    ArrayRef<mlir::Value> newOutBufs, const OutputTiling& outTiles,
                                                    bool stridedDataAccessSupport,
                                                    mlir::PatternRewriter& rewriter) const {
    auto newSwKernelTask = rewriter.create<VPUIP::SwKernelOp>(swKernelOp->getLoc(), newInputs, newOutBufs,
                                                              swKernelOp.kernelFunction(), swKernelOp.tileIndexAttr());
    auto swKernelRun = *swKernelOp.body().getOps<VPUIP::SwKernelRun>().begin();
    VPUIP::initSwKernel(newSwKernelTask, swKernelRun, _log);
    _log.trace("create new swKernel op {0}", newSwKernelTask);

    if (stridedDataAccessSupport) {
        auto outBufOp = mlir::dyn_cast<mlir::memref::AllocOp>(swKernelOp.output_buffs()[0].getDefiningOp());
        rewriter.replaceOpWithNewOp<VPUIP::ConcatViewOp>(swKernelOp, newSwKernelTask->getResults(), outBufOp);
    } else {
        // create concat op
        VPUX_THROW_UNLESS(outTiles.size() == newSwKernelTask.getNumResults(), "Invalid result number at {0}",
                          newSwKernelTask->getLoc());

        auto output = swKernelOp->getResult(0);
        auto outputType = output.getType().dyn_cast<vpux::NDTypeInterface>();
        rewriter.setInsertionPointAfterValue(output);
        auto outBufOp = rewriter.create<mlir::memref::AllocOp>(output.getLoc(), outputType.cast<mlir::MemRefType>());

        SmallVector<mlir::Value> results;
        for (const auto& item : outTiles | indexed) {
            const auto& outTile = item.value();
            const auto& index = item.index();
            auto outShape = to_small_vector(outTile.shape);
            auto outOffset = to_small_vector(outTile.offsets);
            auto outSubview =
                    rewriter.create<VPUIP::SubViewOp>(newSwKernelTask->getLoc(), outBufOp, outOffset, outShape);
            auto copyOp = rewriter.create<VPUIP::CopyOp>(newSwKernelTask->getLoc(), newSwKernelTask.getResult(index),
                                                         outSubview);
            results.push_back(copyOp);
        }

        rewriter.replaceOpWithNewOp<VPUIP::ConcatViewOp>(swKernelOp, results, outBufOp);
    }
}

//
// ClusterSwKernelRewriter
//

class ClusterSwKernelRewriter final : public SwKernelRewriterBase {
public:
    ClusterSwKernelRewriter(mlir::MLIRContext* ctx, int64_t shaveCout, Logger log)
            : SwKernelRewriterBase(ctx, shaveCout, log) {
        setDebugName("ClusterSwKernelRewriter");
    }

    bool checkTilePattern(VPUIP::SwKernelOp swKernelOp) const override;
    SmallVector<mlir::Value> createNewInputs(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange operands,
                                             ArrayRef<TileInfo> inTiles, bool stridedDataAccessSupport,
                                             mlir::PatternRewriter& rewriter) const override;
    SmallVector<mlir::Value> createNewOutBuffs(mlir::ValueRange operands, ArrayRef<TileInfo> inTiles,
                                               bool stridedDataAccessSupport,
                                               mlir::PatternRewriter& rewriter) const override;

    void createNewSwKernelAndConcatOp(VPUIP::SwKernelOp swKernelOp, ArrayRef<mlir::Value> newInputs,
                                      ArrayRef<mlir::Value> newOutBufs, const OutputTiling& outTiles,
                                      bool stridedDataAccessSupport,

                                      mlir::PatternRewriter& rewriter) const override;
};

bool ClusterSwKernelRewriter::checkTilePattern(VPUIP::SwKernelOp swKernelOp) const {
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    if (clusterTilingOp == nullptr) {
        return false;
    }
    auto tileDim = getSwKernelTileDim(swKernelOp);
    auto distributedType = clusterTilingOp.getResult(0).getType().dyn_cast<VPUIP::DistributedBufferType>();
    if (distributedType == nullptr) {
        return false;
    }
    auto perClusterShape = distributedType.getPerClusterComputeShapes();
    return llvm::all_of(perClusterShape, [&](auto shape) {
        return shape == perClusterShape.front() && shape[tileDim] > 1;
    });
}

SmallVector<mlir::Value> ClusterSwKernelRewriter::createNewInputs(VPUIP::SwKernelOp swKernelOp,
                                                                  mlir::ValueRange operands, ArrayRef<TileInfo> inTiles,
                                                                  bool stridedDataAccessSupport,
                                                                  mlir::PatternRewriter& rewriter) const {
    SmallVector<mlir::Value> newInputs;
    VPUX_THROW_UNLESS(operands.size() == inTiles.size(), " operand size is not equal to tile size");
    for (const auto& p : operands | indexed) {
        const auto& index = p.index();
        const auto& operand = p.value();
        const auto& offset = inTiles[index].offsets;
        const auto& tiledShape = inTiles[index].shape;

        // handle swkernel's input copy
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfterValue(operand);
        if (stridedDataAccessSupport) {
            auto inputSubview = rewriter.create<VPUIP::SubViewOp>(operand.getLoc(), operand, offset, tiledShape);
            newInputs.push_back(inputSubview);

        } else {
            // Since the compiler doesn't support copy from DistributedBufferType to DistributedBufferType, input data
            // need copy to DDR then copy back to CMX
            auto origType = swKernelOp.inputs()[index].getType().dyn_cast<vpux::NDTypeInterface>();
            auto newDDRType = origType.changeMemSpace(VPU::MemoryKind::DDR);
            auto newAllocDDROp =
                    rewriter.create<mlir::memref::AllocOp>(operand.getLoc(), newDDRType.cast<mlir::MemRefType>());
            auto newTilingCopyBackToDDROp =
                    createNewTilingCopyOp(rewriter, operand.getLoc(), newDDRType, {operand, newAllocDDROp});
            auto inputSubview = rewriter.create<VPUIP::SubViewOp>(
                    operand.getLoc(), newTilingCopyBackToDDROp->getResult(0), offset, tiledShape);
            auto newDistributedType = operand.getType().dyn_cast<vpux::NDTypeInterface>().changeShape(tiledShape);
            auto newAllocCMXOp =
                    rewriter.create<VPURT::AllocDistributed>(operand.getLoc(), newDistributedType, nullptr, nullptr);
            auto newTilingCopyToCMXOp = createNewTilingCopyOp(rewriter, operand.getLoc(), newDistributedType,
                                                              {inputSubview, newAllocCMXOp});
            newInputs.push_back(newTilingCopyToCMXOp->getResult(0));
        }
    }
    return newInputs;
}

SmallVector<mlir::Value> ClusterSwKernelRewriter::createNewOutBuffs(mlir::ValueRange operands,
                                                                    ArrayRef<TileInfo> inTiles,
                                                                    bool stridedDataAccessSupport,
                                                                    mlir::PatternRewriter& rewriter) const {
    SmallVector<mlir::Value> newOutBuffs;
    for (const auto& p : operands | indexed) {
        const auto& index = p.index();
        const auto& operand = p.value();
        const auto& tiledShape = inTiles[index].shape;
        const auto& offset = inTiles[index].offsets;
        // handle swkernel's output buf
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfterValue(operand);
        if (stridedDataAccessSupport) {
            auto outputSubview = rewriter.create<VPUIP::SubViewOp>(operand.getLoc(), operand, offset, tiledShape);
            newOutBuffs.push_back(outputSubview);
        } else {
            auto allocType = operand.getType().dyn_cast<vpux::NDTypeInterface>();
            auto newAllocType = allocType.changeShape(tiledShape);
            auto newAllocOp =
                    rewriter.create<VPURT::AllocDistributed>(operand.getLoc(), newAllocType, nullptr, nullptr);
            newOutBuffs.push_back(newAllocOp);
        }
    }
    return newOutBuffs;
}

void ClusterSwKernelRewriter::createNewSwKernelAndConcatOp(VPUIP::SwKernelOp swKernelOp,
                                                           ArrayRef<mlir::Value> newInputs,
                                                           ArrayRef<mlir::Value> newOutBufs,
                                                           const OutputTiling& outTiles, bool stridedDataAccessSupport,
                                                           mlir::PatternRewriter& rewriter) const {
    auto swKernelRun = *swKernelOp.body().getOps<VPUIP::SwKernelRun>().begin();
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    rewriter.setInsertionPointAfter(clusterTilingOp);
    auto newType = newOutBufs.front().getType().dyn_cast<VPUIP::DistributedBufferType>();
    VPUX_THROW_WHEN(newType == nullptr, "Did not find Distributed Buffer Type in new output buffers {0}", newOutBufs);
    auto mode = newType.getDistribution().mode().getValue();
    mlir::ArrayAttr strideAttr = nullptr;
    if (mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED) {
        auto shape = newType.getShape();
        auto dimOrder = newType.getDimsOrder();
        SmallVector<int64_t> strideOnPerCluster(shape.size());
        int64_t preStride = 1;
        for (int64_t idx = dimOrder.numDims() - 1; idx >= 0; idx--) {
            auto dim = dimOrder.dimAt(idx);
            strideOnPerCluster[dim.ind()] = preStride;
            preStride *= shape[dim];
        }
        strideAttr = vpux::getIntArrayAttr(rewriter.getContext(), strideOnPerCluster);
    }
    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange operands) {
        SmallVector<mlir::Value> inputs(operands.begin(), operands.begin() + newInputs.size());
        SmallVector<mlir::Value> outputs(operands.begin() + newInputs.size(), operands.end());
        auto newSwKernelTask = builder.create<VPUIP::SwKernelOp>(loc, inputs, outputs, swKernelOp.kernelFunction(),
                                                                 swKernelOp.tileIndexAttr(), strideAttr);
        VPUIP::initSwKernel(newSwKernelTask, swKernelRun, _log);
    };

    SmallVector<mlir::Value> newOperands;
    newOperands.append(newInputs.begin(), newInputs.end());
    newOperands.append(newOutBufs.begin(), newOutBufs.end());

    SmallVector<mlir::Type> resultTypes;
    for (auto outBuf : newOutBufs) {
        resultTypes.push_back(outBuf.getType());
    }
    auto newSwKernelTask =
            rewriter.create<VPUIP::NCEClusterTilingOp>(swKernelOp->getLoc(), resultTypes, newOperands, bodyBuilder);
    _log.trace("create new cluster shave {0}", newSwKernelTask);

    VPUX_THROW_UNLESS(outTiles.size() == newSwKernelTask.getNumResults(), "Invalid result number at {0}",
                      newSwKernelTask->getLoc());

    if (stridedDataAccessSupport) {
        auto outBufOp = clusterTilingOp.output_buffs()[0].getDefiningOp();
        rewriter.replaceOpWithNewOp<VPUIP::ConcatViewOp>(clusterTilingOp, newSwKernelTask->getResults(),
                                                         outBufOp->getResult(0));
        return;
    }

    // create concat op
    VPUX_THROW_UNLESS(outTiles.size() == newSwKernelTask.getNumResults(), "Invalid result number at {0}",
                      newSwKernelTask->getLoc());

    rewriter.setInsertionPointAfter(newSwKernelTask);

    auto origType = swKernelOp->getResult(0).getType().dyn_cast<vpux::NDTypeInterface>();
    auto newDDRType = origType.changeMemSpace(VPU::MemoryKind::DDR);
    auto newAllocDDROp =
            rewriter.create<mlir::memref::AllocOp>(newSwKernelTask->getLoc(), newDDRType.cast<mlir::MemRefType>());

    SmallVector<mlir::Value> results;
    for (const auto& item : outTiles | indexed) {
        const auto& outTile = item.value();
        const auto& index = item.index();
        auto outShape = to_small_vector(outTile.shape);
        auto outOffset = to_small_vector(outTile.offsets);
        auto outSubview =
                rewriter.create<VPUIP::SubViewOp>(newSwKernelTask->getLoc(), newAllocDDROp, outOffset, outShape);
        auto copyOp = createNewTilingCopyOp(rewriter, newSwKernelTask->getLoc(), outSubview.getType(),
                                            {newSwKernelTask.getResult(index), outSubview});
        results.push_back(copyOp->getResult(0));
    }
    auto concatOp = rewriter.create<VPUIP::ConcatViewOp>(newSwKernelTask->getLoc(), results, newAllocDDROp);

    auto outType = clusterTilingOp->getResult(0).getType().dyn_cast<vpux::NDTypeInterface>();
    auto newAllocCMXOp = rewriter.create<VPURT::AllocDistributed>(clusterTilingOp->getLoc(), outType, nullptr, nullptr);
    auto newTilingCopyToCMXOp =
            createNewTilingCopyOp(rewriter, newSwKernelTask->getLoc(), outType, {concatOp, newAllocCMXOp});
    rewriter.replaceOp(clusterTilingOp, newTilingCopyToCMXOp->getResults());
}

//
// TileActShaveKernelTaskPass
//

class TileActShaveKernelTaskPass final : public VPUIP::TileActShaveKernelTaskBase<TileActShaveKernelTaskPass> {
public:
    explicit TileActShaveKernelTaskPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void TileActShaveKernelTaskPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    // NOTE: the pass is enabled only for VPUX37XX for now
    if (arch != VPU::ArchKind::VPUX37XX) {
        return;
    }

    auto shaveActCount = IE::getAvailableExecutor(module, VPU::ExecutorKind::SHAVE_ACT).count();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwKernelRewriter>(&ctx, shaveActCount, _log);
    patterns.add<ClusterSwKernelRewriter>(&ctx, shaveActCount, _log);
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createTileActShaveKernelTaskPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createTileActShaveKernelTaskPass(Logger log) {
    return std::make_unique<TileActShaveKernelTaskPass>(log);
}
