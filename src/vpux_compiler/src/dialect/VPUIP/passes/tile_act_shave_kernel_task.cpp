//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;
using namespace VPUIP;

namespace {

SmallVector<mlir::Attribute> kernelArgsRange(VPUIP::SwKernelOp swKernelOp) {
    SmallVector<mlir::Attribute> attrStorage;

    for (auto&& kernelRun : swKernelOp.body().getOps<VPUIP::SwKernelRun>()) {
        if (kernelRun.attrs().has_value()) {
            const mlir::ArrayAttr arrayAttrs = kernelRun.attrs().value();
            const auto& attrs = arrayAttrs.getValue();
            for (const auto& attr : attrs) {
                attrStorage.push_back(attr);
            }
        }
    }
    return attrStorage;
}

Dim convertKernelAxisToDim(mlir::Value tensorArg, int64_t kernelAxis) {
    const auto inOrder = DimsOrder::fromValue(tensorArg);

    const auto shape = getShape(tensorArg);
    auto nDims = checked_cast<uint32_t>(shape.size());

    auto pos = nDims - 1 - kernelAxis;

    return inOrder.dimAt(pos);
}

bool isSoftmax(VPUIP::SwKernelOp swKernelOp) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    return kernelEntryName == "singleShaveSoftmax";
}

bool isSoftmaxAxis(VPUIP::SwKernelOp swKernelOp, Dim axis) {
    if (!isSoftmax(swKernelOp)) {
        return false;
    }

    auto taskArgs = kernelArgsRange(swKernelOp);
    const auto kernelAxis = taskArgs[0].dyn_cast<mlir::IntegerAttr>().getInt();

    auto softmaxAxis = convertKernelAxisToDim(swKernelOp.getResult(0), kernelAxis);

    if (softmaxAxis == axis) {
        return true;
    }

    return false;
}

bool isTopKAxis(VPUIP::SwKernelOp swKernelOp, Dim axis) {
    auto taskArgs = kernelArgsRange(swKernelOp);
    const auto kernelAxis = taskArgs.front().cast<mlir::IntegerAttr>().getInt();
    auto topKAxis = convertKernelAxisToDim(swKernelOp.getResult(0), kernelAxis);

    return topKAxis == axis;
}

Dim getHighestDim(VPUIP::SwKernelOp swKernelOp) {
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

Dim getHighestDimOfSoftmax(VPUIP::SwKernelOp swKernelOp) {
    const auto output = swKernelOp->getResult(0);
    const auto outputType = output.getType().cast<vpux::NDTypeInterface>();
    const auto outOrder = outputType.getDimsOrder();
    const auto outShape = outputType.getShape();

    auto taskArgs = kernelArgsRange(swKernelOp);
    const auto kernelAxis = taskArgs.front().cast<mlir::IntegerAttr>().getInt();
    auto softmaxAxis = convertKernelAxisToDim(swKernelOp.getResult(0), kernelAxis);

    for (auto i : irange(outOrder.numDims())) {
        auto dim = outOrder.dimAt(i);
        if (outShape[dim] > 1 && dim != softmaxAxis) {
            return dim;
        }
    }
    return outOrder.dimAt(0);
}

bool doesSwKernelSupportTiling(VPUIP::SwKernelOp swKernelOp, vpux::Logger log) {
    auto isAllOutputShapeEqual = llvm::all_of(swKernelOp.getOutputs(), [&](auto output) {
        return getShape(output) == getShape(*swKernelOp.getOutputs().begin());
    });

    if (swKernelOp.getOutputs().size() > 2 || !isAllOutputShapeEqual) {
        log.trace("SW kernel op has outputs with different shapes at '{0}'", swKernelOp->getLoc());
        return false;
    }

    if (!isSwKernelTilingSupported(swKernelOp)) {
        return false;
    }

    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (kernelEntryName == "singleShaveMVN") {
        auto taskArgs = kernelArgsRange(swKernelOp);
        const auto acrossChannels = taskArgs[0].dyn_cast<mlir::BoolAttr>();
        return !acrossChannels.getValue();
    } else if (kernelEntryName == "singleShaveInterpolate") {
        auto taskArgs = kernelArgsRange(swKernelOp);
        // E#67003, note that currenly only enable multi cluster when mode is linear_onnx and coord mode is half pixel
        const auto mode = static_cast<IE::InterpolateMode>(taskArgs[0].dyn_cast<mlir::IntegerAttr>().getInt());
        const auto coordMode =
                static_cast<IE::InterpolateCoordMode>(taskArgs[1].dyn_cast<mlir::IntegerAttr>().getInt());
        return mode == IE::InterpolateMode::LINEAR_ONNX && (coordMode == IE::InterpolateCoordMode::HALF_PIXEL ||
                                                            coordMode == IE::InterpolateCoordMode::ALIGN_CORNERS ||
                                                            coordMode == IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL);
    } else if (kernelEntryName == "singleShaveSoftmax") {
        auto highestDim = getHighestDimOfSoftmax(swKernelOp);
        if (isSoftmaxAxis(swKernelOp, highestDim)) {
            return false;
        }
    } else if (kernelEntryName == "single_shave_convert") {
        // E#83794 Case with aligned tiling not supported
        // Offsets for the inputs and outputs needs to be adjusted based on aligned shapes
        if (auto clusterTilingOp = swKernelOp->getParentOfType<VPUIP::NCEClusterTilingOp>()) {
            auto ndType = clusterTilingOp->getOperand(0).getType().cast<VPUIP::DistributedBufferType>();
            if (ndType != nullptr && ndType.getDistribution().getAlignment() != nullptr) {
                return false;
            }
        }
    } else if (kernelEntryName == "single_shave_topk") {
        auto highestDim = getHighestDim(swKernelOp);
        if (isTopKAxis(swKernelOp, highestDim)) {
            return false;
        }
    } else if (kernelEntryName == "single_shave_gather") {
        const auto outputShape = getShape(swKernelOp.getResult(0));
        const auto nonTrivialDimPredicate = [](const int64_t dim) -> bool {
            return dim > 1;
        };

        const auto nonTrivialInputDims =
                std::count_if(outputShape.raw().begin(), outputShape.raw().end(), nonTrivialDimPredicate);

        if (nonTrivialInputDims == 1) {
            return false;
        }
    }

    return true;
}

Dim getSwKernelTileDim(VPUIP::SwKernelOp swKernelOp) {
    if (auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp())) {
        // if swKernelOp has parent op of NCEClusterTilingOp, the tiling dim need to be aligned with the distributed
        // buffer
        auto outType = clusterTilingOp.getResult(0).getType();
        auto dimIdx = VPUIP::getTilingDimIndex(outType);
        if (dimIdx.has_value()) {
            return Dim(dimIdx.value());
        }
    }
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (kernelEntryName == "singleShaveMVN") {
        // MVN only supports tiling on C
        return Dims4D::Act::C;
    } else if (kernelEntryName == "singleShaveInterpolate") {
        return Dims4D::Act::H;
    } else if (kernelEntryName == "singleShaveSoftmax") {
        return getHighestDimOfSoftmax(swKernelOp);
    } else {
        // get highest dim by default
        return getHighestDim(swKernelOp);
    }
}

mlir::FailureOr<OutputTiling> getSwKernelOutputTiling(VPUIP::SwKernelOp swKernelOp, ShapeRef outputShape,
                                                      int64_t maxNumTiles, vpux::Logger log) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    // Gather op's output always is non-4D and Gather's backInfer has it's own logic later, skip the check here.
    if (kernelEntryName != "single_shave_gather") {
        VPUX_THROW_UNLESS(outputShape.size() == 4, "Unsupported operation '{0}' at '{1}', it has non 4D result",
                          swKernelOp->getName(), swKernelOp->getLoc());
    }

    Shape nTilesOnDim(outputShape.size(), 1);
    const auto tileDim = getSwKernelTileDim(swKernelOp);
    log.trace("Tile Dim is {0}", tileDim);
    nTilesOnDim[tileDim] = std::min(maxNumTiles, outputShape[tileDim]);
    return fillDividedTiles(nTilesOnDim, outputShape, None);
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

bool checkSwKernelTilingAlignment(VPUIP::SwKernelOp swKernelOp, size_t nTiles, const vpux::NDTypeInterface valueType,
                                  const std::function<mlir::Value(VPUIP::NCEClusterTilingOp)>& getParentValue,
                                  vpux::Logger log) {
    auto clusterOp = swKernelOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
    if (clusterOp == nullptr) {
        return true;
    }

    auto parentValue = getParentValue(clusterOp);
    auto parentValueType = parentValue.getType().dyn_cast<VPUIP::DistributedBufferType>();
    VPUX_THROW_UNLESS(parentValueType != nullptr, "Operand must have distributed type. Got: {0}", parentValueType);
    auto distribution = parentValueType.getDistribution();
    auto alignAttr = distribution.getAlignment();
    if (alignAttr == nullptr) {
        return true;
    }
    auto numClustersAttr = distribution.getNumClusters();
    VPUX_THROW_UNLESS(numClustersAttr != nullptr, "Distribution must have num_cluster() attribute. {0}", distribution);

    const auto alignmentPerTile = parseIntArrayAttr<int64_t>(alignAttr);
    const auto tileDim = getSwKernelTileDim(swKernelOp);
    if (alignmentPerTile[tileDim.ind()] == 1) {
        return true;
    }

    const auto valueShape = valueType.getShape();
    int64_t totalAlignment = 0;
    if (distribution.getMode().getValue() == VPU::DistributionMode::DUPLICATED) {
        totalAlignment = alignmentPerTile[tileDim.ind()] * static_cast<int64_t>(nTiles);
    } else {
        totalAlignment = alignmentPerTile[tileDim.ind()] * numClustersAttr.getInt() * static_cast<int64_t>(nTiles);
    }
    if (valueShape[tileDim] % totalAlignment) {
        log.info("Skip tiling for swKernelOp {0}, shape is not aliged. Shape '{1}', distribution '{2}'",
                 swKernelOp->getLoc(), valueShape, distribution);
        return false;
    }

    return true;
}

// Output tiles for each shave on all clusters
using OutShaveTiles = SmallVector<OutputTiling>;
// Input tiles for each shave on all clusters
using InShaveTiles = SmallVector<SmallVector<InputTiling>, 1>;

//
// SwKernelRewriterBase
//

class SwKernelRewriterBase : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    SwKernelRewriterBase(mlir::MLIRContext* ctx, int64_t shaveCount, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _shaveCount(shaveCount), _log(log) {
        setDebugName("SwKernelRewriterBase");
    }
    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const override;
    virtual bool checkTilePattern(VPUIP::SwKernelOp swKernelOp, bool insertSubview) const = 0;
    virtual bool needInsertSubviewOnly(VPUIP::SwKernelOp swKernelOp) const;
    virtual Optional<OutShaveTiles> calculateOutputTiles(VPUIP::SwKernelOp swKernelOp) const = 0;
    virtual Optional<InShaveTiles> calculateInputTiles(VPUIP::SwKernelOp swKernelOp) const = 0;

    virtual SmallVector<mlir::Value> createNewInputs(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange operands,
                                                     bool insertSubview, int64_t outTileIndex,
                                                     mlir::PatternRewriter& rewriter) const = 0;
    virtual SmallVector<mlir::Value> createNewOutBuffs(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange operands,
                                                       bool insertSubview, int64_t outTileIndex,
                                                       mlir::PatternRewriter& rewriter) const = 0;
    virtual VPUIP::SwKernelOp createNewSwKernelOp(VPUIP::SwKernelOp swKernelOp, ArrayRef<mlir::Value> newInputs,
                                                  ArrayRef<mlir::Value> newOutBufs, bool insertSubview,
                                                  mlir::PatternRewriter& rewriter) const = 0;
    virtual void replaceOpWithConcatView(VPUIP::SwKernelOp origOp, VPUIP::SwKernelOp newSwkernelOp, bool insertSubview,
                                         mlir::PatternRewriter& rewriter) const = 0;
    virtual OutputTiling getOuterMostOutputTiling(VPUIP::SwKernelOp swKernelOp) const = 0;
    virtual InputTiling getOuterMostInputTiling(VPUIP::SwKernelOp swKernelOp, int64_t outTileIndx) const = 0;
    virtual SmallVector<mlir::Attribute> updateSwKernelAttrs(VPUIP::SwKernelOp swKernelOp,
                                                             int64_t outTileIndexInsideCluster) const;

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

    // check output tiles on all shaves
    auto outTiles = calculateOutputTiles(swKernelOp);
    if (!outTiles.has_value()) {
        return mlir::failure();
    }

    // check input tiles on all shaves
    auto inTiles = calculateInputTiles(swKernelOp);
    if (!inTiles.has_value()) {
        return mlir::failure();
    }

    auto insertSubview = needInsertSubviewOnly(swKernelOp);

    if (!checkTilePattern(swKernelOp, insertSubview)) {
        return mlir::failure();
    }

    _log.trace("process swKernelOp at {0}", swKernelOp->getLoc());

    SmallVector<mlir::Value> newInputs;
    SmallVector<mlir::Value> newOutBuffs;

    // Get tile num on each cluster, since all clusters have same tile nums, so we only need get it from the first
    // cluster
    auto tileSize = outTiles.value().front().size();
    SmallVector<SmallVector<mlir::Attribute>> newAttrs;
    for (auto tileIndex : irange(tileSize)) {
        auto inputs = getOuterMappingOperand(swKernelOp, swKernelOp.inputs());
        auto outBuffs = getOuterMappingOperand(swKernelOp, swKernelOp.output_buffs());

        newInputs.append(createNewInputs(swKernelOp, inputs, insertSubview, tileIndex, rewriter));
        newOutBuffs.append(createNewOutBuffs(swKernelOp, outBuffs, insertSubview, tileIndex, rewriter));
        newAttrs.push_back(updateSwKernelAttrs(swKernelOp, tileIndex));
    }

    auto newSwKernelOp = createNewSwKernelOp(swKernelOp, newInputs, newOutBuffs, insertSubview, rewriter);
    replaceOpWithConcatView(swKernelOp, newSwKernelOp, insertSubview, rewriter);
    auto newSwKernelRuns = newSwKernelOp.body().getOps<VPUIP::SwKernelRun>();
    auto newSwKernelRunIter = newSwKernelRuns.begin();
    for (auto idx : irange(tileSize)) {
        VPUX_THROW_WHEN(newSwKernelRunIter == newSwKernelRuns.end(), "Cannot get SwKernelRun Op for output tile {0} ",
                        idx);
        auto newSwKernelRun = *newSwKernelRunIter;
        newSwKernelRun.attrsAttr(mlir::ArrayAttr::get(newSwKernelOp->getContext(), newAttrs[idx]));
        newSwKernelRunIter++;
    }
    return mlir::success();
}

bool SwKernelRewriterBase::needInsertSubviewOnly(VPUIP::SwKernelOp swKernelOp) const {
    const auto tileDim = getSwKernelTileDim(swKernelOp);
    if (tileDim == getHighestDim(swKernelOp)) {
        return true;
    }
    // If swkernel doesn't support strided data access, the tiling input has to be created by subview and copy to make
    // sure the new input is continuous
    return isStridedDataAccessSupported(swKernelOp);
}

SmallVector<mlir::Attribute> SwKernelRewriterBase::updateSwKernelAttrs(VPUIP::SwKernelOp swKernelOp,
                                                                       int64_t outTileIndexInsideCluster) const {
    auto swKernelRun = *swKernelOp.body().getOps<VPUIP::SwKernelRun>().begin();
    if (!swKernelRun.attrs().has_value()) {
        return {};
    }

    const auto outTiles = getOuterMostOutputTiling(swKernelOp);
    const auto inputTiles = getOuterMostInputTiling(swKernelOp, outTileIndexInsideCluster);
    auto origAttr = swKernelRun.attrs().value();
    SmallVector<mlir::Attribute> attrs(origAttr.begin(), origAttr.end());
    return VPUIP::getSwkernelNewAttrsAfterTiling(swKernelOp, attrs, inputTiles, outTiles[outTileIndexInsideCluster],
                                                 _log);
}

//
// SwKernelRewriter
//

class SwKernelRewriter final : public SwKernelRewriterBase {
public:
    SwKernelRewriter(mlir::MLIRContext* ctx, int64_t shaveCout, Logger log): SwKernelRewriterBase(ctx, shaveCout, log) {
        setDebugName("SwKernelRewriter");
    }

    bool checkTilePattern(VPUIP::SwKernelOp swKernelOp, bool insertSubview) const override;
    Optional<OutShaveTiles> calculateOutputTiles(VPUIP::SwKernelOp swKernelOp) const override;
    Optional<InShaveTiles> calculateInputTiles(VPUIP::SwKernelOp swKernelOp) const override;
    SmallVector<mlir::Value> createNewInputs(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange operands,
                                             bool insertSubview, int64_t outTileIndex,
                                             mlir::PatternRewriter& rewriter) const override;
    SmallVector<mlir::Value> createNewOutBuffs(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange operands,
                                               bool insertSubview, int64_t outTileIndex,
                                               mlir::PatternRewriter& rewriter) const override;

    VPUIP::SwKernelOp createNewSwKernelOp(VPUIP::SwKernelOp swKernelOp, ArrayRef<mlir::Value> newInputs,
                                          ArrayRef<mlir::Value> newOutBufs, bool insertSubview,
                                          mlir::PatternRewriter& rewriter) const override;
    void replaceOpWithConcatView(VPUIP::SwKernelOp origOp, VPUIP::SwKernelOp newSwkernelOp, bool insertSubview,
                                 mlir::PatternRewriter& rewriter) const override;

    OutputTiling getOuterMostOutputTiling(VPUIP::SwKernelOp swKernelOp) const override;
    InputTiling getOuterMostInputTiling(VPUIP::SwKernelOp swKernelOp, int64_t outTileIndx) const override;
};

bool SwKernelRewriter::checkTilePattern(VPUIP::SwKernelOp swKernelOp, bool insertSubview) const {
    if (mlir::isa<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp())) {
        return false;
    }
    if (insertSubview) {
        return true;
    }

    // Strided data access is not supported, will try to insert extra copy ops for inputs and output buf. So
    // need to check the cmx requirement for:
    // 1. the new input tile copy(CMX2CMX) ops
    // 2. the new output tile copy(CMX2CMX) ops
    // 3. the new swkernel op
    auto getNewTiledAllocSize = [](mlir::Value origOperand, ShapeRef newTiledShape) {
        auto origType = origOperand.getType().dyn_cast<vpux::NDTypeInterface>();
        auto newTiledType = origType.changeShape(newTiledShape);
        return newTiledType.getTotalAllocSize();
    };

    auto totalCMXSize = VPU::getTotalCMXSize(swKernelOp);
    auto inputs = getOuterMappingOperand(swKernelOp, swKernelOp.inputs());
    auto outTiles = getOuterMostOutputTiling(swKernelOp);
    Byte requiredCMXForTiledSwKernelOp(0);
    for (auto outIndex : irange(outTiles.size())) {
        const auto inTiles = getOuterMostInputTiling(swKernelOp, outIndex);
        for (const auto& item : inputs | indexed) {
            auto input = item.value();
            auto index = item.index();
            auto newInputRequiredSize = getNewTiledAllocSize(input, inTiles.tiles[index].shape);
            Byte requiredCMXForInputCopy = newInputRequiredSize * 2;
            // check cmx requirement for each input tile copy
            if (requiredCMXForInputCopy > totalCMXSize) {
                return false;
            }
            requiredCMXForTiledSwKernelOp += newInputRequiredSize;
        }
        auto newOutputRequiredSize = getNewTiledAllocSize(swKernelOp.getResult(0), outTiles[outIndex].shape);
        // check cmx requirement for each output tile copy
        Byte requiredCMXForOutputCopy = newOutputRequiredSize * 2;
        if (requiredCMXForOutputCopy > totalCMXSize) {
            return false;
        }
        requiredCMXForTiledSwKernelOp += newOutputRequiredSize;
    }

    return requiredCMXForTiledSwKernelOp <= totalCMXSize;
}

Optional<OutShaveTiles> SwKernelRewriter::calculateOutputTiles(VPUIP::SwKernelOp swKernelOp) const {
    auto tiles = getSwKernelOutputTiling(swKernelOp, getShape(swKernelOp.getResult(0)), _shaveCount, _log);
    if (mlir::failed(tiles)) {
        return Optional<OutShaveTiles>{None};
    }

    auto outTiles = tiles.value();
    return outTiles.size() == 1 ? Optional<OutShaveTiles>{None} : OutShaveTiles{outTiles};
}

Optional<InShaveTiles> SwKernelRewriter::calculateInputTiles(VPUIP::SwKernelOp swKernelOp) const {
    auto outTilesOnAllClusters = calculateOutputTiles(swKernelOp);
    if (!outTilesOnAllClusters.has_value()) {
        return None;
    }
    SmallVector<InputTiling> inTiles;
    auto outTilesValues = outTilesOnAllClusters.value();
    for (const auto& outTile : outTilesValues.front()) {
        inTiles.push_back(VPUIP::backInferSwKernelInputTile(swKernelOp, outTile, _log));
    }
    return InShaveTiles{inTiles};
}

SmallVector<mlir::Value> SwKernelRewriter::createNewInputs(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange operands,
                                                           bool insertSubview, int64_t outTileIndex,
                                                           mlir::PatternRewriter& rewriter) const {
    const auto inShaveTiles = calculateInputTiles(swKernelOp).value().front();
    const auto& inTiles = inShaveTiles[outTileIndex];
    SmallVector<mlir::Value> newInputs;
    for (const auto& p : operands | indexed) {
        const auto& index = p.index();
        const auto& operand = p.value();
        const auto& offset = inTiles.tiles[index].offsets;
        const auto& tiledShape = inTiles.tiles[index].shape;

        // handle swkernel's input copy
        if (insertSubview) {
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

SmallVector<mlir::Value> SwKernelRewriter::createNewOutBuffs(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange operands,
                                                             bool insertSubview, int64_t outTileIndex,
                                                             mlir::PatternRewriter& rewriter) const {
    const auto outTiles = calculateOutputTiles(swKernelOp).value().front();
    const auto& tiledShape = outTiles[outTileIndex].shape;
    const auto& offset = outTiles[outTileIndex].offsets;

    SmallVector<mlir::Value> newOutputs;
    for (auto operand : operands) {
        if (insertSubview) {
            auto outputSubview = rewriter.create<VPUIP::SubViewOp>(operand.getLoc(), operand, offset, tiledShape);
            newOutputs.push_back(outputSubview);
        } else {
            auto allocType = operand.getType().cast<vpux::NDTypeInterface>();
            auto newAllocType = allocType.changeShape(tiledShape);
            auto newOutputAllocOp =
                    rewriter.create<mlir::memref::AllocOp>(operand.getLoc(), newAllocType.cast<mlir::MemRefType>());
            newOutputs.push_back(newOutputAllocOp);
        }
    }
    return newOutputs;
}

VPUIP::SwKernelOp SwKernelRewriter::createNewSwKernelOp(VPUIP::SwKernelOp swKernelOp, ArrayRef<mlir::Value> newInputs,
                                                        ArrayRef<mlir::Value> newOutBufs, bool,
                                                        mlir::PatternRewriter& rewriter) const {
    auto newSwKernelTask = rewriter.create<VPUIP::SwKernelOp>(swKernelOp->getLoc(), newInputs, newOutBufs,
                                                              swKernelOp.kernelFunction(), swKernelOp.tileIndexAttr());
    auto swKernelRun = *swKernelOp.body().getOps<VPUIP::SwKernelRun>().begin();
    VPUIP::initSwKernel(newSwKernelTask, swKernelRun, _log);
    _log.trace("create new swKernel op {0}", newSwKernelTask);
    return newSwKernelTask;
}

void SwKernelRewriter::replaceOpWithConcatView(VPUIP::SwKernelOp origOp, VPUIP::SwKernelOp newSwKernelOp,
                                               bool insertSubview, mlir::PatternRewriter& rewriter) const {
    auto origOutBufOp = mlir::dyn_cast<mlir::memref::AllocOp>(origOp.output_buffs()[0].getDefiningOp());
    if (insertSubview) {
        rewriter.replaceOpWithNewOp<VPUIP::ConcatViewOp>(origOp, newSwKernelOp->getResults(), origOutBufOp);
        return;
    }

    const auto outTiles = calculateOutputTiles(origOp).value().front();
    // create concat op
    VPUX_THROW_UNLESS(outTiles.size() == newSwKernelOp.getNumResults(), "Invalid result number at {0}",
                      newSwKernelOp->getLoc());

    auto output = origOp->getResult(0);
    auto outputType = output.getType().dyn_cast<vpux::NDTypeInterface>();
    rewriter.setInsertionPointAfterValue(output);
    auto outBufOp = rewriter.create<mlir::memref::AllocOp>(output.getLoc(), outputType.cast<mlir::MemRefType>());

    SmallVector<mlir::Value> results;
    for (const auto& item : outTiles | indexed) {
        const auto& outTile = item.value();
        const auto& index = item.index();
        auto outShape = to_small_vector(outTile.shape);
        auto outOffset = to_small_vector(outTile.offsets);
        auto outSubview = rewriter.create<VPUIP::SubViewOp>(newSwKernelOp->getLoc(), outBufOp, outOffset, outShape);
        auto copyOp =
                rewriter.create<VPUIP::CopyOp>(newSwKernelOp->getLoc(), newSwKernelOp.getResult(index), outSubview);
        results.push_back(copyOp);
    }

    rewriter.replaceOpWithNewOp<VPUIP::ConcatViewOp>(origOp, results, outBufOp);
    if (origOutBufOp->use_empty()) {
        rewriter.eraseOp(origOutBufOp);
    }
    return;
}

OutputTiling SwKernelRewriter::getOuterMostOutputTiling(VPUIP::SwKernelOp swKernelOp) const {
    return calculateOutputTiles(swKernelOp).value().front();
}

InputTiling SwKernelRewriter::getOuterMostInputTiling(VPUIP::SwKernelOp swKernelOp, int64_t outTileIndx) const {
    const auto inTiles = calculateInputTiles(swKernelOp).value().front();
    return inTiles[outTileIndx];
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

    bool checkTilePattern(VPUIP::SwKernelOp swKernelOp, bool insertSubview) const override;
    bool needInsertSubviewOnly(VPUIP::SwKernelOp swKernelOp) const override;
    Optional<OutShaveTiles> calculateOutputTiles(VPUIP::SwKernelOp swKernelOp) const override;
    Optional<InShaveTiles> calculateInputTiles(VPUIP::SwKernelOp swKernelOp) const override;
    SmallVector<mlir::Value> createNewInputs(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange operands,
                                             bool insertSubview, int64_t outTileIndex,
                                             mlir::PatternRewriter& rewriter) const override;
    SmallVector<mlir::Value> createNewOutBuffs(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange operands,
                                               bool insertSubview, int64_t outTileIndex,
                                               mlir::PatternRewriter& rewriter) const override;
    VPUIP::SwKernelOp createNewSwKernelOp(VPUIP::SwKernelOp swKernelOp, ArrayRef<mlir::Value> newInputs,
                                          ArrayRef<mlir::Value> newOutBufs, bool insertSubview,
                                          mlir::PatternRewriter& rewriter) const override;
    void replaceOpWithConcatView(VPUIP::SwKernelOp origOp, VPUIP::SwKernelOp newSwKernelOp, bool insertSubview,
                                 mlir::PatternRewriter& rewriter) const override;
    OutputTiling getOuterMostOutputTiling(VPUIP::SwKernelOp swKernelOp) const override;
    InputTiling getOuterMostInputTiling(VPUIP::SwKernelOp swKernelOp, int64_t outTileIndx) const override;

private:
    bool onlyHasCopyOpUser(VPUIP::SwKernelOp swKernelOp) const;
    vpux::NDTypeInterface getNewTiledDistributedType(VPUIP::SwKernelOp swKernelOp, mlir::Value outerOperand,
                                                     int64_t operandIdx, int64_t outTileIndex,
                                                     ShapeRef tiledShape) const;

    mlir::ArrayAttr getStrideOnEachCluster(VPUIP::SwKernelOp swKernelOp, bool insertSubview) const;
};

bool ClusterSwKernelRewriter::checkTilePattern(VPUIP::SwKernelOp swKernelOp, bool insertSubview) const {
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    if (clusterTilingOp == nullptr) {
        return false;
    }
    auto outShaveTiles = calculateOutputTiles(swKernelOp).value();
    auto tileSizeOnEachCluster = outShaveTiles.size();

    const auto getParentInput = [](VPUIP::NCEClusterTilingOp parent) -> mlir::Value {
        return parent->getOperand(0);
    };
    const auto getParentOutput = [](VPUIP::NCEClusterTilingOp parent) -> mlir::Value {
        return *parent.getOutputs().begin();
    };
    const auto inputType = swKernelOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputType = swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    if (!checkSwKernelTilingAlignment(swKernelOp, tileSizeOnEachCluster, outputType, getParentOutput, _log) ||
        !checkSwKernelTilingAlignment(swKernelOp, tileSizeOnEachCluster, inputType, getParentInput, _log)) {
        return false;
    }

    auto distributedType = clusterTilingOp.getResult(0).getType().dyn_cast<VPUIP::DistributedBufferType>();
    if (distributedType == nullptr) {
        return false;
    }
    auto tileDim = getSwKernelTileDim(swKernelOp);
    auto perClusterShapes = distributedType.getPerClusterComputeShapes();
    auto tileOnAllClusters = llvm::all_of(perClusterShapes, [&](const auto& shape) {
        return shape[tileDim] > 1;
    });
    if (!tileOnAllClusters) {
        return false;
    }

    if (insertSubview) {
        return true;
    }

    // Calculate requried cmx size since the input cmx size may be changed due to overlapped input tiles like
    // Interpolate
    Byte requiredCMX = distributedType.getTotalAllocSize();
    const auto outTiles = getOuterMostOutputTiling(swKernelOp);
    auto inputs = getOuterMappingOperand(swKernelOp, swKernelOp.inputs());
    for (auto outIndex : irange(outTiles.size())) {
        const auto inTiles = getOuterMostInputTiling(swKernelOp, outIndex);
        for (const auto& item : inputs | indexed) {
            auto input = item.value();
            auto index = item.index();
            auto tiledShape = inTiles.tiles[index].shape;
            auto newTiledInputDistributedType =
                    getNewTiledDistributedType(swKernelOp, input, index, outIndex, tiledShape);
            requiredCMX += newTiledInputDistributedType.getTotalAllocSize();
        }
    }
    return requiredCMX <= VPU::getTotalCMXSize(swKernelOp);
}

bool ClusterSwKernelRewriter::needInsertSubviewOnly(VPUIP::SwKernelOp swKernelOp) const {
    // For swkernel op with different input and output shapes like interpolate, the per cluster compute offset
    // are directly set which may have different values on all clusters. And the related subview might not be converted
    // to correct declare buffers without copy op inserted
    if (getShape(swKernelOp.getInputs()[0]) != getShape(swKernelOp.getResult(0))) {
        return false;
    }
    return SwKernelRewriterBase::needInsertSubviewOnly(swKernelOp);
}

Optional<OutShaveTiles> ClusterSwKernelRewriter::calculateOutputTiles(VPUIP::SwKernelOp swKernelOp) const {
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    if (clusterTilingOp == nullptr) {
        return None;
    }
    auto distributedType = clusterTilingOp.getResult(0).getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto perClusterShapes = distributedType.getPerClusterComputeShapes();

    // Get output tiles on each cluster
    SmallVector<mlir::FailureOr<OutputTiling>> tiles;
    std::transform(perClusterShapes.begin(), perClusterShapes.end(), std::back_inserter(tiles), [&](const auto& shape) {
        return getSwKernelOutputTiling(swKernelOp, shape, _shaveCount, _log);
    });
    if (tiles.empty()) {
        return None;
    }

    auto hasInvalidTiles = llvm::any_of(tiles, [&](const auto& tile) {
        return mlir::failed(tile);
    });
    if (hasInvalidTiles) {
        return None;
    }

    OutShaveTiles outTiles;
    for (auto& tile : tiles) {
        outTiles.push_back(tile.value());
    }

    // For each cluster, the output tile size should be equal and greater than one
    int64_t tileSize = outTiles[0].size();
    auto findNoSuitableTileSizeOnClusters = llvm::any_of(outTiles, [&](const auto& tile) {
        return tile.size() != static_cast<size_t>(tileSize) || tile.size() <= 1;
    });
    if (findNoSuitableTileSizeOnClusters) {
        return None;
    }
    auto allClustersHaveSameShape = llvm::all_of(perClusterShapes, [&perClusterShapes](auto shape) {
        return shape == perClusterShapes.front();
    });

    if (!allClustersHaveSameShape) {
        // Need to adjust the tiling size due to aligment requriement, otherwise the compiler can not get required
        // distributed buffer by subview since the offsets on each cluster are not same.
        // For example shape [1, 33, 1, 1] tiled on C. So the tiled shape could be
        // cluster 0 [1, 9, 1, 1], [1, 8, 1, 1]
        // cluster 1 [1, 8, 1, 1], [1, 8, 1, 1]
        // we can't represent the second distributed buffer {cluster 0[1, 8, 1, 1], cluster 1[1, 8, 1, 1]} since the
        // offset on each cluster are different(cluster 0 offset = 9, cluster 1 offset = 8). So we need adjust the tile
        // size to make sure the offsets are equal for each cluster. The logic is to find the smallest tile value and
        // change all the tiles' value equal to it except the last one.
        // In this case, the tiles are changed to
        // cluster 0 [1, 8, 1, 1], [1, 9, 1, 1]
        // cluster 1 [1, 8, 1, 1], [1, 8, 1, 1]
        const auto tileDim = getSwKernelTileDim(swKernelOp);
        auto iter = std::min_element(outTiles.begin(), outTiles.end(), [&](const auto& a, const auto& b) {
            return a.front().shape[tileDim] <= b.front().shape[tileDim];
        });
        VPUX_THROW_WHEN(iter == outTiles.end(), "Can not find min tile value for '{0}'", swKernelOp);
        auto minTileVal = iter->front().shape[tileDim];

        SmallVector<OutputTiling> adjustedOutTiles;
        // Adjust the front tiles with same tile value
        for (auto& item : outTiles | indexed) {
            const auto& clusterId = item.index();
            auto& outTilePerCluster = item.value();
            int64_t offsetVal = 0;
            for (auto i : irange(tileSize - 1)) {
                Shape outShape(outTilePerCluster.front().shape);
                Shape offset(outShape.size(), 0);
                outShape[tileDim] = minTileVal;
                offset[tileDim] = offsetVal;
                offsetVal = offsetVal + outShape[tileDim];
                outTilePerCluster[i] = TileInfo(outShape, offset, outTilePerCluster.front().axis);
            }
            // Recalculate the last tile value
            Shape lastTileShape(outTilePerCluster.front().shape);
            lastTileShape[tileDim] = perClusterShapes[clusterId][tileDim] - (tileSize - 1) * minTileVal;
            Shape lastTileOffset(lastTileShape.size(), 0);
            lastTileOffset[tileDim] = offsetVal;
            outTilePerCluster[tileSize - 1] = TileInfo(lastTileShape, lastTileOffset, outTilePerCluster.front().axis);
        }
    }
    return outTiles;
}

Optional<InShaveTiles> ClusterSwKernelRewriter::calculateInputTiles(VPUIP::SwKernelOp swKernelOp) const {
    const auto outTilesOnAllClusters = calculateOutputTiles(swKernelOp);
    if (!outTilesOnAllClusters.has_value()) {
        return None;
    }
    const auto outTiles = outTilesOnAllClusters.value();

    InShaveTiles inTiles;
    for (auto clusterId : irange(outTiles.size())) {
        SmallVector<InputTiling> inTilesPerCluster;
        for (const auto& outTile : outTiles[clusterId]) {
            inTilesPerCluster.push_back(VPUIP::backInferSwKernelInputTile(swKernelOp, outTile, _log));
        }
        inTiles.push_back(inTilesPerCluster);
    }
    return inTiles;
}

SmallVector<mlir::Value> ClusterSwKernelRewriter::createNewInputs(VPUIP::SwKernelOp swKernelOp,
                                                                  mlir::ValueRange operands, bool insertSubview,
                                                                  int64_t outTileIndex,
                                                                  mlir::PatternRewriter& rewriter) const {
    const auto inTiles = getOuterMostInputTiling(swKernelOp, outTileIndex);
    SmallVector<mlir::Value> newInputs;
    VPUX_THROW_UNLESS(operands.size() == inTiles.tiles.size(), " operand size is not equal to tile size");

    // if the operand comes from TilingCopy(DDR2CMX), get the op's input
    auto getSourceBufferFromDDR = [](mlir::Value operand) -> mlir::Value {
        auto sourceOp = operand.getDefiningOp<VPUIP::NCEClusterTilingOp>();
        if (sourceOp == nullptr) {
            return nullptr;
        }
        auto innerCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(sourceOp.getInnerTaskOp());
        if (innerCopyOp == nullptr) {
            return nullptr;
        }
        VPUX_THROW_UNLESS(VPUIP::isCopyFromDDR(innerCopyOp), "Tiling Copy is supposed to be from DDR at '{0}'",
                          sourceOp->getLoc());
        return sourceOp.getInputs()[0];
    };

    for (const auto& p : operands | indexed) {
        const auto& index = p.index();
        const auto& operand = p.value();
        const auto& offset = inTiles.tiles[index].offsets;
        const auto& tiledShape = inTiles.tiles[index].shape;

        // handle swkernel's input copy
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfterValue(operand);
        if (insertSubview) {
            auto inputSubview = rewriter.create<VPUIP::SubViewOp>(operand.getLoc(), operand, offset, tiledShape);
            newInputs.push_back(inputSubview);

        } else {
            auto sourceBuffer = getSourceBufferFromDDR(operand);
            if (sourceBuffer == nullptr) {
                // Since the compiler doesn't support copy from DistributedBufferType to DistributedBufferType, input
                // data need copy to DDR then copy back to CMX
                auto origType = swKernelOp.inputs()[index].getType().dyn_cast<vpux::NDTypeInterface>();
                auto newDDRType = origType.changeMemSpace(VPU::MemoryKind::DDR);
                auto newAllocDDROp =
                        rewriter.create<mlir::memref::AllocOp>(operand.getLoc(), newDDRType.cast<mlir::MemRefType>());
                auto tilingCopyBackToDDROp =
                        createNewTilingCopyOp(rewriter, operand.getLoc(), newDDRType, {operand, newAllocDDROp});
                sourceBuffer = tilingCopyBackToDDROp->getResult(0);
            }

            auto inputSubview = rewriter.create<VPUIP::SubViewOp>(operand.getLoc(), sourceBuffer, offset, tiledShape);
            auto newDistributedType = getNewTiledDistributedType(swKernelOp, operand, index, outTileIndex, tiledShape);
            auto newAllocCMXOp =
                    rewriter.create<VPURT::AllocDistributed>(operand.getLoc(), newDistributedType, nullptr, nullptr);
            auto newTilingCopyToCMXOp = createNewTilingCopyOp(rewriter, operand.getLoc(), newDistributedType,
                                                              {inputSubview, newAllocCMXOp});
            newInputs.push_back(newTilingCopyToCMXOp->getResult(0));
        }
    }
    return newInputs;
}

SmallVector<mlir::Value> ClusterSwKernelRewriter::createNewOutBuffs(VPUIP::SwKernelOp swKernelOp,
                                                                    mlir::ValueRange operands, bool insertSubview,
                                                                    int64_t outTileIndex,
                                                                    mlir::PatternRewriter& rewriter) const {
    const auto outTiles = getOuterMostOutputTiling(swKernelOp);

    const auto& tiledShape = outTiles[outTileIndex].shape;
    const auto& offset = outTiles[outTileIndex].offsets;

    SmallVector<mlir::Value> newOutputs;
    for (auto operand : operands) {
        // handle swkernel's output buf
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfterValue(operand);

        if (insertSubview) {
            auto outputSubview = rewriter.create<VPUIP::SubViewOp>(operand.getLoc(), operand, offset, tiledShape);
            newOutputs.push_back(outputSubview);
        } else {
            auto allocType = operand.getType().cast<vpux::NDTypeInterface>();
            auto newAllocType = allocType.changeShape(tiledShape);
            auto newOutputAllocType =
                    rewriter.create<VPURT::AllocDistributed>(operand.getLoc(), newAllocType, nullptr, nullptr);
            newOutputs.push_back(newOutputAllocType);
        }
    }

    return newOutputs;
}

VPUIP::SwKernelOp ClusterSwKernelRewriter::createNewSwKernelOp(VPUIP::SwKernelOp swKernelOp,
                                                               ArrayRef<mlir::Value> newInputs,
                                                               ArrayRef<mlir::Value> newOutBufs, bool insertSubview,
                                                               mlir::PatternRewriter& rewriter) const {
    auto swKernelRun = *swKernelOp.body().getOps<VPUIP::SwKernelRun>().begin();
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    rewriter.setInsertionPointAfter(clusterTilingOp);
    mlir::ArrayAttr strideAttr = getStrideOnEachCluster(swKernelOp, insertSubview);
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
    for (auto& outBuf : newOutBufs) {
        resultTypes.push_back(outBuf.getType());
    }
    auto newSwKernelTask =
            rewriter.create<VPUIP::NCEClusterTilingOp>(swKernelOp->getLoc(), resultTypes, newOperands, bodyBuilder);
    _log.trace("create new cluster shave {0}", newSwKernelTask);

    return mlir::dyn_cast<VPUIP::SwKernelOp>(newSwKernelTask.getInnerTaskOp());
}

void ClusterSwKernelRewriter::replaceOpWithConcatView(VPUIP::SwKernelOp origOp, VPUIP::SwKernelOp newSwKernelOp,
                                                      bool insertSubview, mlir::PatternRewriter& rewriter) const {
    auto origClusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(origOp->getParentOp());
    auto newClusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(newSwKernelOp->getParentOp());
    // Get input ops
    SmallVector<mlir::Operation*> inputDefingOps;
    for (const auto& input : origClusterTilingOp.getInputs()) {
        if (const auto& inputOp = input.getDefiningOp()) {
            inputDefingOps.push_back(inputOp);
        }
    }

    const auto origClusterTilingResults = origClusterTilingOp.getResults();
    const auto resultsNum = origClusterTilingResults.size();
    if (insertSubview) {
        llvm::SmallVector<mlir::Value> newConcats;
        for (auto p : origClusterTilingResults | indexed) {
            const auto index = p.index();
            const auto newResults = newClusterTilingOp->getResults();
            auto concatInputs = llvm::SmallVector<mlir::Value>{newResults[index], newResults[resultsNum + index]};
            auto outBufOp = origClusterTilingOp.output_buffs()[index].getDefiningOp();
            auto concatOp = rewriter.create<VPUIP::ConcatViewOp>(newClusterTilingOp->getLoc(), concatInputs,
                                                                 outBufOp->getResult(0));
            newConcats.push_back(concatOp.getResult());
        }
        rewriter.replaceOp(origClusterTilingOp, mlir::ValueRange{newConcats});
        return;
    }

    auto outTiles = getOuterMostOutputTiling(origOp);
    const auto hasCopyUser = onlyHasCopyOpUser(origOp);
    mlir::DenseMap<int64_t, mlir::memref::AllocOp> newAllocDDROpsMap;
    if (hasCopyUser) {
        for (auto user : origClusterTilingOp->getUsers()) {
            if (auto userCopyOp = mlir::cast<VPUIP::NCEClusterTilingOp>(*user)) {
                rewriter.setInsertionPointAfter(userCopyOp);
                auto newAllocDDROp =
                        mlir::cast<mlir::memref::AllocOp>(userCopyOp.output_buffs().front().getDefiningOp());
                auto operandIt = std::find(origClusterTilingResults.begin(), origClusterTilingResults.end(),
                                           userCopyOp.getOperand(0));
                if (operandIt != origClusterTilingResults.end()) {
                    newAllocDDROpsMap[operandIt - origClusterTilingResults.begin()] = newAllocDDROp;
                }
            }
        }
    } else {
        rewriter.setInsertionPointAfter(newClusterTilingOp);
        for (auto result : origOp.getResults() | indexed) {
            auto newDDRType =
                    result.value().getType().cast<vpux::NDTypeInterface>().changeMemSpace(VPU::MemoryKind::DDR);
            auto newAllocDDROp = rewriter.create<mlir::memref::AllocOp>(newClusterTilingOp->getLoc(),
                                                                        newDDRType.cast<mlir::MemRefType>());
            newAllocDDROpsMap[result.index()] = newAllocDDROp;
        }
    }

    SmallVector<mlir::Value> results;
    for (const auto& item : outTiles | indexed) {
        const auto& outTile = item.value();
        const auto& index = item.index();
        auto outShape = to_small_vector(outTile.shape);
        auto outOffset = to_small_vector(outTile.offsets);

        for (auto p : origClusterTilingResults | indexed) {
            const auto result = p.value();
            const auto resultIdx = p.index();
            if (!result.getUsers().empty()) {
                auto it = newAllocDDROpsMap.find(resultIdx);
                auto outSubview = rewriter.create<VPUIP::SubViewOp>(newClusterTilingOp->getLoc(), it->second, outOffset,
                                                                    outShape);
                auto copyOp = createNewTilingCopyOp(
                        rewriter, newClusterTilingOp->getLoc(), outSubview.getType(),
                        {newClusterTilingOp.getResult(index * resultsNum + resultIdx), outSubview});
                results.push_back(copyOp->getResult(0));
            }
        }
    }

    if (hasCopyUser) {
        for (auto user : llvm::make_early_inc_range(origClusterTilingOp->getUsers())) {
            if (auto userCopyOp = mlir::cast<VPUIP::NCEClusterTilingOp>(*user)) {
                auto operandIt = std::find(origClusterTilingResults.begin(), origClusterTilingResults.end(),
                                           userCopyOp.getOperand(0));
                auto it = newAllocDDROpsMap.find(operandIt - origClusterTilingResults.begin());
                rewriter.replaceOpWithNewOp<VPUIP::ConcatViewOp>(userCopyOp, results, it->second);
            }
        }
        rewriter.eraseOp(origClusterTilingOp);
    } else {
        llvm::SmallVector<mlir::Value> newTilingCopys;
        for (auto p : origClusterTilingResults | indexed) {
            const auto index = p.index();
            auto concatInputs = llvm::SmallVector<mlir::Value>{results[index], results[resultsNum + index]};
            auto it = newAllocDDROpsMap.find(index);
            auto concatOp =
                    rewriter.create<VPUIP::ConcatViewOp>(newClusterTilingOp->getLoc(), concatInputs, it->second);
            auto outType = origClusterTilingOp->getResult(index).getType().cast<vpux::NDTypeInterface>();
            auto newAllocCMXOp =
                    rewriter.create<VPURT::AllocDistributed>(origClusterTilingOp->getLoc(), outType, nullptr, nullptr);

            auto newTilingCopyToCMXOp =
                    createNewTilingCopyOp(rewriter, newClusterTilingOp->getLoc(), outType, {concatOp, newAllocCMXOp});
            newTilingCopys.push_back(newTilingCopyToCMXOp.getResult(0));
        }
        rewriter.replaceOp(origClusterTilingOp, mlir::ValueRange{newTilingCopys});
    }

    for (auto originInputOp : inputDefingOps) {
        if (originInputOp->use_empty()) {
            rewriter.eraseOp(originInputOp);
        }
    }
}

OutputTiling ClusterSwKernelRewriter::getOuterMostOutputTiling(VPUIP::SwKernelOp swKernelOp) const {
    auto outShaveTiles = calculateOutputTiles(swKernelOp).value();

    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    VPUX_THROW_WHEN(clusterTilingOp == nullptr, "Unexpected parent op type at '{0}'", swKernelOp->getLoc());
    auto distributedType = clusterTilingOp.getResult(0).getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto mode = distributedType.getDistribution().getMode().getValue();
    if (mode == VPU::DistributionMode::DUPLICATED) {
        return outShaveTiles.front();
    }

    auto tileDim = getSwKernelTileDim(swKernelOp);

    auto tiledNumOnEachCluster = outShaveTiles.front().size();

    auto getOuterMostShapeValueOnTiledDim = [&](int64_t idx) {
        int64_t tiledDimShapeValue = 0;
        for (auto& outTile : outShaveTiles) {
            tiledDimShapeValue += outTile[idx].shape[tileDim];
        }
        return tiledDimShapeValue;
    };
    OutputTiling outputTiles;
    for (auto outTileIndex : irange(tiledNumOnEachCluster)) {
        Shape shape(outShaveTiles.front()[outTileIndex].shape);
        shape[tileDim] = getOuterMostShapeValueOnTiledDim(outTileIndex);

        int64_t tiledDimOffsetValue = 0;
        for (auto idx : irange(outTileIndex)) {
            tiledDimOffsetValue += getOuterMostShapeValueOnTiledDim(idx);
        }
        Shape offset(shape.size(), 0);
        offset[tileDim] = tiledDimOffsetValue;
        Shape axis(shape.size(), 0);
        axis[tileDim] = outShaveTiles.front().size();
        outputTiles.push_back(TileInfo(shape, offset, axis));
    }
    return outputTiles;
}

InputTiling ClusterSwKernelRewriter::getOuterMostInputTiling(VPUIP::SwKernelOp swKernelOp, int64_t outTileIdx) const {
    auto outTiles = getOuterMostOutputTiling(swKernelOp);
    return VPUIP::backInferSwKernelInputTile(swKernelOp, outTiles[outTileIdx], _log);
}

bool ClusterSwKernelRewriter::onlyHasCopyOpUser(VPUIP::SwKernelOp swKernelOp) const {
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    if (!clusterTilingOp->hasOneUse()) {
        return false;
    }
    auto userCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(*clusterTilingOp->getUsers().begin());
    return userCopyOp != nullptr && mlir::isa<VPUIP::CopyOp>(userCopyOp.getInnerTaskOp());
}

vpux::NDTypeInterface ClusterSwKernelRewriter::getNewTiledDistributedType(VPUIP::SwKernelOp swKernelOp,
                                                                          mlir::Value outerOperand, int64_t operandIdx,
                                                                          int64_t outTileIndex,
                                                                          ShapeRef tiledShape) const {
    auto distributionAttr = outerOperand.getType().dyn_cast<VPUIP::DistributedBufferType>().getDistribution();
    auto shapes = distributionAttr.getComputeShapes();
    auto offsets = distributionAttr.getComputeOffsets();
    if (shapes == nullptr && offsets == nullptr) {
        return outerOperand.getType().dyn_cast<VPUIP::DistributedBufferType>().changeShape(tiledShape);
    }
    auto numCluster = shapes.size();
    auto outTiles = getOuterMostOutputTiling(swKernelOp);
    auto inTiles = getOuterMostInputTiling(swKernelOp, outTileIndex);
    auto subOutTile = getSwKernelOutputTiling(swKernelOp, outTiles[outTileIndex].shape, numCluster, _log);
    VPUX_THROW_WHEN(mlir::failed(subOutTile), "Invalid output tiling for {0}", swKernelOp.getLoc());

    SmallVector<SmallVector<int64_t>> newTiledShape;
    SmallVector<SmallVector<int64_t>> newTiledOffset;
    auto baseOutOffset = to_small_vector(outTiles[outTileIndex].offsets);
    for (auto& tile : subOutTile.value()) {
        // Adjust the offset against the original output
        auto offset = to_small_vector(tile.offsets);
        SmallVector<int64_t> adjustedOffset;
        std::transform(offset.begin(), offset.end(), baseOutOffset.begin(), std::back_inserter(adjustedOffset),
                       std::plus<int64_t>());
        tile.offsets = Shape(adjustedOffset);

        // Back infer the input tiles against the original input
        auto subInTiles = VPUIP::backInferSwKernelInputTile(swKernelOp, tile, _log);
        newTiledShape.push_back(to_small_vector(subInTiles.tiles[operandIdx].shape));
        newTiledOffset.push_back(to_small_vector(subInTiles.tiles[operandIdx].offsets));
    }

    // Get the offset against current input tile
    auto baseInOffset = newTiledOffset[0];
    for (auto& tileOffset : newTiledOffset) {
        SmallVector<int64_t> adjustedOffset;
        std::transform(tileOffset.begin(), tileOffset.end(), baseInOffset.begin(), std::back_inserter(adjustedOffset),
                       std::minus<int64_t>());
        tileOffset = std::move(adjustedOffset);
    }

    // create new input distributed type with updated shape and offset
    auto distributedType = outerOperand.getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto ctx = swKernelOp->getContext();
    auto shapesAttr = vpux::getIntArrayOfArray(ctx, newTiledShape);
    auto offsetsAttr = vpux::getIntArrayOfArray(ctx, newTiledOffset);
    auto newDistribution = VPU::DistributedTensorAttr::get(
            ctx, distributionAttr.getMode(), distributionAttr.getNumTiles(), distributionAttr.getKernel(),
            distributionAttr.getPads(), distributionAttr.getStrides(), distributionAttr.getNumClusters(),
            distributionAttr.getAlignment(), distributionAttr.getUniformDistributedSegments(), shapesAttr, offsetsAttr,
            shapesAttr, offsetsAttr, nullptr);
    return VPUIP::DistributedBufferType::get(ctx, tiledShape.raw(), distributedType.getElementType(),
                                             distributedType.getLayout(), distributedType.getMemSpace(),
                                             newDistribution, distributedType.getCompressionScheme());
}

mlir::ArrayAttr ClusterSwKernelRewriter::getStrideOnEachCluster(VPUIP::SwKernelOp swKernelOp,
                                                                bool insertSubview) const {
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    VPUX_THROW_WHEN(clusterTilingOp == nullptr, "Unexpected parent op type at '{0}'", swKernelOp->getLoc());
    auto distributedType = clusterTilingOp.getResult(0).getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto dimOrder = distributedType.getDimsOrder();
    mlir::ArrayAttr strideAttr = nullptr;
    SmallVector<SmallVector<int64_t>> strideOnPerClusters;
    if (insertSubview) {
        // If swkernel supports stride access, the operands and results are created by subview of the original
        // distributed buffer. Need calculate the stride by the original shape on each cluster
        for (auto& shape : distributedType.getPerClusterComputeShapes()) {
            SmallVector<int64_t> strideOnPerCluster(shape.size());
            int64_t preStride = 1;
            for (int64_t idx = dimOrder.numDims() - 1; idx >= 0; idx--) {
                auto dim = dimOrder.dimAt(idx);
                strideOnPerCluster[dim.ind()] = preStride;
                preStride *= shape[dim];
            }
            strideOnPerClusters.push_back(strideOnPerCluster);
        }
        strideAttr = vpux::getIntArrayOfArray(swKernelOp->getContext(), strideOnPerClusters);
    }
    return strideAttr;
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
    // TODO: #70860

    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto nceTile = IE::getAvailableExecutor(module, VPU::ExecutorKind::NCE);
    auto shaveActCount = nceTile.getSubExecutor(VPU::ExecutorKind::SHAVE_ACT).count();

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
