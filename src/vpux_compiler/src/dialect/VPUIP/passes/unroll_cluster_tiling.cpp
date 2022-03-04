//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <llvm/ADT/DenseMap.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <vpux/compiler/utils/rewriter.hpp>

using namespace vpux;

namespace {

bool isSegmentedNCETask(VPUIP::NCEClusterTaskOp nceTask, VPUIP::DistributedBufferType inputType) {
    // Only for explicit SEGMENTED mode, not in combination with
    // DUPLICATED or MULTICASTED
    if (inputType.getDistribution().mode().getValue() != VPU::DistributionMode::SEGMENTED) {
        return false;
    }

    // Segmentation not present on H axis
    const auto numTiles = parseIntArrayAttr<int64_t>(inputType.getDistribution().num_tiles());
    if (numTiles[Dims4D::Act::H.ind()] <= 1) {
        return false;
    }

    // Segmentation not supported with non NHWC input such as CM Conv
    if (inputType.getDimsOrder() != DimsOrder::NHWC) {
        return false;
    }

    // Only for valid segmentation cases, when need to read lines from
    // other clusters
    if (nceTask.kernel_sizeAttr() != nullptr) {
        const auto kernels = parseIntArrayAttr<int64_t>(nceTask.kernel_sizeAttr());
        if (kernels[Dims4D::Kernel::Y.ind()] <= 1) {
            return false;
        }
    }

    return true;
}

//
// ClusterNCERewriter
//

class ClusterNCERewriter final : public mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp> {
public:
    ClusterNCERewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp>(ctx), _log(log) {
        setDebugName("ClusterNCERewriter");

        _ctx = getContext();
        _cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::NCEClusterTaskOp nceTask, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::Value getClusterOperand(VPUIP::NCEClusterTilingOp clusterOp, mlir::Value innerOperand) const;
    SmallVector<mlir::Value> getPerClusterBuffers(mlir::Location loc, StringRef bufferName,
                                                  VPUIP::NCEClusterTilingOp clusterOp, mlir::Value innerOperand,
                                                  int64_t numClusters, mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
    mlir::MLIRContext* _ctx;
    mlir::FlatSymbolRefAttr _cmxNameAttr;
};

mlir::Value ClusterNCERewriter::getClusterOperand(VPUIP::NCEClusterTilingOp clusterOp, mlir::Value innerOperand) const {
    if (innerOperand == nullptr) {
        return nullptr;
    }

    const auto blockArg = innerOperand.dyn_cast<mlir::BlockArgument>();
    VPUX_THROW_WHEN(blockArg == nullptr, "Inner operand '{0}' is not a block argument", innerOperand);
    VPUX_THROW_WHEN(blockArg.getOwner() != clusterOp.getInnerTaskOp()->getBlock(),
                    "Cannot match the origin operand with the inner one '{0}'", innerOperand);

    return clusterOp->getOperand(blockArg.getArgNumber());
}

SmallVector<mlir::Value> ClusterNCERewriter::getPerClusterBuffers(mlir::Location loc, StringRef bufferName,
                                                                  VPUIP::NCEClusterTilingOp clusterOp,
                                                                  mlir::Value innerOperand, int64_t numClusters,
                                                                  mlir::PatternRewriter& rewriter) const {
    auto clusterOperand = getClusterOperand(clusterOp, innerOperand);
    if (clusterOperand == nullptr) {
        return SmallVector<mlir::Value>(numClusters, nullptr);
    }

    auto innerOperandType = innerOperand.getType().cast<vpux::NDTypeInterface>();

    auto operandType = clusterOperand.getType();
    auto distributedType = operandType.dyn_cast<VPUIP::DistributedBufferType>();
    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported operand type {0}", operandType);

    auto perClusterShapes = distributedType.getPerClusterComputeShapes();
    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(numClusters),
                      "Number of shapes '{0}' and clusters '{1}' are mismatch", perClusterShapes.size(), numClusters);

    const auto distribution = distributedType.getDistribution();
    const auto distributionMode = distribution.mode().getValue();

    auto declBuff = clusterOperand.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset");

    SmallVector<mlir::Value> perClusterBuffers(numClusters);
    if (distributionMode == VPU::DistributionMode::SEGMENTED || distributionMode == VPU::DistributionMode::DUPLICATED) {
        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            auto cmxBuffType = innerOperandType.changeShape(perClusterShapes[clusterId]);
            const auto symbolAttr =
                    vpux::IndexedSymbolAttr::get(_ctx, {_cmxNameAttr, vpux::getIntAttr(_ctx, clusterId)});
            cmxBuffType = cmxBuffType.changeMemSpace(symbolAttr);

            const auto newLoc = appendLoc(loc, llvm::formatv("_{0}_cluster_{1}", bufferName, clusterId).str());
            auto newCmxBuffer = rewriter.create<VPURT::DeclareBufferOp>(
                    newLoc, cmxBuffType, VPURT::BufferSection::CMX_NN, clusterId, declBuff.byteOffset());
            _log.trace("Insert new CMX buffer: '{0}'", newCmxBuffer);

            perClusterBuffers[clusterId] = newCmxBuffer;
        }

        return perClusterBuffers;
    }

    //       Task1(SOK)
    // CMX0 |-out part1-|-out part2-|
    // CMX1 |-out part1-|-out part2-|
    //                    Task2(SOK)
    if (distributionMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED)) {
        SmallVector<int64_t> clusters(numClusters);
        for (int64_t i = 0; i < numClusters; ++i) {
            clusters[i] = i;
        }

        const auto elemSize = innerOperandType.getElemTypeSize();
        const auto elemStrides = to_small_vector(innerOperandType.getStrides() | transformed([&](Bit stride) {
                                                     return stride.count() / elemSize.count();
                                                 }));

        const auto order = innerOperandType.getDimsOrder();
        const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(_ctx));
        const auto stridesAttr = getIntArrayAttr(_ctx, elemStrides);
        auto layout = IERT::MemRefAttr::get(orderAttr, stridesAttr, _ctx);

        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            const auto newDistributedType = VPUIP::DistributedBufferType::get(
                    _ctx, perClusterShapes[clusterId].raw(), distributedType.getElementType(), layout,
                    distributedType.getMemSpace(), distributedType.getDistribution());

            const auto newLoc = appendLoc(loc, llvm::formatv("_{0}_cluster_{1}", bufferName, clusterId).str());
            auto newCmxBuffer = rewriter.create<VPURT::DeclareBufferOp>(
                    newLoc, newDistributedType, VPURT::BufferSection::CMX_NN, clusters, declBuff.byteOffset());
            _log.trace("Insert new CMX buffer: '{0}'", newCmxBuffer);

            perClusterBuffers[clusterId] = newCmxBuffer;
        }

        return perClusterBuffers;
    }

    VPUX_THROW("Unsupported distribution mode: {0}", VPU::stringifyDistributionMode(distributionMode));
}

mlir::LogicalResult ClusterNCERewriter::matchAndRewrite(VPUIP::NCEClusterTaskOp nceTask,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Process NCE op: '{0}'", nceTask);
    auto clusterOp = nceTask->getParentOfType<VPUIP::NCEClusterTilingOp>();
    if (clusterOp == nullptr) {
        return mlir::failure();
    }

    auto vpurtTask = clusterOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");
    rewriter.setInsertionPointAfter(vpurtTask);

    VPUX_THROW_UNLESS(!clusterOp.getInputs().empty(), "Wrong inputs size: {0}", clusterOp.getInputs().size());
    VPUX_THROW_UNLESS(clusterOp.getOutputs().size() == 1, "Wrong outputs size: {0}", clusterOp.getOutputs().size());

    auto parentInput = *clusterOp.getInputs().begin();
    auto parentOutput = *clusterOp.getOutputs().begin();

    auto parentInputType = parentInput.getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto parentOutputType = parentOutput.getType().dyn_cast<VPUIP::DistributedBufferType>();

    VPUX_THROW_UNLESS(parentInputType != nullptr && parentOutputType != nullptr,
                      "Input and output types must have distributed type. Got: inT={0}, outT={1}", parentInputType,
                      parentOutputType);

    auto inDistribution = parentInputType.getDistribution();
    auto outDistribution = parentOutputType.getDistribution();

    VPUX_THROW_UNLESS(inDistribution.num_clusters() == outDistribution.num_clusters(),
                      "Input '{0}' and output '{1}' number of clusters are not equal", inDistribution.num_clusters(),
                      outDistribution.num_clusters());

    auto numClusters = inDistribution.num_clusters().getInt();
    auto loc = nceTask->getLoc();
    auto inputBuffs = getPerClusterBuffers(loc, "input", clusterOp, nceTask.input(), numClusters, rewriter);
    auto weightsBuffs = getPerClusterBuffers(loc, "weights", clusterOp, nceTask.weights(), numClusters, rewriter);
    auto weightTableBuffs =
            getPerClusterBuffers(loc, "weightTable", clusterOp, nceTask.weight_table(), numClusters, rewriter);
    auto activationWindowBuffs = getPerClusterBuffers(loc, "activationWindow", clusterOp, nceTask.activation_window(),
                                                      numClusters, rewriter);
    auto outputBuffs = getPerClusterBuffers(loc, "outputBuff", clusterOp, nceTask.output_buff(), numClusters, rewriter);

    mlir::UnitAttr isSegmented = nullptr;
    if (isSegmentedNCETask(nceTask, parentInputType)) {
        isSegmented = mlir::UnitAttr::get(nceTask.getContext());
    }

    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        const auto newLoc = appendLoc(loc, llvm::formatv("_cluster_{0}", clusterId).str());
        auto newTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
                rewriter, vpurtTask.waitBarriers(), vpurtTask.updateBarriers(), newLoc, inputBuffs[clusterId],
                weightsBuffs[clusterId], weightTableBuffs[clusterId], activationWindowBuffs[clusterId], parentInput,
                parentOutput, outputBuffs[clusterId], nceTask.task_type(), nceTask.kernel_sizeAttr(),
                nceTask.kernel_stridesAttr(), nceTask.kernel_paddingAttr(),
                nceTask.activation_window_channel_lengthAttr(), nullptr, nullptr, isSegmented);

        {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToEnd(&newTask.variants().front());

            for (auto variant : nceTask.variants().getOps<VPUIP::DPUTaskOp>()) {
                VPUX_THROW_UNLESS(variant.cluster_id().hasValue(), "Unable to distribute workload");
                if (variant.cluster_id().getValue() == clusterId) {
                    rewriter.clone(*variant);
                }
            }
        }

        {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToEnd(&newTask.ppe().front());

            for (auto& ppe : nceTask.ppe().getOps()) {
                rewriter.clone(ppe);
            }
        }

        _log.trace("Insert new NCE task: '{0}'", newTask);
    }

    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

//
// ClusterDMARewriter
//

class ClusterDMARewriter final : public mlir::OpRewritePattern<VPUIP::NNDMAOp> {
public:
    ClusterDMARewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::NNDMAOp>(ctx), _log(log) {
        setDebugName("ClusterDMARewriter");

        _ctx = getContext();
        _cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::NNDMAOp nndmaOp, mlir::PatternRewriter& rewriter) const final;

private:
    void unrollSegmented(mlir::Location loc, VPUIP::NCEClusterTilingOp clusterOp, VPURT::TaskOp vpurtTask,
                         VPUIP::DistributedBufferType distributedType, mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
    mlir::MLIRContext* _ctx;
    mlir::FlatSymbolRefAttr _cmxNameAttr;
};

void ClusterDMARewriter::unrollSegmented(mlir::Location loc, VPUIP::NCEClusterTilingOp clusterOp,
                                         VPURT::TaskOp vpurtTask, VPUIP::DistributedBufferType distributedType,
                                         mlir::PatternRewriter& rewriter) const {
    const auto input = *clusterOp.getInputs().begin();
    const auto output = *clusterOp.getOutputs().begin();

    const auto inputType = input.getType().cast<vpux::NDTypeInterface>();
    const auto outputType = output.getType().cast<vpux::NDTypeInterface>();

    const auto innerInput = *clusterOp.getInnerInputs().begin();
    const auto innerOutput = *clusterOp.getInnerOutputs().begin();

    const auto innerInputType = innerInput.getType().cast<vpux::NDTypeInterface>();
    const auto innerOutputType = innerOutput.getType().cast<vpux::NDTypeInterface>();

    const auto distributionAttr = distributedType.getDistribution();
    const auto numClusters = distributionAttr.num_clusters().getInt();

    const auto originInShape = inputType.getShape().raw();
    const auto originOutShape = outputType.getShape().raw();

    const auto strideInReqs = StrideReqs::compact(originInShape.size());
    VPUX_THROW_UNLESS(strideInReqs.checkStrides(input), "Only compact strides are supported");
    const auto strideOutReqs = StrideReqs::compact(originOutShape.size());
    VPUX_THROW_UNLESS(strideOutReqs.checkStrides(output), "Only compact strides are supported");

    const auto numTiles = parseIntArrayAttr<int64_t>(distributionAttr.num_tiles());
    VPUX_THROW_UNLESS(originInShape.size() == numTiles.size(),
                      "Input shape size '{0}' and tiles array size '{1}' are mismatch", originInShape.size(),
                      numTiles.size());

    const auto perClusterShapes = distributedType.getPerClusterComputeShapes();
    VPUX_THROW_UNLESS(checked_cast<int64_t>(perClusterShapes.size()) == numClusters,
                      "Number of shapes '{0}' and clusters '{1}' are mismatch", perClusterShapes.size(), numClusters);
    const auto perClusterShapeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    VPUX_THROW_UNLESS(checked_cast<int64_t>(perClusterShapeOffsets.size()) == numClusters,
                      "Number of shape offsets '{0}' and clusters '{1}' are mismatch", perClusterShapeOffsets.size(),
                      numClusters);

    const auto createNewTypes = [&](vpux::NDTypeInterface innerType) {
        SmallVector<vpux::NDTypeInterface> newTypes(numClusters);
        for (size_t clusterId = 0; clusterId < perClusterShapes.size(); ++clusterId) {
            newTypes[clusterId] = innerType.changeShape(perClusterShapes[clusterId]);
        }

        return newTypes;
    };

    const auto isValidTile = [](auto dim) {
        return dim > 1;
    };

    const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.num_tiles());
    const auto axis = std::distance(tilingScheme.begin(), llvm::find_if(tilingScheme, isValidTile));
    const auto strides = distributedType.getStrides();

    const auto inTypes = createNewTypes(innerInputType);
    const auto outTypes = createNewTypes(innerOutputType);

    const auto getOperand = [&](int64_t clusterId, mlir::Value operand, vpux::NDTypeInterface newType) -> mlir::Value {
        // For example, copy of weights in case of SOK
        // <32x16x1x1xfp16, @DDR>  -> <16x16x1x1xfp16, [@CMX, 0]>
        //                         -> <16x16x1x1xfp16, [@CMX, 1]>
        if (auto cts = operand.getDefiningOp<Const::DeclareOp>()) {
            VPUX_THROW_UNLESS(outputType.getMemoryKind() == VPU::MemoryKind::CMX_NN,
                              "Output operand type must have NN_CMX memory space. Got: {0}",
                              outputType.getMemoryKind());

            return rewriter.create<IERT::SubViewOp>(loc, cts, perClusterShapeOffsets[clusterId].raw(),
                                                    perClusterShapes[clusterId].raw());
        }

        auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset");

        // For example, copy of input in case of SOH
        // <1x16x33x32xf16, @DDR>  -> <1x16x17x32xf16, [@CMX, 0]>
        //                         -> <1x16x16x32xf16, [@CMX, 1]>

        // OR copy back of output in case of SOH
        // <1x16x17x32xf16, [@CMX, 0]>  -> <1x16x33x32xf16, @DDR>
        // <1x16x16x32xf16, [@CMX, 1]>  /

        if (newType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
            const auto symbolAttr =
                    vpux::IndexedSymbolAttr::get(_ctx, {_cmxNameAttr, vpux::getIntAttr(_ctx, clusterId)});
            auto newCMXType = newType.changeMemSpace(symbolAttr);
            return rewriter.create<VPURT::DeclareBufferOp>(loc, newCMXType, VPURT::BufferSection::CMX_NN, clusterId,
                                                           declBuff.byteOffset());
        }

        Byte ddrOffset{declBuff.byteOffset()};
        ddrOffset += perClusterShapeOffsets[clusterId][Dim(axis)] * static_cast<Byte>(strides[Dim(axis)]);

        return rewriter.create<VPURT::DeclareBufferOp>(loc, newType, VPURT::BufferSection::DDR, ddrOffset.count());
    };

    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        const auto newInputType = inTypes[clusterId];
        const auto newOutType = outTypes[clusterId];

        const auto inputBuffer = getOperand(clusterId, input, newInputType);
        _log.trace("Insert new input buffer declaration: '{0}'", inputBuffer);

        const auto outBuffer = getOperand(clusterId, output, newOutType);
        _log.trace("Insert new output buffer declaration: '{0}'", outBuffer);

        const auto newLoc = appendLoc(loc, llvm::formatv("_cluster_{0}", clusterId).str());
        const auto newNNDMA = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                rewriter, vpurtTask.waitBarriers(), vpurtTask.updateBarriers(), newLoc, inputBuffer, outBuffer);
        _log.trace("Insert new NNDMA op: '{0}'", newNNDMA);
    }
}

mlir::LogicalResult ClusterDMARewriter::matchAndRewrite(VPUIP::NNDMAOp nndmaOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Process NNDMA op: {0}", nndmaOp);

    auto clusterOp = nndmaOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
    if (clusterOp == nullptr) {
        _log.trace("NNDMA is not a child of NCEClusterTiling op");
        return mlir::failure();
    }

    VPUX_THROW_UNLESS(clusterOp.getInputs().size() == 1, "Wrong inputs size: {0}", clusterOp.getInputs().size());
    VPUX_THROW_UNLESS(clusterOp.getOutputs().size() == 1, "Wrong outputs size: {0}", clusterOp.getOutputs().size());

    const auto input = *clusterOp.getInputs().begin();
    const auto output = *clusterOp.getOutputs().begin();

    const auto inputType = input.getType().cast<vpux::NDTypeInterface>();
    const auto outputType = output.getType().cast<vpux::NDTypeInterface>();

    VPUX_THROW_UNLESS(clusterOp.getInnerInputs().size() == 1, "Wrong inputs size: {0}",
                      clusterOp.getInnerInputs().size());
    VPUX_THROW_UNLESS(clusterOp.getInnerOutputs().size() == 1, "Wrong outputs size: {0}",
                      clusterOp.getInnerOutputs().size());

    auto vpurtTask = clusterOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");
    rewriter.setInsertionPointAfter(vpurtTask);

    const auto distributedType = inputType.isa<VPUIP::DistributedBufferType>()
                                         ? inputType.dyn_cast<VPUIP::DistributedBufferType>()
                                         : outputType.dyn_cast<VPUIP::DistributedBufferType>();

    VPUX_THROW_UNLESS(distributedType != nullptr, "One of operands must have DistributedBuffer type");
    VPUX_THROW_WHEN(inputType.isa<VPUIP::DistributedBufferType>() && outputType.isa<VPUIP::DistributedBufferType>(),
                    "Only one operand can have DistributedBuffer type");

    const auto loc = nndmaOp->getLoc();
    const auto distributionAttr = distributedType.getDistribution();
    const auto mode = distributionAttr.mode().getValue();
    if (mode == VPU::DistributionMode::SEGMENTED) {
        _log.trace("Process SEGMENTED mode");
        unrollSegmented(loc, clusterOp, vpurtTask, distributedType, rewriter);
    } else if (mode == VPU::DistributionMode::DUPLICATED) {
        // For example, copy of weights in case of SOH
        // <16x16x1x1xf16, @DDR> -> <16x16x1x1xf16, [@CMX, 0]>
        //                       -> <16x16x1x1xf16, [@CMX, 1]>

        _log.trace("Process DUPLICATED mode");
        VPUX_THROW_UNLESS(outputType.isa<VPUIP::DistributedBufferType>(),
                          "Output operand must have DistributedBuffer type");

        auto outDeclBuff = output.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(outDeclBuff != nullptr, "Can't get output buffer offset");

        const auto numClusters = distributionAttr.num_clusters().getInt();
        SmallVector<int64_t> clusters(numClusters);
        for (int64_t i = 0; i < numClusters; ++i) {
            clusters[i] = i;
        }

        auto cmxBuffer = rewriter.create<VPURT::DeclareBufferOp>(
                loc, outDeclBuff.getType(), VPURT::BufferSection::CMX_NN, clusters, outDeclBuff.byteOffset());
        _log.trace("Insert new CMX buffer declaration: '{0}'", cmxBuffer);

        const auto newLoc = appendLoc(
                loc, llvm::formatv("_broadcast_copy_to_CMX[{0},{1}]", clusters.front(), clusters.back()).str());
        const auto newNNDMA = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                rewriter, vpurtTask.waitBarriers(), vpurtTask.updateBarriers(), newLoc, input, cmxBuffer);
        _log.trace("Insert new NNDMA op: '{0}'", newNNDMA);
    } else if (mode == (VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED)) {
        // For example, copy back of output in case of SOK
        // <1x32x32x32xf16, [@CMX, 0]> -> <1x32x32x32xf16, @DDR>
        // <1x32x32x32xf16, [@CMX, 1]>

        _log.trace("Process DUPLICATED|SEGMENTED mode");
        VPUX_THROW_UNLESS(inputType.isa<VPUIP::DistributedBufferType>(),
                          "Input operand must have DistributedBuffer type");

        auto inDeclBuff = input.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(inDeclBuff != nullptr, "Can't get input buffer offset");

        const auto symbolAttr = vpux::IndexedSymbolAttr::get(_ctx, {_cmxNameAttr, vpux::getIntAttr(_ctx, 0)});

        const auto innerInput = *clusterOp.getInnerInputs().begin();
        const auto innerInputType = innerInput.getType().cast<vpux::NDTypeInterface>();
        const auto newInType = innerInputType.changeMemSpace(symbolAttr);

        auto cmxBuffer = rewriter.create<VPURT::DeclareBufferOp>(loc, newInType, VPURT::BufferSection::CMX_NN, 0,
                                                                 inDeclBuff.byteOffset());
        _log.trace("Insert new CMX buffer declaration: '{0}'", cmxBuffer);

        const auto newNNDMA = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(rewriter, vpurtTask.waitBarriers(),
                                                                    vpurtTask.updateBarriers(), loc, cmxBuffer, output);
        _log.trace("Insert new NNDMA op: '{0}'", newNNDMA);
    } else {
        VPUX_THROW("Unsupported distribution mode: {0}", VPU::stringifyDistributionMode(mode));
    }

    rewriter.eraseOp(vpurtTask);

    return mlir::success();
}

//
// UnrollClusterTilingPass
//

class UnrollClusterTilingPass final : public VPUIP::UnrollClusterTilingBase<UnrollClusterTilingPass> {
public:
    explicit UnrollClusterTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollClusterTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ClusterDMARewriter>(&ctx, _log);
    patterns.insert<ClusterNCERewriter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollClusterTilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnrollClusterTilingPass(Logger log) {
    return std::make_unique<UnrollClusterTilingPass>(log);
}
