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

mlir::Value getClusterOperand(VPUIP::NCEClusterTilingOp clusterOp, mlir::Value innerOperand) {
    if (innerOperand == nullptr) {
        return nullptr;
    }

    const auto blockArg = innerOperand.dyn_cast<mlir::BlockArgument>();
    VPUX_THROW_WHEN(blockArg == nullptr, "Inner operand '{0}' is not a block argument", innerOperand);
    VPUX_THROW_WHEN(blockArg.getOwner() != clusterOp.getInnerTaskOp()->getBlock(),
                    "Cannot match the origin operand with the inner one '{0}'", innerOperand);

    return clusterOp->getOperand(blockArg.getArgNumber());
}

SmallVector<mlir::Value> getPerClusterBuffers(mlir::Location loc, VPUIP::NCEClusterTilingOp clusterOp,
                                              mlir::Value innerOperand, int64_t numClusters,
                                              mlir::PatternRewriter& rewriter) {
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
    VPUX_THROW_UNLESS(distributionMode == VPU::DistributionMode::SEGMENTED ||
                              distributionMode == VPU::DistributionMode::DUPLICATED,
                      "Unsupported distribution mode {0}", VPU::stringifyDistributionMode(distributionMode));

    auto declBuff = clusterOperand.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset");

    SmallVector<mlir::Value> perClusterBuffers{};
    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        auto cmxBuffType = innerOperandType.changeShape(perClusterShapes[clusterId]);

        auto* ctx = rewriter.getContext();
        const auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, "CMX_NN");
        const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, clusterId)});
        cmxBuffType = cmxBuffType.changeMemSpace(symbolAttr);

        auto cmxBuffer = rewriter.create<VPURT::DeclareBufferOp>(loc, cmxBuffType, VPURT::BufferSection::CMX_NN,
                                                                 clusterId, declBuff.byteOffset());

        perClusterBuffers.push_back(cmxBuffer.getResult());
    }

    return perClusterBuffers;
}

//
// ClusterNCERewriter
//

class ClusterNCERewriter final : public mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp> {
public:
    ClusterNCERewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp>(ctx), _log(log) {
        setDebugName("ClusterNCERewriter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::NCEClusterTaskOp nceTask, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ClusterNCERewriter::matchAndRewrite(VPUIP::NCEClusterTaskOp nceTask,
                                                        mlir::PatternRewriter& rewriter) const {
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

    auto inDistributionMode = inDistribution.mode().getValue();
    auto outDistributionMode = outDistribution.mode().getValue();
    VPUX_THROW_UNLESS(
            inDistributionMode == outDistributionMode && inDistributionMode == VPU::DistributionMode::SEGMENTED,
            "Only SEGMENTED distribution mode is supported. in mode = '{0}', out mode = '{1}'",
            VPU::stringifyDistributionMode(inDistributionMode), VPU::stringifyDistributionMode(outDistributionMode));

    auto numClusters = inDistribution.num_clusters().getInt();
    auto loc = vpurtTask->getLoc();
    auto inputBuffs = getPerClusterBuffers(loc, clusterOp, nceTask.input(), numClusters, rewriter);
    auto weightsBuffs = getPerClusterBuffers(loc, clusterOp, nceTask.weights(), numClusters, rewriter);
    auto weightTableBuffs = getPerClusterBuffers(loc, clusterOp, nceTask.weight_table(), numClusters, rewriter);
    auto activationWindowBuffs =
            getPerClusterBuffers(loc, clusterOp, nceTask.activation_window(), numClusters, rewriter);
    auto outputBuffs = getPerClusterBuffers(loc, clusterOp, nceTask.output_buff(), numClusters, rewriter);

    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        auto newTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
                rewriter, vpurtTask.waitBarriers(), vpurtTask.updateBarriers(), loc, inputBuffs[clusterId],
                weightsBuffs[clusterId], weightTableBuffs[clusterId], activationWindowBuffs[clusterId], parentInput,
                parentOutput, outputBuffs[clusterId], nceTask.task_type(), nceTask.kernel_sizeAttr(),
                nceTask.kernel_stridesAttr(), nceTask.kernel_paddingAttr(),
                nceTask.activation_window_channel_lengthAttr());

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
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::NNDMAOp nndmaOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ClusterDMARewriter::matchAndRewrite(VPUIP::NNDMAOp nndmaOp, mlir::PatternRewriter& rewriter) const {
    auto clusterOp = nndmaOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
    if (clusterOp == nullptr) {
        return mlir::failure();
    }

    VPUX_THROW_UNLESS(clusterOp.getInputs().size() == 1, "Wrong inputs size: {0}", clusterOp.getInputs().size());
    VPUX_THROW_UNLESS(clusterOp.getOutputs().size() == 1, "Wrong outputs size: {0}", clusterOp.getOutputs().size());

    auto input = *clusterOp.getInputs().begin();
    auto output = *clusterOp.getOutputs().begin();

    auto inputType = input.getType().cast<vpux::NDTypeInterface>();
    auto outputType = output.getType().cast<vpux::NDTypeInterface>();

    VPUX_THROW_UNLESS(clusterOp.getInnerInputs().size() == 1, "Wrong inputs size: {0}",
                      clusterOp.getInnerInputs().size());
    VPUX_THROW_UNLESS(clusterOp.getInnerOutputs().size() == 1, "Wrong outputs size: {0}",
                      clusterOp.getInnerOutputs().size());

    auto innerInput = *clusterOp.getInnerInputs().begin();
    auto innerOutput = *clusterOp.getInnerOutputs().begin();

    auto innerInputType = innerInput.getType().cast<vpux::NDTypeInterface>();
    auto innerOutputType = innerOutput.getType().cast<vpux::NDTypeInterface>();

    auto vpurtTask = clusterOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");
    rewriter.setInsertionPointAfter(vpurtTask);

    auto distributedType = inputType.isa<VPUIP::DistributedBufferType>()
                                   ? inputType.dyn_cast<VPUIP::DistributedBufferType>()
                                   : outputType.dyn_cast<VPUIP::DistributedBufferType>();

    VPUX_THROW_UNLESS(distributedType != nullptr, "One of operands must have DistributedBuffer type");
    VPUX_THROW_WHEN(inputType.isa<VPUIP::DistributedBufferType>() && outputType.isa<VPUIP::DistributedBufferType>(),
                    "Only one operand can have DistributedBuffer type");

    auto distributionAttr = distributedType.getDistribution();
    auto numClusters = distributionAttr.num_clusters().getInt();

    auto outDeclBuff = output.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(outDeclBuff != nullptr, "Can't get output buffer offset");

    auto mode = distributionAttr.mode().getValue();
    if (mode == VPU::DistributionMode::SEGMENTED) {
        auto originInShape = inputType.getShape().raw();
        auto originOutShape = outputType.getShape().raw();

        const auto strideInReqs = StrideReqs::compact(originInShape.size());
        VPUX_THROW_UNLESS(strideInReqs.checkStrides(input), "Only compact strides are supported");
        const auto strideOutReqs = StrideReqs::compact(originOutShape.size());
        VPUX_THROW_UNLESS(strideOutReqs.checkStrides(output), "Only compact strides are supported");

        auto numTiles = parseIntArrayAttr<int64_t>(distributionAttr.num_tiles());
        VPUX_THROW_UNLESS(originInShape.size() == numTiles.size(),
                          "Input shape size '{0}' and tiles array size '{1}' are mismatch", originInShape.size(),
                          numTiles.size());

        auto inDeclBuff = input.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(inDeclBuff != nullptr, "Can't get input buffer offset");

        Byte inByteOffset{inDeclBuff.byteOffset()};
        Byte outByteOffset{outDeclBuff.byteOffset()};

        auto perClusterShapes = distributedType.getPerClusterComputeShapes();
        VPUX_THROW_UNLESS(checked_cast<int64_t>(perClusterShapes.size()) == numClusters,
                          "Number of shapes '{0}' and clusters '{1}' are mismatch", perClusterShapes.size(),
                          numClusters);

        const auto createNewTypes = [&](vpux::NDTypeInterface innerType) {
            SmallVector<vpux::NDTypeInterface> newTypes(numClusters);
            for (size_t clusterId = 0; clusterId < perClusterShapes.size(); ++clusterId) {
                newTypes[clusterId] = innerType.changeShape(perClusterShapes[clusterId]);
            }

            return newTypes;
        };

        auto inTypes = createNewTypes(innerInputType);
        auto outTypes = createNewTypes(innerOutputType);

        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            auto newInputType = inTypes[clusterId];
            auto newOutType = outTypes[clusterId];

            auto* ctx = getContext();
            const auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, "CMX_NN");
            const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, clusterId)});

            const auto insertBufferDeclarations = [&](vpux::NDTypeInterface newCMXType, Byte cmxOffset,
                                                      vpux::NDTypeInterface newDDRType,
                                                      Byte& ddrOffset) -> std::pair<mlir::Value, mlir::Value> {
                auto cmxBuffer = rewriter.create<VPURT::DeclareBufferOp>(
                        vpurtTask->getLoc(), newCMXType, VPURT::BufferSection::CMX_NN, clusterId, cmxOffset.count());
                auto ddrBuffer = rewriter.create<VPURT::DeclareBufferOp>(vpurtTask->getLoc(), newDDRType,
                                                                         VPURT::BufferSection::DDR, ddrOffset.count());

                ddrOffset += newOutType.getCompactAllocSize();

                return {cmxBuffer, ddrBuffer};
            };

            mlir::Value inputBuffer;
            mlir::Value outBuffer;
            if (inputType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
                newInputType = newInputType.changeMemSpace(symbolAttr);

                auto newDecls = insertBufferDeclarations(newInputType, inByteOffset, newOutType, outByteOffset);
                inputBuffer = newDecls.first;
                outBuffer = newDecls.second;
            } else if (outputType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
                newOutType = newOutType.changeMemSpace(symbolAttr);

                auto newDecls = insertBufferDeclarations(newOutType, outByteOffset, newInputType, inByteOffset);
                inputBuffer = newDecls.second;
                outBuffer = newDecls.first;
            } else {
                VPUX_THROW("One of operands type must have NN_CMX memory space");
            }

            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(rewriter, vpurtTask.waitBarriers(), vpurtTask.updateBarriers(),
                                                  vpurtTask->getLoc(), inputBuffer, outBuffer);
        }
    } else if (mode == VPU::DistributionMode::DUPLICATED) {
        VPUX_THROW_UNLESS(outputType.isa<VPUIP::DistributedBufferType>(),
                          "Only output operand can have DistributedBuffer type");

        SmallVector<int64_t> clusters(numClusters);
        for (int64_t i = 0; i < numClusters; ++i) {
            clusters[i] = i;
        }

        auto cmxBuffer = rewriter.create<VPURT::DeclareBufferOp>(vpurtTask->getLoc(), outDeclBuff.getType(),
                                                                 VPURT::BufferSection::CMX_NN, clusters,
                                                                 outDeclBuff.byteOffset());
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(rewriter, vpurtTask.waitBarriers(), vpurtTask.updateBarriers(),
                                              vpurtTask->getLoc(), input, cmxBuffer);
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
