//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/constant_fusion.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

namespace {

//
// PatchWeightsTablePass
//

class PatchWeightsTablePass final : public VPUIP::PatchWeightsTableBase<PatchWeightsTablePass> {
public:
    explicit PatchWeightsTablePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    uint64_t getPointer(mlir::Value value, uint64_t defaultValue);
    SmallVector<int32_t> getWeightsPerClusterPointers(VPUIP::NCEClusterTaskOp nceOp, int64_t numCluster);
    void patchWeightsTable(Const::DeclareOp cstOp, mlir::Operation* cstLoadOp, VPUIP::NCEClusterTaskOp nceOp,
                           VPURT::DeclareBufferOp wtDecBuf, int64_t weightsElemBitSize,
                           VPUIP::CompressionSchemeAttr weightsCompression);
    mlir::Operation* getLoadOpForDstBuffer(mlir::Value dstBuffer);
};

//
// safeRunOnFunc
//

void PatchWeightsTablePass::safeRunOnFunc() {
    auto funcOp = getOperation();
    // For each nceOp.weight_table find related DeclareBufferOp. Next find dmaOp which
    // fills the buffer. DmaOp's input is expected to be Const::DeclareOp which
    // should be modified by adding relocateWeightTable transformation.
    funcOp.walk([this](vpux::VPUIP::NCEClusterTaskOp nceOp) {
        // Don't need to run this when the constants are fused
        // The patch weight table is done in patch_fused_constants.cpp
        // However some of the Op's are skipped e.g. the cases when the source for weight
        // is activation tensor etc which would be delagated to this pass for patching
        // E#44100 - Consolidate weight table and patching logic
        if (nceOp->hasAttr(vpux::ConstantFusing::constantsFused)) {
            nceOp->removeAttr(vpux::ConstantFusing::constantsFused);
            return;
        }

        auto wTable = nceOp.getWeightTable();
        if (wTable == nullptr) {
            return;
        }
        _log.trace("WeightTable identified for operation '{0}'", nceOp->getLoc());
        auto wtDecBuf = wTable.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(wtDecBuf != nullptr, "DeclareBufferOp expected as a weight_table parent");

        mlir::Value inputBuffer;

        // Get operation that loads weights table to CMX for NCE Task
        auto* cstLoadOp = getLoadOpForDstBuffer(wtDecBuf.getResult());
        VPUX_THROW_UNLESS(cstLoadOp != nullptr, "Operation loading weight table expected, but not located");

        // Get the constant definition op whose content will be patched
        inputBuffer = cstLoadOp->getOperand(0);
        auto cstOp = inputBuffer.getDefiningOp<Const::DeclareOp>();

        // In case weights table was spilled there can be a sequence of DMAs
        // Need to resolve it and update this DMA to have const as input directly
        while (cstOp == nullptr) {
            cstLoadOp = getLoadOpForDstBuffer(inputBuffer);
            VPUX_THROW_UNLESS(cstLoadOp != nullptr, "Next DMA op as source operation expected");

            inputBuffer = cstLoadOp->getOperand(0);
            cstOp = inputBuffer.getDefiningOp<Const::DeclareOp>();
        }

        _log.nest().trace("Operation loading weight table '{0}' '{1}'", cstLoadOp->getName(), cstLoadOp->getLoc());

        VPUX_THROW_UNLESS(cstOp != nullptr, "Constant expected as DMA input for weights table - {0}", *cstLoadOp);

        int64_t weightsElemBitSize = CHAR_BIT;
        VPUIP::CompressionSchemeAttr weightsCompression = nullptr;
        if (nceOp.getWeights() != nullptr) {
            weightsElemBitSize = getElemTypeSize(nceOp.getWeights().getType()).count();
            weightsCompression = VPUIP::getCompressionSchemeAttr(nceOp.getWeights().getType());
        }

        // On top of existing transformation a new transformation is added to the content attribute
        // of weight table const. The new transformation will patch offsets in this constant
        // with sparsity and weights pointers. The pointers are passed as  parameters of the
        // new transformation.
        _log.nest().trace("Constant for patching '{0}'", cstOp->getLoc());
        patchWeightsTable(cstOp, cstLoadOp, nceOp, wtDecBuf, weightsElemBitSize, weightsCompression);
    });
}

// Find a DMA operation that loads data into a given buffer
mlir::Operation* PatchWeightsTablePass::getLoadOpForDstBuffer(mlir::Value dstBuffer) {
    for (const auto& user : dstBuffer.getUsers()) {
        auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(user);
        if ((dmaOp != nullptr) && (dmaOp.getOutputBuff() == dstBuffer)) {
            return dmaOp.getOperation();
        }
    }
    return nullptr;
}

void PatchWeightsTablePass::patchWeightsTable(Const::DeclareOp cstOp, mlir::Operation* cstLoadOp,
                                              VPUIP::NCEClusterTaskOp nceOp, VPURT::DeclareBufferOp wtDecBuf,
                                              int64_t weightsElemBitSize,
                                              VPUIP::CompressionSchemeAttr weightsCompression) {
    // Retrieve sparsity and weight pointers which have correct values as they are already allocated
    // by the memory scheduler
    auto sparsityMap = nceOp.getWeightsSparsityMap();
    if (sparsityMap == nullptr) {
        sparsityMap = nceOp.getActivationWindow();
    }
    uint64_t sparsityBasePtr = getPointer(sparsityMap, VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY);

    SmallVector<int64_t> offsets;
    if (auto distributedType = wtDecBuf.getType().dyn_cast<VPUIP::DistributedBufferType>()) {
        auto distributionAttr = distributedType.getDistribution();
        const auto numClusters = distributionAttr.getNumClusters().getInt();
        const auto perClusterShapeOffsets = distributedType.getPerClusterMemoryShapeOffsets();
        VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                          "Number of shape offsets '{0}' and clusters '{1}' are mismatch",
                          perClusterShapeOffsets.size(), numClusters);

        for (auto clusterOffsets : perClusterShapeOffsets) {
            offsets.push_back(clusterOffsets[Dims4D::Filter::OC]);
        }
    } else {
        offsets.push_back(0);
    }
    auto weightsPerClusterPtrs = getWeightsPerClusterPointers(nceOp, offsets.size());
    // Extract content attrib with existing transformations
    auto origConstAttr = cstOp.getContentAttr();
    // Create new attribute based on existing one by adding new relocateWeightTable
    // transformation
    auto newConstAttr = origConstAttr.relocateWeightsTablePointers(
            weightsPerClusterPtrs, sparsityBasePtr, ShapeRef(offsets), weightsElemBitSize, weightsCompression);
    mlir::OpBuilder builder(cstOp);

    // Create new DeclareOp with the new content attribute and replace the old DeclareOp
    // with it
    auto newConstOp = builder.create<Const::DeclareOp>(cstOp.getLoc(), cstOp.getOutput().getType(), newConstAttr);
    cstLoadOp->setOperand(0, newConstOp.getOutput());
    if (cstOp->getUses().empty()) {
        cstOp.erase();
    }
}  // namespace

uint64_t PatchWeightsTablePass::getPointer(mlir::Value value, uint64_t defaultValue) {
    if (value == nullptr) {
        return defaultValue;
    }
    auto valueDeclareBuffer = mlir::dyn_cast<VPURT::DeclareBufferOp>(value.getDefiningOp());
    if (valueDeclareBuffer == nullptr) {
        return defaultValue;
    }
    return valueDeclareBuffer.getByteOffset();
}

SmallVector<int32_t> PatchWeightsTablePass::getWeightsPerClusterPointers(VPUIP::NCEClusterTaskOp nceOp,
                                                                         int64_t numCluster) {
    auto weights = nceOp.getWeights();
    uint64_t weightsBasePointer = getPointer(weights, 0);

    if (weights == nullptr || !weights.getType().isa<VPUIP::DistributedBufferType>()) {
        return SmallVector<int32_t>(numCluster, static_cast<int32_t>(weightsBasePointer));
    }
    auto distributedType = weights.getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto distributionAttr = distributedType.getDistribution();
    auto mode = distributionAttr.getMode().getValue();
    if (mode != (VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED)) {
        // For distribution modes except Duplicated|Segmented, the weights on every cluster is supposed to be symmetric.
        // So just return the base pointer for each weights
        return SmallVector<int32_t>(numCluster, static_cast<int32_t>(weightsBasePointer));
    }

    // For Duplicated|Segmented weights, weights should be non const for ops like MatMul. It's supposed to
    // be segmented on N, while the entire data is copied to all CMXs. So the weight's offset need to updated
    // accordingly since it's not symmetric.
    SmallVector<int32_t> weightsPerClusterPtrs;
    VPUX_THROW_UNLESS(nceOp.getWeightsSparsityMap() == nullptr,
                      "Weight sparisity map is found, weights is supposed to be non const at '{0}'", nceOp->getLoc());
    const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
    const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
    VPUX_THROW_UNLESS(axis == Dims4D::Act::N.ind(), "Invalid Tile dim, get {0}, expect tiling on N.", axis);
    const auto perClusterShapeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    for (auto clusterOffsets : perClusterShapeOffsets) {
        auto perClusterOffset = Byte(clusterOffsets[Dim(axis)] * distributedType.getStrides()[Dim(axis)]).count();
        weightsPerClusterPtrs.push_back(static_cast<int32_t>(weightsBasePointer + perClusterOffset));
    }
    return weightsPerClusterPtrs;
}
}  // namespace

//
// createPatchWeightsTablePass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createPatchWeightsTablePass(Logger log) {
    return std::make_unique<PatchWeightsTablePass>(log);
}
