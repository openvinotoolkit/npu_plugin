//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/constant_fusion.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/range.hpp"

using namespace vpux;

namespace {

//
// PatchFusedConstants
//

// TODO - Consolidate with PatchWeightsTable E#44100
class PatchFusedConstants final : public VPUIP::PatchFusedConstantsBase<PatchFusedConstants> {
public:
    explicit PatchFusedConstants(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    std::vector<int32_t> patchWeightTableInFusedConstant(uint32_t baseOffset, int32_t weightsOffset,
                                                         int32_t sparsityOffset, Const::Content& wtContent,
                                                         bool hasSparsity, int64_t weightsElemByteSize,
                                                         VPUIP::CompressionSchemeAttr weightsCompression);
};

Const::DeclareOp createPatchedDeclareOp(const ArrayRef<char>& patchedValuesArr, uint32_t totalSize,
                                        Const::DeclareOp& weightsTable, Const::SwizzleConstantAttr swizzleConstAttr,
                                        const mlir::Type patchedConstType) {
    mlir::OpBuilder builder(weightsTable);
    uint32_t elementSizeBytes = patchedConstType.getIntOrFloatBitWidth() / CHAR_BIT;
    VPUX_THROW_WHEN(totalSize % elementSizeBytes != 0,
                    "createPatchedDeclareOp: totalSize {0}, should be integer multiplicity of patchedValuesBuf element "
                    "size {1} ",
                    totalSize, elementSizeBytes);
    SmallVector<int64_t> patchedConstShape({1, 1, 1, totalSize / elementSizeBytes});

    mlir::RankedTensorType tensorType = mlir::RankedTensorType::get(patchedConstShape, patchedConstType);
    mlir::ElementsAttr value = mlir::DenseElementsAttr::getFromRawBuffer(tensorType, patchedValuesArr);

    auto contentAttr = Const::ContentAttr::get(value);
    if (swizzleConstAttr) {
        contentAttr = Const::ContentAttr::addTransformation(contentAttr, swizzleConstAttr);
    }

    return builder.create<Const::DeclareOp>(weightsTable->getLoc(), weightsTable.getType(), contentAttr);
}

std::vector<int32_t> PatchFusedConstants::patchWeightTableInFusedConstant(
        uint32_t baseOffset, int32_t weightsOffset, int32_t sparsityOffset, Const::Content& wtContent, bool hasSparsity,
        int64_t weightsElemByteSize, VPUIP::CompressionSchemeAttr weightsCompression) {
    int32_t numWTEntries, weightPtrStep = 0, sparsityPtrStep = 0, sparsityPtr;
    constexpr int32_t numElemPerOC = static_cast<size_t>(VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC);

    // TODO: Set the correct offsets for Multi-Cluster E#42447
    SmallVector<int32_t> offsets{0};

    auto weightsPtr = (weightsOffset != 0) ? static_cast<int32_t>(baseOffset + weightsOffset) : 0;
    // If the activation window is present for the layer, use the activation offset otherwise set it to 0xFFFFFF
    if (hasSparsity) {
        sparsityPtr = (sparsityOffset != 0) ? static_cast<int32_t>(baseOffset + sparsityOffset) : 0;
    } else {
        sparsityPtr = VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY;
    }

    auto totalSizeBytes = static_cast<uint32_t>(wtContent.getType().getTotalAllocSize().count());

    // Convert original content to i32 for patching WT Entries
    std::vector<int32_t> wtValuesI32Patched = wtContent.vec<int32_t>();

    //
    // Since the order of fusion is constant it can be thought as a contiguous array with
    // offsets for various constants which can be used to figure out the num of WT entries
    //
    //       <-------WT Entries---->
    //      [_______________________|____________________________|__________]
    //     Base                    Base                         Base
    //      Fused Const/WT          Weights                      Activation Win
    //
    // We only need to patch these entries, rest all are weights/activation or both
    // When both weights and activation are present i.e. the op is CMCONV
    // Number of WT entries would be base address of weights minus base address of fused const
    if (weightsPtr > 0 && sparsityPtr > 0) {
        numWTEntries = (weightsPtr - baseOffset) / sizeof(int32_t);
    }
    // When only weights are present i.e. In case of CONV op number of WT entries is
    // base address of weights minus base address of fused const as sparsity would be 0xFF
    else if (weightsPtr > 0) {
        numWTEntries = (weightsPtr - baseOffset) / sizeof(int32_t);
    }
    // When only activation is present i.e. Op is MaxPool
    // Number of WT entries is base address of activation window minus base address of fused const
    else if (sparsityPtr > 0) {
        numWTEntries = (sparsityPtr - baseOffset) / sizeof(int32_t);
    }
    // When only WT is present
    else {
        numWTEntries = totalSizeBytes / sizeof(int32_t);
    }

    if (numWTEntries >= numElemPerOC * 2) {
        weightPtrStep = wtValuesI32Patched[1 * numElemPerOC + 0] - wtValuesI32Patched[0 * numElemPerOC + 0];
        sparsityPtrStep = wtValuesI32Patched[1 * numElemPerOC + 1] - wtValuesI32Patched[0 * numElemPerOC + 1];
    }

    const int64_t OC = checked_cast<int64_t>(numWTEntries / numElemPerOC);
    const int64_t numClusters = checked_cast<int64_t>(offsets.size());

    // In case all clusters have the same channel offsets, the weights are not segmented
    const auto areWeightsSegmented =
            std::adjacent_find(offsets.begin(), offsets.end(), std::not_equal_to<>()) != offsets.end();

    const auto isNewCluster = [&](const int64_t oc, const int64_t currentClusterIdx) -> bool {
        return areWeightsSegmented && (currentClusterIdx + 1) < numClusters && oc >= offsets[currentClusterIdx + 1];
    };

    SmallVector<int64_t> weightsPtrSteps(OC);
    if (weightsCompression != nullptr) {
        const auto numElems = to_small_vector(weightsCompression.getNumElems().getValues<int64_t>());
        VPUX_THROW_UNLESS(numElems.size() == static_cast<size_t>(OC),
                          "Invalid weights compression with {0} elements for {1} channels", numElems.size(), OC);
        const auto alignment = (weightsCompression.getAlignment() != nullptr)
                                       ? weightsCompression.getAlignment().getInt()
                                       : VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT;

        int64_t weightsPtrOffset = 0;
        for (int64_t oc = 0, clusterIdx = 0; oc < OC; ++oc) {
            if (isNewCluster(oc, clusterIdx)) {
                clusterIdx++;
                weightsPtrOffset = 0;
            }
            weightsPtrSteps[oc] = weightsPtrOffset;
            const auto weightSetSize = (numElems[oc] * weightsElemByteSize);
            weightsPtrOffset += alignValUp<int64_t>(weightSetSize, alignment);
        }
    } else {
        for (int64_t oc = 0, clusterIdx = 0; oc < OC; ++oc) {
            if (isNewCluster(oc, clusterIdx)) {
                clusterIdx++;
            }
            weightsPtrSteps[oc] = weightPtrStep * (oc - offsets[clusterIdx]);
        }
    }

    for (int64_t oc = 0, clusterIdx = 0; oc < OC; ++oc) {
        if (isNewCluster(oc, clusterIdx)) {
            clusterIdx++;
        }
        const auto wtInd = oc * numElemPerOC;

        wtValuesI32Patched[wtInd + 0] = checked_cast<int32_t>(weightsPtr + weightsPtrSteps[oc]);

        if (wtValuesI32Patched[wtInd + 1] != VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY) {
            wtValuesI32Patched[wtInd + 1] =
                    checked_cast<int32_t>(sparsityPtr + (oc - offsets[clusterIdx]) * sparsityPtrStep);
        }
    }

    return wtValuesI32Patched;
}

//
// safeRunOnFunc
//

void PatchFusedConstants::safeRunOnFunc() {
    auto funcOp = getOperation();

    funcOp.walk([&](vpux::VPUIP::NCEClusterTaskOp nceOp) {
        if (!nceOp->hasAttr(vpux::ConstantFusing::constantsFused)) {
            return;
        }
        _log.trace("Patch fused constant for NCEOp - '{0}'", nceOp->getLoc());

        VPUIP::NNDMAOp constDmaOp;
        VPUIP::StaticAllocOp staticAllocOp;

        auto weights = nceOp.getWeights();
        auto weightsSM = nceOp.getWeightsSparsityMap();
        auto activationWindow = nceOp.getActivationWindow();
        bool hasSparsity = (weightsSM != nullptr) || (activationWindow != nullptr);
        auto weightTable = nceOp.getWeightTable();

        mlir::Type fusedConstantElementType =
                weightTable.getDefiningOp()->getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();
        unsigned fusedConstantElementSizeBytes = fusedConstantElementType.getIntOrFloatBitWidth() / CHAR_BIT;

        uint32_t weightsOffset =
                vpux::ConstantFusing::getOffsetForConstant(nceOp, weights) * fusedConstantElementSizeBytes;
        uint32_t weightsSMOffset =
                vpux::ConstantFusing::getOffsetForConstant(nceOp, weightsSM) * fusedConstantElementSizeBytes;
        uint32_t actOffset =
                vpux::ConstantFusing::getOffsetForConstant(nceOp, activationWindow) * fusedConstantElementSizeBytes;
        uint32_t sparsityOffset = (weightsSM != nullptr) ? weightsSMOffset : actOffset;

        int64_t weightsElemByteSize = 1;
        VPUIP::CompressionSchemeAttr weightsCompression = nullptr;
        if (weights != nullptr) {
            weightsElemByteSize = getElemTypeSize(weights.getType()).to<Byte>().count();
            weightsCompression = VPUIP::getCompressionSchemeAttr(weights.getType());
        }
        mlir::Operation* op = nullptr;
        auto weightsTable = vpux::ConstantFusing::getConstAndDma(nceOp, weightTable, &op);
        constDmaOp = mlir::dyn_cast_or_null<VPUIP::NNDMAOp>(op);
        VPUX_THROW_UNLESS(weightsTable != nullptr, "Couldn't find Weight Table Declare Op");
        VPUX_THROW_UNLESS(constDmaOp != nullptr, "Couldn't find Dma Op for Weight Table");

        auto contentAttr = weightsTable.getContentAttr();

        Const::ContentAttr newContentAttr = Const::ContentAttr::get(contentAttr.getBaseContent());

        // Check if constant had swizzling transformation. If yes then patching should be applied
        // without swizzling. Swizzling transformation will be reattached after patching of
        // weights table is completed
        Const::SwizzleConstantAttr swizzleConstAttr;
        for (auto attr : contentAttr.getTransformations()) {
            swizzleConstAttr = attr.dyn_cast_or_null<Const::SwizzleConstantAttr>();
            if (swizzleConstAttr != nullptr) {
                _log.nest().trace("Swizzling transformation detected");
                continue;
            }
            newContentAttr = Const::ContentAttr::addTransformation(newContentAttr, attr);
        }

        Const::Content wtContent = newContentAttr.fold();
        auto totalSize = static_cast<uint32_t>(wtContent.getType().getTotalAllocSize().count());

        // Locate root buffer that will be a place where fused constant is allocated
        // Address of this buffer is the offset that should be used for patching
        // weights table content, no matter if such constant was previously spilled or not
        // as only final location in CMX matters
        vpux::ValueSourceInfo aliasInfo(weightTable);
        auto rootBuffers = aliasInfo.getRoots(weightTable);
        VPUX_THROW_UNLESS(rootBuffers.size() == 1, "Value expected to have only one root. Got {1}", rootBuffers.size());
        const auto rootBuffer = *rootBuffers.begin();

        uint32_t baseOffset = 0;
        if (auto staticAllocOp = rootBuffer.getDefiningOp<VPUIP::StaticAllocOp>()) {
            baseOffset = static_cast<uint32_t>(staticAllocOp.getOffset());
        } else if (auto declareBuffer = rootBuffer.getDefiningOp<VPURT::DeclareBufferOp>()) {
            baseOffset = static_cast<uint32_t>(declareBuffer.getByteOffset());
        } else {
            VPUX_THROW("Unsupported declare op for buffer- '{0}'", rootBuffer);
        }

        auto wtValuesI32Patched = patchWeightTableInFusedConstant(baseOffset, weightsOffset, sparsityOffset, wtContent,
                                                                  hasSparsity, weightsElemByteSize, weightsCompression);

        auto wtValuesI32PatchedArr =
                ArrayRef(reinterpret_cast<char*>(wtValuesI32Patched.data()), wtValuesI32Patched.size() * 4);

        Const::DeclareOp newOp = createPatchedDeclareOp(wtValuesI32PatchedArr, totalSize, weightsTable,
                                                        swizzleConstAttr, fusedConstantElementType);

        constDmaOp.setOperand(0, newOp);

        if (weightsTable->getUses().empty()) {
            weightsTable.erase();
        }
    });
}

}  // namespace

//
// createPatchFusedConstants
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createPatchFusedConstantsPass(Logger log) {
    return std::make_unique<PatchFusedConstants>(log);
}
