//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/constant_fusion.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/compiler/utils/logging.hpp"
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

Const::DeclareOp createPatchedDeclareOp(std::vector<uint8_t>& patchedValuesBuf, uint32_t totalSize,
                                        Const::DeclareOp& weightsTable, Const::SwizzleConstantAttr swizzleConstAttr) {
    mlir::OpBuilder builder(weightsTable);
    SmallVector<int64_t> patchedConstShape({1, 1, 1, totalSize});
    auto patchedConstElemType = getUInt8Type(builder.getContext());

    // create type
    const auto patchedTensorType = mlir::RankedTensorType::get(patchedConstShape, patchedConstElemType);

    auto rawWeights = reinterpret_cast<char*>(patchedValuesBuf.data());
    const auto rawWeightsBuffer = makeArrayRef(rawWeights, patchedValuesBuf.size() * sizeof(uint8_t));

    mlir::ElementsAttr value = mlir::DenseElementsAttr::getFromRawBuffer(patchedTensorType, rawWeightsBuffer);

    auto contentAttr = Const::ContentAttr::get(value);
    if (swizzleConstAttr) {
        contentAttr = Const::ContentAttr::addTransformation(contentAttr, swizzleConstAttr);
    }

    return builder.create<Const::DeclareOp>(weightsTable->getLoc(), weightsTable.getType(), contentAttr);
}

std::vector<int32_t> PatchFusedConstants::patchWeightTableInFusedConstant(
        uint32_t baseOffset, int32_t weightsOffset, int32_t sparsityOffset, Const::Content& wtContent, bool hasSparsity,
        int64_t weightsElemByteSize, VPUIP::CompressionSchemeAttr weightsCompression) {
    int32_t totalSize, numWTEntries, weightPtrStep = 0, sparsityPtrStep = 0, sparsityPtr;
    constexpr int32_t numElemPerOC = static_cast<size_t>(VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC);
    std::vector<int32_t> wtValuesI32;

    // TODO: Set the correct offsets for Multi-Cluster E#42447
    SmallVector<int32_t> offsets{0};

    auto weightsPtr = (weightsOffset != 0) ? static_cast<int32_t>(baseOffset + weightsOffset) : 0;
    // If the activation window is present for the layer, use the activation offset otherwise set it to 0xFFFFFF
    if (hasSparsity) {
        sparsityPtr = (sparsityOffset != 0) ? static_cast<int32_t>(baseOffset + sparsityOffset) : 0;
    } else {
        sparsityPtr = VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY;
    }

    auto wtValues = wtContent.getValues<uint8_t>();
    totalSize = static_cast<int32_t>(wtValues.size());

    // Convert Original Values to i32 (This will convert fused constant to i32)
    const auto wtSizeI32 = wtValues.size() / 4;
    std::vector<int32_t> wtValuesI32Patched(wtSizeI32);
    wtValuesI32.reserve(wtSizeI32);

    // convert U8 to I32 for patching
    vpux::ConstantFusing::convertInputToI32(wtValues, wtValuesI32);

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
    // Numbe rof WT entries is base address of activation window minus base address of fused const
    else if (sparsityPtr > 0) {
        numWTEntries = (sparsityPtr - baseOffset) / sizeof(int32_t);
    }
    // When only WT is present
    else {
        numWTEntries = totalSize / sizeof(int32_t);
    }

    if (numWTEntries >= numElemPerOC * 2) {
        weightPtrStep = wtValuesI32[1 * numElemPerOC + 0] - wtValuesI32[0 * numElemPerOC + 0];
        sparsityPtrStep = wtValuesI32[1 * numElemPerOC + 1] - wtValuesI32[0 * numElemPerOC + 1];
    }

    const int64_t OC = checked_cast<int64_t>(numWTEntries / numElemPerOC);
    const int64_t numClusters = checked_cast<int64_t>(offsets.size());

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
            if ((clusterIdx + 1) < numClusters && oc >= offsets[clusterIdx + 1]) {
                clusterIdx++;
                weightsPtrOffset = 0;
            }
            weightsPtrSteps[oc] = weightsPtrOffset;
            const auto weightSetSize = (numElems[oc] * weightsElemByteSize);
            weightsPtrOffset += alignValUp<int64_t>(weightSetSize, alignment);
        }
    } else {
        for (int64_t oc = 0, clusterIdx = 0; oc < OC; ++oc) {
            if ((clusterIdx + 1) < numClusters && oc >= offsets[clusterIdx + 1]) {
                clusterIdx++;
            }
            weightsPtrSteps[oc] = weightPtrStep * (oc - offsets[clusterIdx]);
        }
    }

    for (int64_t oc = 0, clusterIdx = 0; oc < OC; ++oc) {
        if ((clusterIdx + 1) < numClusters && oc >= offsets[clusterIdx + 1]) {
            clusterIdx++;
        }
        const auto wtInd = oc * numElemPerOC;

        wtValuesI32Patched[wtInd + 0] = checked_cast<int32_t>(weightsPtr + weightsPtrSteps[oc]);

        wtValuesI32Patched[wtInd + 1] = wtValuesI32[wtInd + 1];
        if (wtValuesI32[wtInd + 1] != VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY) {
            wtValuesI32Patched[wtInd + 1] =
                    checked_cast<int32_t>(sparsityPtr + (oc - offsets[clusterIdx]) * sparsityPtrStep);
        }

        wtValuesI32Patched[wtInd + 2] = wtValuesI32[wtInd + 2];
        wtValuesI32Patched[wtInd + 3] = wtValuesI32[wtInd + 3];
    }
    // Fill in the remaing values as is (For activation and weights if present)
    for (auto i = numWTEntries; i < (int32_t)wtValuesI32.size(); ++i) {
        wtValuesI32Patched[i] = wtValuesI32[i];
    }

    return wtValuesI32Patched;
}

//
// safeRunOnFunc
//

void PatchFusedConstants::safeRunOnFunc() {
    auto funcOp = getOperation();
    auto& aliasInfo = getAnalysis<AliasesInfo>();

    funcOp.walk([&](vpux::VPUIP::NCEClusterTaskOp nceOp) {
        if (!nceOp->hasAttr(vpux::ConstantFusing::constantsFused)) {
            return;
        }
        _log.trace("Patch fused constant for NCEOp - '{0}'", nceOp->getLoc());

        VPUIP::CopyOp constCopyOp;
        VPUIP::StaticAllocOp staticAllocOp;
        std::vector<uint8_t> wtValuesU8;

        auto weights = nceOp.weights();
        auto weightsSM = nceOp.weights_sparsity_map();
        auto activationWindow = nceOp.activation_window();
        bool hasSparsity = (weightsSM != nullptr) || (activationWindow != nullptr);
        auto weightTable = VPUIP::getTopBufferOfNCEClusterTiling(nceOp, nceOp.weight_table());

        uint32_t weightsOffset = vpux::ConstantFusing::getOffsetForConstant(nceOp, weights);
        uint32_t weightsSMOffset = vpux::ConstantFusing::getOffsetForConstant(nceOp, weightsSM);
        uint32_t actOffset = vpux::ConstantFusing::getOffsetForConstant(nceOp, activationWindow);
        uint32_t sparsityOffset = (weightsSM != nullptr) ? weightsSMOffset : actOffset;

        int64_t weightsElemByteSize = 1;
        VPUIP::CompressionSchemeAttr weightsCompression = nullptr;
        if (weights != nullptr) {
            weightsElemByteSize = getElemTypeSize(weights.getType()).to<Byte>().count();
            weightsCompression = VPUIP::getCompressionSchemeAttr(weights.getType());
        }
        auto weightsTable = vpux::ConstantFusing::getConstAndCopyOp(nceOp, weightTable, constCopyOp);
        VPUX_THROW_UNLESS(weightsTable != nullptr, "Couldn't find Weight Table Declare Op");

        auto contentAttr = weightsTable.getContentAttr();

        Const::ContentAttr newContentAttr;
        auto baseContent = contentAttr.getBaseContent();

        if (auto denseBaseAttr = baseContent.dyn_cast<mlir::DenseElementsAttr>()) {
            newContentAttr = Const::ContentAttr::get(denseBaseAttr);
        } else if (auto opaqueBaseAttr = baseContent.dyn_cast<Const::OpaqueElementsAttr>()) {
            newContentAttr = Const::ContentAttr::get(opaqueBaseAttr);
        } else {
            VPUX_THROW("Got unsupported 'baseContent' in 'ContentAttr'");
        }

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
        auto wtValues = wtContent.getValues<uint8_t>();
        auto totalSize = static_cast<uint32_t>(wtValues.size());

        VPUX_THROW_UNLESS(constCopyOp != nullptr, "Couldn't find Copy Op for Weight Table");

        // Locate root buffer that will be a place where fused constant is allocated
        // Address of this buffer is the offset that should be used for patching
        // weights table content, no matter if such constant was previously spilled or not
        // as only final location in CMX matters
        const auto rootBuffers = aliasInfo.getRoots(weightTable);
        VPUX_THROW_UNLESS(rootBuffers.size() == 1, "Value expected to have only one root. Got {1}", rootBuffers.size());
        const auto rootBuffer = *rootBuffers.begin();

        uint32_t baseOffset = 0;
        if (auto staticAllocOp = rootBuffer.getDefiningOp<VPUIP::StaticAllocOp>()) {
            baseOffset = static_cast<uint32_t>(staticAllocOp.offset());
        } else if (auto declareBuffer = rootBuffer.getDefiningOp<VPURT::DeclareBufferOp>()) {
            baseOffset = static_cast<uint32_t>(declareBuffer.getByteOffset());
        } else {
            VPUX_THROW("Unsupported declare op for buffer- '{0}'", rootBuffer);
        }

        auto wtValuesI32Patched = patchWeightTableInFusedConstant(baseOffset, weightsOffset, sparsityOffset, wtContent,
                                                                  hasSparsity, weightsElemByteSize, weightsCompression);

        for (auto i : wtValuesI32Patched) {
            vpux::ConstantFusing::convertToU8<int32_t>(i, wtValuesU8);
        }

        auto newOp = createPatchedDeclareOp(wtValuesU8, totalSize, weightsTable, swizzleConstAttr);

        if (auto tilingOp = constCopyOp->getParentOfType<VPUIP::NCEClusterTilingOp>()) {
            tilingOp.setOperand(0, newOp);
        } else {
            constCopyOp.setOperand(0, newOp);
        }

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
