//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"

#include "vpux/compiler/core/aliases_info.hpp"

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/PatternMatch.h>

#include <mlir/Pass/PassManager.h>

using namespace vpux;

namespace {

//
// AlignmentForSwizzling
//

class AlignmentForSwizzling final : public VPUIP::AlignmentForSwizzlingBase<AlignmentForSwizzling> {
public:
    explicit AlignmentForSwizzling(bool enableWeightSwizzling, bool enableActivationSwizzling, Logger log)
            : _enableWeightSwizzling(enableWeightSwizzling), _enableActivationSwizzling(enableActivationSwizzling) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    bool _enableWeightSwizzling;
    bool _enableActivationSwizzling;
    void safeRunOnFunc() final;
    int64_t getAlignmentForSwizzling(int64_t swizzlingKey);
    void activationBufferSwizzling(mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp nceOp);
    void constantBufferSwizzling(mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp nceOp, mlir::Value cst);
    void addPaddingAttrToBuffers(mlir::OpBuilder& builder, VPUIP::CopyOp& copyOp, mlir::memref::AllocOp& allocOp);
    template <typename InAllocOp, typename OutAllocOp>
    void addSwizzlingAttributesToBuffer(mlir::OpBuilder& builder, InAllocOp inAllocOp);
};

//
//  Function creates a buffer and returns the AllocOp
//  TODO - Update the utility for multiclustering case E#45641
//
mlir::memref::AllocOp createCMXBuffer(vpux::NDTypeInterface ndType, mlir::Location loc, mlir::OpBuilder& builder) {
    vpux::IndexedSymbolAttr memKindAttr =
            IndexedSymbolAttr::get(builder.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), 0);
    const auto dataTypeCMX = ndType.changeMemSpace(memKindAttr);
    return builder.create<mlir::memref::AllocOp>(loc, dataTypeCMX.cast<mlir::MemRefType>());
}

int64_t getRequiredPadding(Const::DeclareOp decOp) {
    auto type = decOp.getType().cast<NDTypeInterface>();
    auto totalSize = type.getTotalAllocSize().count();
    auto elementSize = type.getElemTypeSize().count();
    ShapeRef shapeRef = type.getShape();
    if (totalSize % VPUIP::SWIZZLING_SIZE_ALIGNMENT == 0) {
        return 0;
    }

    auto newSize = totalSize + (VPUIP::SWIZZLING_SIZE_ALIGNMENT - (totalSize % VPUIP::SWIZZLING_SIZE_ALIGNMENT));
    auto totalElements = (newSize * CHAR_BIT) / elementSize;
    auto paddingValue = shapeRef.front();
    auto padding = abs(((totalElements * paddingValue) / type.getNumElements()) - paddingValue);
    return padding;
}

// 2 Major checks are required, the rest are just null checks
// 1. Are weights constant <- These buffers are swizzled as part of activation swizzling
//                            so avoid double swizzling here
// 2. Is padding required <- This case will be enabled with E#48057
bool canSwizzleWeights(VPUIP::NCEClusterTaskOp nceOp) {
    auto isTiled = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>() != nullptr;
    auto weights = VPUIP::getTopBufferOfNCEClusterTiling(nceOp, nceOp.weights());
    auto weightTable = VPUIP::getTopBufferOfNCEClusterTiling(nceOp, nceOp.weight_table());

    Const::DeclareOp decOp = nullptr;

    // Swizzling for ELTWISE is handled with activation swizzling
    if (weights == nullptr || weightTable == nullptr || nceOp.task_type() == VPUIP::NCETaskType::ELTWISE) {
        return false;
    }

    if (isTiled) {
        auto bufferTilingOp = weights.getDefiningOp<VPUIP::NCEClusterTilingOp>();
        auto copyOp = bufferTilingOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
        if (copyOp == nullptr) {
            return false;
        }

        // Weights from output of another layer are handled with DPU -> DPU buffers skip any weight swizzling in
        // such case here
        decOp = bufferTilingOp.inputs()[0].getDefiningOp<Const::DeclareOp>();
        if (decOp == nullptr) {
            return false;
        }

        auto distributedBufferType = bufferTilingOp.output_buffs()[0].getDefiningOp<VPURT::AllocDistributed>();
        if (distributedBufferType == nullptr) {
            return false;
        }

        auto inputType = distributedBufferType.getType().dyn_cast<VPUIP::DistributedBufferType>();
        if (inputType.getDistribution().mode().getValue() != VPU::DistributionMode::DUPLICATED) {
            return false;
        }

        // Temporarily disable swizzling for WT buffer size not multiple of 512
        // TODO E#48057 Resize these buffers
        auto wtBufferTilingOp = weightTable.getDefiningOp<VPUIP::NCEClusterTilingOp>();
        if (wtBufferTilingOp == nullptr) {
            return false;
        }

        auto wtDecOp = wtBufferTilingOp.inputs()[0].getDefiningOp<Const::DeclareOp>();
        if (getRequiredPadding(wtDecOp)) {
            return false;
        }

    } else {
        auto copyOp = weights.getDefiningOp<VPUIP::CopyOp>();
        if (copyOp == nullptr) {
            return false;
        }

        // Weights from output of another layer are handled with DPU -> DPU buffers skip any weight swizzling in
        // such case here
        if (!mlir::isa<Const::DeclareOp>(copyOp.input().getDefiningOp())) {
            return false;
        }

        decOp = copyOp.input().getDefiningOp<Const::DeclareOp>();
        auto allocOp = copyOp.output_buff().getDefiningOp<mlir::memref::AllocOp>();
        if (allocOp == nullptr) {
            return false;
        }
    }

    auto requiredPadding = getRequiredPadding(decOp);
    if (requiredPadding) {
        // TODO E#48057 Resize these buffers
        return false;
    }
    return true;
}

template <typename InAllocOp, typename OutAllocOp>
void AlignmentForSwizzling::addSwizzlingAttributesToBuffer(mlir::OpBuilder& builder, InAllocOp inAllocOp) {
    auto swizzlingAttr = builder.getI64IntegerAttr(VPUIP::SWIZZLING_KEY_5);
    auto alignmentAttr = builder.getI64IntegerAttr(getAlignmentForSwizzling(VPUIP::SWIZZLING_KEY_5));

    builder.setInsertionPoint(inAllocOp);

    auto origLoc = inAllocOp->getLoc();
    auto newLoc = appendLoc(origLoc, "_alloc_swizzling");
    auto origType = inAllocOp.getType().template cast<vpux::NDTypeInterface>();
    auto outAllocOp = builder.create<OutAllocOp>(newLoc, origType, alignmentAttr, swizzlingAttr);

    inAllocOp->replaceAllUsesWith(outAllocOp);
    inAllocOp->erase();
}

void AlignmentForSwizzling::addPaddingAttrToBuffers(mlir::OpBuilder& builder, VPUIP::CopyOp& copyOp,
                                                    mlir::memref::AllocOp& allocOp) {
    builder.setInsertionPoint(copyOp);

    auto decOp = copyOp.input().getDefiningOp<Const::DeclareOp>();
    auto requiredPadding = getRequiredPadding(decOp);

    if (requiredPadding) {
        const auto inputType = decOp.getType().cast<NDTypeInterface>();
        const auto origOutputType = decOp.output().getType().cast<vpux::NDTypeInterface>();
        SmallVector<int64_t> origShape(inputType.getShape().begin(), inputType.getShape().end());

        auto contentAttr = decOp.contentAttr();
        // TODO : E#48057 Currently this logic only works for weight table resize
        auto padAttr = contentAttr.padWithZero({0, 0, 0, 0}, {requiredPadding, 0, 0, 0});

        // Create new declare op and copy op
        const auto alignedWeightShape =
                SmallVector<int64_t>{origShape[0] + requiredPadding, origShape[1], origShape[2], origShape[3]};

        vpux::NDTypeInterface outType = nullptr;
        if (const auto perAxisQType =
                    origOutputType.getElementType().dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
            auto newElemType = expandScalesAndZP(perAxisQType, {0, 0, 0, 0}, {requiredPadding, 0, 0, 0});
            auto outMemRef = mlir::MemRefType::get(alignedWeightShape, newElemType);
            auto type = outMemRef.cast<NDTypeInterface>();
            outType = type.changeDimsOrder(origOutputType.getDimsOrder());
        } else {
            auto outMemRef = mlir::MemRefType::get(alignedWeightShape, origOutputType.getElementType());
            outType = outMemRef.cast<NDTypeInterface>();
        }

        auto paddedDecOp = builder.create<Const::DeclareOp>(decOp.getLoc(), outType, padAttr);
        auto newAllocOp = createCMXBuffer(outType, paddedDecOp.getLoc(), builder);
        auto newCopyOp = builder.create<VPUIP::CopyOp>(paddedDecOp.getLoc(), paddedDecOp.output(), newAllocOp.memref());

        copyOp->replaceAllUsesWith(newCopyOp);
        copyOp->erase();
        if (decOp->getUses().empty()) {
            decOp->erase();
        }
        if (allocOp->getUses().empty()) {
            allocOp->erase();
        }
        allocOp = newAllocOp;
    }
}

int64_t AlignmentForSwizzling::getAlignmentForSwizzling(int64_t swizzlingKey) {
    if (swizzlingKey < 1 || swizzlingKey > 5) {
        return 0;
    }

    const EnumMap<int64_t, int64_t> swizzlingAlignment = {{1, 1024}, {2, 2048}, {3, 4096}, {4, 8192}, {5, 16384}};

    return swizzlingAlignment.at(swizzlingKey);
}

void AlignmentForSwizzling::constantBufferSwizzling(mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp nceOp,
                                                    mlir::Value cst) {
    auto isTiled = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>() != nullptr;
    auto constant = VPUIP::getTopBufferOfNCEClusterTiling(nceOp, cst);

    if (isTiled) {
        auto bufferTilingOp = constant.getDefiningOp<VPUIP::NCEClusterTilingOp>();
        auto distributedBuffer = bufferTilingOp.output_buffs()[0].getDefiningOp<VPURT::AllocDistributed>();

        _log.trace("Enable swizzling for distributed weights table buffer of NCE task - '{0}'", nceOp->getLoc());
        addSwizzlingAttributesToBuffer<VPURT::AllocDistributed, VPURT::AllocDistributed>(builder, distributedBuffer);

    } else {
        auto copyOp = constant.getDefiningOp<VPUIP::CopyOp>();
        auto allocOp = copyOp.output_buff().getDefiningOp<mlir::memref::AllocOp>();

        _log.trace("Enable swizzling for weights table buffer of NCE task - '{0}'", nceOp->getLoc());
        addPaddingAttrToBuffers(builder, copyOp, allocOp);
        addSwizzlingAttributesToBuffer<mlir::memref::AllocOp, VPURT::Alloc>(builder, allocOp);
    }
}

void AlignmentForSwizzling::activationBufferSwizzling(mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp nceOp) {
    mlir::Value nceResult = nceOp.output();
    mlir::Value outputBuff = nceOp.output_buff();
    // Check if wrapped in NCEClusterTiling
    if (auto nceClusterOp = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>()) {
        nceResult = nceClusterOp->getResult(0);
        outputBuff = nceClusterOp.output_buffs()[0];
    }

    // Find DPU->DPU buffers that can be swizzled
    for (auto* user : nceResult.getUsers()) {
        auto* userTask = user;
        if (auto nceClusterUser = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(user)) {
            userTask = nceClusterUser.getInnerTaskOp();
        }

        if (!mlir::isa<VPUIP::NCEClusterTaskOp>(userTask)) {
            return;
        }
    }

    auto swizzlingAttr = builder.getI64IntegerAttr(VPUIP::SWIZZLING_KEY_5);
    auto alignmentAttr = builder.getI64IntegerAttr(getAlignmentForSwizzling(VPUIP::SWIZZLING_KEY_5));

    auto* bufOp = outputBuff.getDefiningOp();

    if (!mlir::isa_and_nonnull<mlir::memref::AllocOp, VPURT::AllocDistributed>(bufOp)) {
        return;
    }

    auto origType = (*bufOp->getResultTypes().begin()).cast<vpux::NDTypeInterface>();
    auto origSize = origType.getTotalAllocSize().count();

    // TODO: Swizzling for buffers of size not aligned to 512 is not yet supported until
    // spilling in such scenario is complete and tested - E#48022
    if (origSize % VPUIP::SWIZZLING_SIZE_ALIGNMENT) {
        return;
    }

    _log.trace("Enable swizzling for output buffer of NCE task - '{0}'", nceOp->getLoc());

    builder.setInsertionPoint(bufOp);
    auto newLoc = appendLoc(bufOp->getLoc(), "_alloc_swizzling");

    if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(bufOp)) {
        auto newAlloc = builder.create<VPURT::Alloc>(newLoc, origType, alignmentAttr, swizzlingAttr);
        allocOp->replaceAllUsesWith(newAlloc);
        allocOp->erase();
    } else if (auto allocOp = mlir::dyn_cast<VPURT::AllocDistributed>(bufOp)) {
        auto newAlloc = builder.create<VPURT::AllocDistributed>(newLoc, origType, alignmentAttr, swizzlingAttr);
        allocOp->replaceAllUsesWith(newAlloc);
        allocOp->erase();
    }
}

//
// safeRunOnFunc
//

void AlignmentForSwizzling::safeRunOnFunc() {
    auto func = getFunction();

    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    if (arch != VPU::ArchKind::VPUX37XX && arch != VPU::ArchKind::VPUX40XX) {
        _log.trace("Swizzling is supported starting from VPUX37XX");
        return;
    }

    if (!_enableActivationSwizzling && !_enableWeightSwizzling) {
        _log.trace("Swizzling is disabled");
        return;
    }

    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&func.getBody().front().front(), &builderLog);

    func->walk([&](VPUIP::NCEClusterTaskOp nceOp) {
        if (_enableWeightSwizzling) {
            if (canSwizzleWeights(nceOp)) {
                constantBufferSwizzling(builder, nceOp, nceOp.weights());
                constantBufferSwizzling(builder, nceOp, nceOp.weight_table());
            }
        }
        if (_enableActivationSwizzling) {
            activationBufferSwizzling(builder, nceOp);
        }
    });

    auto& aliasInfo = getAnalysis<AliasesInfo>();

    auto getSwizzlingAttribute = [](mlir::Operation* op) -> int64_t {
        if (auto allocOp = mlir::dyn_cast_or_null<VPURT::Alloc>(op)) {
            return allocOp.swizzlingKey().getValueOr(0);
        } else if (auto distAllocOp = mlir::dyn_cast_or_null<VPURT::AllocDistributed>(op)) {
            return distAllocOp.swizzlingKey().getValueOr(0);
        }
        return 0;
    };

    auto removeSwizzlingAttribute = [](mlir::Operation* op) {
        if (auto allocOp = mlir::dyn_cast_or_null<VPURT::Alloc>(op)) {
            allocOp.removeSwizzlingKeyAttr();
            allocOp.removeAlignmentAttr();
        } else if (auto distAllocOp = mlir::dyn_cast_or_null<VPURT::AllocDistributed>(op)) {
            distAllocOp.removeSwizzlingKeyAttr();
            distAllocOp.removeAlignmentAttr();
        }
    };

    // Check Eltwise operations as they require same swizzling attribute on both inputs
    // There might be cases where disabling swizzling for one of eltwise inputs
    // might require doing the same for other eltwise ops which depend on the same buffer.
    // Instead of analyzing such potential chain of eltwise siblings (which should be
    // unlikely to occur) below code runs in a loop until no eltwise related
    // disabling of swizzling was performed
    bool anyEltwiseBuffersUpdated;
    do {
        anyEltwiseBuffersUpdated = false;
        func->walk([&](VPUIP::NCEClusterTaskOp nceOp) {
            if (nceOp.task_type() != VPUIP::NCETaskType::ELTWISE) {
                return;
            }

            auto input1 = nceOp.input();
            auto input2 = nceOp.weights();

            if (auto nceClusterTilingOp = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>()) {
                if (auto blockArg1 = input1.dyn_cast<mlir::BlockArgument>()) {
                    input1 = nceClusterTilingOp->getOperand(blockArg1.getArgNumber());
                }
                if (auto blockArg2 = input2.dyn_cast<mlir::BlockArgument>()) {
                    input2 = nceClusterTilingOp->getOperand(blockArg2.getArgNumber());
                }
            }

            const auto input1Buff = *aliasInfo.getRoots(input1).begin();
            const auto input2Buff = *aliasInfo.getRoots(input2).begin();

            auto* nceInput1Op = input1Buff.getDefiningOp();
            auto* nceInput2Op = input2Buff.getDefiningOp();

            auto input1Swizzling = getSwizzlingAttribute(nceInput1Op);
            auto input2Swizzling = getSwizzlingAttribute(nceInput2Op);

            if (input1Swizzling != input2Swizzling) {
                if (nceInput1Op) {
                    removeSwizzlingAttribute(nceInput1Op);
                }
                if (nceInput2Op) {
                    removeSwizzlingAttribute(nceInput2Op);
                }
                anyEltwiseBuffersUpdated = true;
                _log.trace("Disable swizzling on eltwise inputs due to mismatch of swizzling key setting, task - "
                           "'{0}'",
                           nceOp->getLoc());
            }
        });
    } while (anyEltwiseBuffersUpdated);
}

}  // namespace

//
// createAlignmentForSwizzling
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createAlignmentForSwizzling(bool enableWeightSwizzling,
                                                                     bool enableActivationSwizzling, Logger log) {
    return std::make_unique<AlignmentForSwizzling>(enableWeightSwizzling, enableActivationSwizzling, log);
}
