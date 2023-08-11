//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/constant_fusion.hpp"

#include "vpux/compiler/core/aliases_info.hpp"

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/hw_settings.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/PatternMatch.h>

#include <mlir/Pass/PassManager.h>

#include <algorithm>

using namespace vpux;

namespace {

//
// Swizzling
//

class Swizzling final : public VPUIP::SwizzlingBase<Swizzling> {
public:
    using ValuesSet = mlir::DenseSet<mlir::Value>;

public:
    explicit Swizzling(const bool enableWeightSwizzling, const bool enableActivationSwizzling, Logger log)
            : _enableWeightSwizzling(enableWeightSwizzling), _enableActivationSwizzling(enableActivationSwizzling) {
        Base::initLogger(log, Base::getArgumentName());
    }
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    bool _enableWeightSwizzling;
    bool _enableActivationSwizzling;
    // Flags used for debug purpose and performance experiments
    bool _checkConstantSizeAlignment = false;
    bool _enableSwizzlingOfFusedConsts = false;

    int64_t _cmxSize{};
    mlir::MLIRContext* _ctx{nullptr};
    VPU::ArchKind _archKind{VPU::ArchKind::UNKNOWN};

    // Store information about NCEClusterTask (or NCEClusterTiling) operands which
    // got swizzled
    ValuesSet _swizzledNceOperands;
    void safeRunOnFunc() final;
    void activationBufferSwizzling(mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp nceOp);
    void constantBufferSwizzling(mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp nceOp, mlir::Value cst);
    template <typename InAllocOp, typename OutAllocOp>
    void addSwizzlingAttributesToBuffer(mlir::OpBuilder& builder, InAllocOp inAllocOp, mlir::Type newType,
                                        VPU::ArchKind archKind);
    void attachSwizzleTransformation(Const::DeclareOp decOp, mlir::Operation* cstLoadOp, int64_t swizzlingKey);
    bool canSwizzleWeights(VPUIP::NCEClusterTaskOp nceOp);
    bool canSwizzleActivation(VPUIP::NCEClusterTaskOp nceOp);
    bool checkCMXUsage(mlir::Operation* op, ValuesSet newBufsToSwizzle) const;
    SmallVector<mlir::Operation*> _opsToRemove{};
    struct OpSwizzlingFlags {
        bool activationInput{false};
        bool activationOutput{false};
        bool weightInput{false};
    };
    DenseMap<VPUIP::NCEClusterTaskOp, OpSwizzlingFlags> _opsSwizzlingFlagsMap;
};

void adjustReturnTypesForInputChain(mlir::Value value, int64_t swizzlingKey, VPU::ArchKind archKind) {
    auto adjustReturnType = [&](mlir::Value value) {
        auto adjustedType = setSwizzlingKey(value.getType(), swizzlingKey, archKind);
        value.setType(adjustedType);
    };

    adjustReturnType(value);
    if (auto viewOp = mlir::dyn_cast_or_null<VPUIP::ViewOp>(value.getDefiningOp())) {
        // Update the source return type, subview in this case
        auto subView = viewOp.getViewSource();
        adjustReturnType(subView);
    }
}

VPUIP::NCEClusterTilingOp buildNewNceClusterTilingOp(mlir::OpBuilder& builder,
                                                     VPUIP::NCEClusterTilingOp oldNceClusterTilingOp,
                                                     mlir::Operation* innerOpToClone,
                                                     SmallVector<mlir::Type> newResultTypes) {
    SmallVector<mlir::Value> newClusterTilingWithCopyOpOperands = oldNceClusterTilingOp->getOperands();

    builder.setInsertionPointAfter(oldNceClusterTilingOp);

    // Wrap CopyOp with new NCEClusterTiling with right result type
    const auto bodyBuilder = [&](mlir::OpBuilder& opBuilder, mlir::Location loc, mlir::ValueRange newOperands) {
        std::ignore = loc;

        mlir::BlockAndValueMapping mapper;
        auto origArguments = oldNceClusterTilingOp.body().front().getArguments();
        mapper.map(origArguments, newOperands);

        opBuilder.clone(*innerOpToClone, mapper);
    };

    auto newClusterTilingWithCopyOp = builder.create<VPUIP::NCEClusterTilingOp>(
            oldNceClusterTilingOp->getLoc(), newResultTypes, newClusterTilingWithCopyOpOperands, bodyBuilder);

    oldNceClusterTilingOp->replaceAllUsesWith(newClusterTilingWithCopyOp);

    return newClusterTilingWithCopyOp;
}

// After NCEClusterTiling operation was recreated and it is producing new result type
// it might be used as an operand by another NCEClusterTiling operation which will need to have its
// argument type updated
void updateNceClusterTilingOpResultUsers(VPUIP::NCEClusterTilingOp nceClusterTilingOp, unsigned int index = 0) {
    const auto result = nceClusterTilingOp.getResult(index);
    const auto resultType = result.getType();
    for (auto& use : llvm::make_early_inc_range(result.getUses())) {
        auto operandIdx = use.getOperandNumber();
        auto userOp = use.getOwner();
        if (auto nceClusterTilingUser = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(userOp)) {
            auto argument = nceClusterTilingUser.body().getArgument(operandIdx);
            if (auto distType = resultType.dyn_cast<VPUIP::DistributedBufferType>()) {
                argument.setType(distType.getCompactType());
            } else {
                argument.setType(resultType);
            }
        }
    }
}

VPUIP::DistributedBufferType getDistributedBufferTypeWithSwizzling(VPUIP::DistributedBufferType origDistType,
                                                                   VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr) {
    const auto ctx = origDistType.getContext();
    const auto order = origDistType.getDimsOrder();
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
    const auto layoutAttr =
            VPUIP::MemRefAttr::get(orderAttr, nullptr, swizzlingSchemeAttr, nullptr, /*allocSize=*/nullptr, ctx);

    return VPUIP::DistributedBufferType::get(ctx, origDistType.getShape().raw(), origDistType.getElementType(),
                                             layoutAttr, origDistType.getMemSpace(), origDistType.getDistribution(),
                                             origDistType.getCompressionScheme());
}

bool isSizeAlignmentRequired(Const::DeclareOp decOp, VPU::ArchKind archKind,
                             VPUIP::DistributedBufferType distributedType = nullptr) {
    auto isAlignmentRequired = [&](NDTypeInterface type) {
        auto swizzlingSizeAlignment = vpux::getSizeAlignmentForSwizzling(archKind);
        auto totalSize = type.getTotalAllocSize().count();
        if (totalSize % swizzlingSizeAlignment == 0) {
            return false;
        }
        return true;
    };

    auto type = decOp.getType().cast<NDTypeInterface>();
    if (distributedType == nullptr) {
        return isAlignmentRequired(type);
    } else {
        const auto distributionAttr = distributedType.getDistribution();
        const auto numClusters = distributionAttr.num_clusters().getInt();

        const auto perClusterShapes = distributedType.getPerClusterMemoryShapes();
        VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(numClusters),
                          "Number of shapes '{0}' and clusters '{1}' are mismatch", perClusterShapes.size(),
                          numClusters);

        const auto perClusterOffsets = distributedType.getPerClusterMemoryShapeOffsets();
        VPUX_THROW_UNLESS(perClusterOffsets.size() == checked_cast<size_t>(numClusters),
                          "Number of offsets '{0}' and clusters '{1}' are mismatch", perClusterOffsets.size(),
                          numClusters);

        auto elemType = type.getElementType();

        for (auto i = 0; i < numClusters; i++) {
            mlir::Type newType = nullptr;
            if (auto qType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
                const auto newQType = tileScalesAndZP(qType, perClusterShapes[i], perClusterOffsets[i]);
                newType = type.changeShapeElemType(perClusterShapes[i], newQType);
            } else {
                newType = type.changeShape(perClusterShapes[i]);
            }

            newType = VPUIP::tileTypeCompressionScheme(newType, perClusterOffsets[i], perClusterShapes[i])
                              .cast<vpux::NDTypeInterface>();

            if (isAlignmentRequired(newType)) {
                return true;
            }
        }
    }
    return false;
}

mlir::LogicalResult Swizzling::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (enableWeightsSwizzling.hasValue()) {
        _enableWeightSwizzling = enableWeightsSwizzling.getValue();
    }
    if (enableActivationSwizzling.hasValue()) {
        _enableActivationSwizzling = enableActivationSwizzling.getValue();
    }

    return mlir::success();
}

// Check if for a given operation adding swizzling for provided buffers will not cause in increase
// of memory demand beyond CMX size
bool Swizzling::checkCMXUsage(mlir::Operation* op, ValuesSet newBufsToSwizzle) const {
    ValuesSet operands(op->getOperands().begin(), op->getOperands().end());

    SmallVector<std::pair<int64_t, int64_t>> buffSizeAndAlignment;
    for (auto operand : operands) {
        bool bufferSwizzled = false;

        if (newBufsToSwizzle.find(operand) != newBufsToSwizzle.end()) {
            bufferSwizzled = true;
        }

        if (_swizzledNceOperands.find(operand) != _swizzledNceOperands.end()) {
            bufferSwizzled = true;
        }

        auto totalSize = operand.getType().cast<vpux::NDTypeInterface>().getTotalAllocSize().count();

        if (bufferSwizzled) {
            buffSizeAndAlignment.push_back(
                    std::make_pair(alignSizeForSwizzling(totalSize, _archKind),
                                   vpux::getAddressAlignmentForSwizzling(vpux::SWIZZLING_KEY_5, _archKind)));
        } else {
            buffSizeAndAlignment.push_back(std::make_pair(totalSize, vpux::DEFAULT_CMX_ALIGNMENT));
        }
    }

    std::sort(buffSizeAndAlignment.begin(), buffSizeAndAlignment.end());

    // Because at this stage the order of allocation that will be used by FeasibleMemoryScheduler is not known,
    // perform the check on CMX usage on all permutations
    do {
        int64_t freeAddress = 0;

        for (auto& buf : buffSizeAndAlignment) {
            auto start = freeAddress;
            if (start % buf.second) {
                start += buf.second - start % buf.second;
            }
            auto end = start + buf.first;
            freeAddress = end;
        }

        if (freeAddress > _cmxSize) {
            return false;
        }

    } while (std::next_permutation(buffSizeAndAlignment.begin(), buffSizeAndAlignment.end()));

    return true;
}

// 2 Major checks are required, the rest are just null checks
// 1. Are weights constant <- These buffers are swizzled as part of activation swizzling
//                            so avoid double swizzling here
// 2. isSizeAlignmentRequired <- This case will be enabled with E#48057
bool Swizzling::canSwizzleWeights(VPUIP::NCEClusterTaskOp nceOp) {
    auto isTiled = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>() != nullptr;
    auto isFused = nceOp->hasAttr(vpux::ConstantFusing::constantsFused);
    auto weights = VPUIP::getTopBufferOfNCEClusterTiling(nceOp, nceOp.weights());
    auto weightsSM = VPUIP::getTopBufferOfNCEClusterTiling(nceOp, nceOp.weights_sparsity_map());
    auto weightTable = VPUIP::getTopBufferOfNCEClusterTiling(nceOp, nceOp.weight_table());

    _log.trace("Check if weights swizzling can be enabled for NCEOp '{0}'", nceOp->getLoc());

    // Swizzling for ELTWISE is handled with activation swizzling
    if (weights == nullptr || weightTable == nullptr || nceOp.task_type() == VPUIP::NCETaskType::ELTWISE) {
        _log.nest().trace("Cannot swizzle weights because of missed weights", nceOp->getLoc());
        return false;
    }

    // WA for disabling swizzling for compressed conv layer
    // E#56431 will remove workaround for a long term fix
    if (auto shapeCastOp = weights.getDefiningOp<VPUIP::ShapeCastOp>()) {
        _log.nest().trace("Cannot swizzle weights because this is compressed conv");
        return false;
    }

    if (isFused) {
        if (!_enableSwizzlingOfFusedConsts) {
            // Even though support is in place enabling this caused schedule change
            // that had negative impact on performance in few cases
            // E#73720 will try to remove this check
            _log.nest().trace("Do not swizzle weights in case of fused consts");
            return false;
        }
        VPUIP::CopyOp copyOp = nullptr;
        auto decOp = vpux::ConstantFusing::getConstAndCopyOp(nceOp, weights, copyOp);
        if (decOp == nullptr) {
            _log.nest().trace("Cannot swizzle weights because of missed declare op");
            return false;
        }
        if (_checkConstantSizeAlignment && isSizeAlignmentRequired(decOp, _archKind)) {
            _log.nest().trace("Cannot swizzle weights. Size alignment required");
            return false;
        }
    } else if (isTiled) {
        auto checkDistributedContent = [&](VPUIP::NCEClusterTilingOp clusterTilingOp) {
            if (clusterTilingOp == nullptr) {
                return false;
            }

            auto copyOp = clusterTilingOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
            if (copyOp == nullptr) {
                return false;
            }

            auto decOp = clusterTilingOp.inputs()[0].getDefiningOp<Const::DeclareOp>();
            if (decOp == nullptr) {
                return false;
            }

            auto distributedBufferOp = clusterTilingOp.output_buffs()[0].getDefiningOp<VPURT::AllocDistributed>();
            if (distributedBufferOp == nullptr) {
                return false;
            }

            auto distrType = clusterTilingOp.output_buffs()[0].getType().cast<VPUIP::DistributedBufferType>();
            if (_checkConstantSizeAlignment && isSizeAlignmentRequired(decOp, _archKind, distrType)) {
                return false;
            }
            return true;
        };

        auto shapeCastOp = weights.getDefiningOp<VPUIP::ShapeCastOp>();

        auto bufferTilingOp = shapeCastOp ? shapeCastOp.source().getDefiningOp<VPUIP::NCEClusterTilingOp>()
                                          : weights.getDefiningOp<VPUIP::NCEClusterTilingOp>();
        if (!checkDistributedContent(bufferTilingOp)) {
            _log.nest().trace("Cannot swizzle weights, because weights do not satisfy distributed type requirements");
            return false;
        }

        // Temporarily disable swizzling for WT buffer size not meeting size alignment requirement
        // TODO E#48057 Resize these buffers
        auto wtBufferTilingOp = weightTable.getDefiningOp<VPUIP::NCEClusterTilingOp>();
        if (!checkDistributedContent(wtBufferTilingOp)) {
            _log.nest().trace(
                    "Cannot swizzle weights, because weightTable do not satisfy distributed type requirements");
            return false;
        }

        if (weightsSM != nullptr) {
            auto weightsSMClusterTilingCopyOp = weightsSM.getDefiningOp<VPUIP::NCEClusterTilingOp>();
            if (!checkDistributedContent(weightsSMClusterTilingCopyOp)) {
                _log.nest().trace(
                        "Cannot swizzle weights, because weightsSM do not satisfy distributed type requirements");
                return false;
            }
        }
    } else {
        auto checkCopyOfContentForSwizzling = [&](VPUIP::CopyOp copyOp) {
            if (copyOp == nullptr) {
                return false;
            }

            auto decOp = copyOp.input().getDefiningOp<Const::DeclareOp>();
            if (decOp == nullptr) {
                return false;
            }

            if (_checkConstantSizeAlignment && isSizeAlignmentRequired(decOp, _archKind)) {
                return false;
            }
            return true;
        };

        auto copyOp = weights.getDefiningOp<VPUIP::CopyOp>();
        if (!checkCopyOfContentForSwizzling(copyOp)) {
            _log.nest().trace("Cannot swizzle weights, because weights do not satisfy distributed type requirements");
            return false;
        }

        auto wtCopyOp = weightTable.getDefiningOp<VPUIP::CopyOp>();
        if (!checkCopyOfContentForSwizzling(wtCopyOp)) {
            _log.nest().trace(
                    "Cannot swizzle weights, because weightTable do not satisfy distributed type requirements");
            return false;
        }

        if (weightsSM != nullptr) {
            auto weightsSMCopyOp = weightsSM.getDefiningOp<VPUIP::CopyOp>();
            if (!checkCopyOfContentForSwizzling(weightsSMCopyOp)) {
                _log.nest().trace(
                        "Cannot swizzle weights, because weightSM do not satisfy distributed type requirements");
                return false;
            }
        }
    }

    ValuesSet operands = {weights, weightTable};
    if (weightsSM != nullptr) {
        operands.insert(weightsSM);
    }

    // Check if adding constants swizzling will not increase operation memory demand beyond CMX size
    if (!checkCMXUsage(isTiled ? nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>() : nceOp, operands)) {
        _log.nest().trace("Do not enable weights swizzling because of increase in memory demand beyond CMX size");
        return false;
    }

    _log.nest().trace("NCEOp weights are eligible for swizzling");
    return true;
}

template <typename InAllocOp, typename OutAllocOp>
void Swizzling::addSwizzlingAttributesToBuffer(mlir::OpBuilder& builder, InAllocOp inAllocOp, mlir::Type newType,
                                               VPU::ArchKind archKind) {
    auto swizzlingSchemeAttr = getSwizzlingSchemeAttr(newType);
    auto addressAlignment = vpux::getAddressAlignmentForSwizzling(swizzlingSchemeAttr.getKey().getInt(), archKind);
    auto addressAlignmentAttr = getIntAttr(_ctx, addressAlignment);

    builder.setInsertionPoint(inAllocOp);

    auto origLoc = inAllocOp->getLoc();
    auto newLoc = appendLoc(origLoc, "_alloc_swizzling");

    auto outAllocOp = builder.create<OutAllocOp>(newLoc, newType, addressAlignmentAttr, swizzlingSchemeAttr.getKey());

    inAllocOp->replaceAllUsesWith(outAllocOp);
    inAllocOp->erase();
    _swizzledNceOperands.insert(outAllocOp->getResult(0));
}

bool isTypeSwizzled(mlir::Type type) {
    VPUIP::MemRefAttr memRefAttr;
    type = VPUIP::extractDataType(type);
    if (auto bufferMemRefType = type.cast<mlir::MemRefType>()) {
        memRefAttr = bufferMemRefType.getLayout().dyn_cast<VPUIP::MemRefAttr>();
    } else if (auto distBufferType = type.cast<VPUIP::DistributedBufferType>()) {
        memRefAttr = distBufferType.getLayout().dyn_cast<VPUIP::MemRefAttr>();
    }

    if (memRefAttr && memRefAttr.swizzlingScheme()) {
        return true;
    }

    return false;
}

void Swizzling::attachSwizzleTransformation(Const::DeclareOp cstOp, mlir::Operation* cstLoadOp, int64_t swizzlingKey) {
    VPUX_THROW_WHEN(cstOp == nullptr, "DeclareOp was not found");
    // On top of existing transformation a new transformation is added to the content attribute
    // of weight table const. The new transformation will swizzle the constant with swizzle key parameter
    _log.nest().trace("Constant for swizzling transformation'{0}'", cstOp->getLoc());

    // Extract content attrib with existing transformations
    auto constAttr = cstOp.contentAttr();

    for (auto transAttr : constAttr.getTransformations()) {
        // Check if swizzling transformation is already attached, this can happen when constant is shared
        // between 2 or more NCEOps or when constants are fused
        auto swizzleConstAttr = transAttr.dyn_cast_or_null<vpux::Const::SwizzleConstantAttr>();
        if (swizzleConstAttr != nullptr) {
            return;
        }
    }

    // Create new attribute based on existing one by adding new swizzleConstant transformation
    auto newConstAttr = constAttr.swizzleConstant(swizzlingKey, static_cast<uint64_t>(_archKind));
    mlir::OpBuilder builder(cstOp);

    auto outputType = cstOp.output().getType();
    outputType = vpux::setSwizzlingKey(outputType, swizzlingKey, _archKind);

    auto newCstOp = builder.create<Const::DeclareOp>(cstOp.getLoc(), outputType, newConstAttr);
    cstLoadOp->setOperand(0, newCstOp.output());

    if (cstOp->getUses().empty()) {
        cstOp.erase();
    }
}

void Swizzling::constantBufferSwizzling(mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp nceOp, mlir::Value cst) {
    if (cst == nullptr) {
        return;
    }

    auto isTiled = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>() != nullptr;
    auto isFused = nceOp->hasAttr(vpux::ConstantFusing::constantsFused);
    auto constant = VPUIP::getTopBufferOfNCEClusterTiling(nceOp, cst);
    auto swizzlingSchemeAttr = createSwizzlingSchemeAttr(_ctx, _archKind, SWIZZLING_KEY_5);

    auto shapeCastOp = constant.getDefiningOp<VPUIP::ShapeCastOp>();
    if (isFused || shapeCastOp) {
        adjustReturnTypesForInputChain(constant, SWIZZLING_KEY_5, _archKind);
    }

    if (isTiled) {
        VPUIP::NCEClusterTilingOp clusterTilingWithCopyOp = nullptr;
        VPUIP::CopyOp copyOp = nullptr;
        if (isFused) {
            std::ignore = vpux::ConstantFusing::getConstAndCopyOp(nceOp, constant, copyOp);
            clusterTilingWithCopyOp = copyOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
        } else {
            clusterTilingWithCopyOp = shapeCastOp != nullptr
                                              ? shapeCastOp.source().getDefiningOp<VPUIP::NCEClusterTilingOp>()
                                              : constant.getDefiningOp<VPUIP::NCEClusterTilingOp>();
            copyOp = mlir::dyn_cast<VPUIP::CopyOp>(clusterTilingWithCopyOp.getInnerTaskOp());
        }

        if (isTypeSwizzled(copyOp.output_buff().getType())) {
            // In case of constant folding buffer might have already been swizzled
            return;
        }

        auto decOp = clusterTilingWithCopyOp.inputs()[0].getDefiningOp<Const::DeclareOp>();
        auto distributedBuffer = clusterTilingWithCopyOp.output_buffs()[0].getDefiningOp<VPURT::AllocDistributed>();

        _log.trace("Enable swizzling for distributed constant buffer of NCE task - '{0}'", nceOp->getLoc());

        attachSwizzleTransformation(decOp, clusterTilingWithCopyOp.getOperation(), SWIZZLING_KEY_5);

        auto origType = distributedBuffer.getType().cast<VPUIP::DistributedBufferType>();

        // Create new DistributedBufferType which as part of layout will have swizzling set
        auto newType = getDistributedBufferTypeWithSwizzling(origType, swizzlingSchemeAttr);

        // Create new allocation with swizzling enabled and required alignment setting
        addSwizzlingAttributesToBuffer<VPURT::AllocDistributed, VPURT::AllocDistributed>(builder, distributedBuffer,
                                                                                         newType, _archKind);

        // Create new CopyOp with correct result type
        builder.setInsertionPointAfter(copyOp);
        auto tempCopyOp = builder.create<VPUIP::CopyOp>(copyOp.getLoc(), newType.getCompactType(), copyOp.input(),
                                                        copyOp.output_buff());
        copyOp->replaceAllUsesWith(tempCopyOp);
        copyOp->erase();

        auto newNceClusterTilingOp =
                buildNewNceClusterTilingOp(builder, clusterTilingWithCopyOp, tempCopyOp.getOperation(), {newType});

        updateNceClusterTilingOpResultUsers(newNceClusterTilingOp);

        tempCopyOp->erase();
        clusterTilingWithCopyOp->erase();
    } else {
        auto copyOp = constant.getDefiningOp<VPUIP::CopyOp>();
        if (isFused) {
            std::ignore = vpux::ConstantFusing::getConstAndCopyOp(nceOp, constant, copyOp);
        }

        if (isTypeSwizzled(copyOp.output_buff().getType())) {
            // In case of constant folding buffer might have already been swizzled
            return;
        }

        auto decOp = copyOp.input().getDefiningOp<Const::DeclareOp>();
        auto allocOp = copyOp.output_buff().getDefiningOp<mlir::memref::AllocOp>();
        VPUX_THROW_WHEN(allocOp == nullptr, "Allocation operation was not identified");

        _log.trace("Enable swizzling for constant buffer of NCE task - '{0}'", nceOp->getLoc());

        attachSwizzleTransformation(decOp, copyOp.getOperation(), SWIZZLING_KEY_5);

        auto origType = allocOp.getType().cast<vpux::NDTypeInterface>();
        auto newType = getMemRefType(origType.getShape(), origType.getElementType(), origType.getDimsOrder(),
                                     origType.getMemSpace(), StridesRef(), swizzlingSchemeAttr,
                                     VPUIP::getCompressionSchemeAttr(origType));

        addSwizzlingAttributesToBuffer<mlir::memref::AllocOp, VPURT::Alloc>(builder, allocOp, newType, _archKind);

        // Create new CopyOp with updated type
        builder.setInsertionPoint(copyOp);
        auto newCopyOp = builder.create<VPUIP::CopyOp>(copyOp.getLoc(), copyOp.input(), copyOp.output_buff());
        copyOp->replaceAllUsesWith(newCopyOp);
        copyOp->erase();
    }
}

bool Swizzling::canSwizzleActivation(VPUIP::NCEClusterTaskOp nceOp) {
    mlir::Value nceDataResult = nceOp.output();
    mlir::Value maybeNceSMResult = nullptr;

    _log.trace("Check if output swizzling can be enabled for NCEOp '{0}'", nceOp->getLoc());

    ValuesSet nceResults = {nceOp.output()};
    ValuesSet outputBuffs = {nceOp.output_buff()};
    if (auto outSM = nceOp.output_sparsity_map()) {
        maybeNceSMResult = outSM;
        nceResults.insert(outSM);
        outputBuffs.insert(nceOp.output_sparsity_map_buff());
    }

    auto* op = nceOp.getOperation();
    // Check if wrapped in NCEClusterTiling
    if (auto nceClusterOp = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>()) {
        nceDataResult = nceClusterOp->getResult(0);
        if (maybeNceSMResult != nullptr) {
            maybeNceSMResult = nceClusterOp->getResult(1);
        }
        nceResults = to_container<ValuesSet>(nceOp->getResults());
        outputBuffs = to_container<ValuesSet>(nceClusterOp.output_buffs());
        op = nceClusterOp.getOperation();
    }

    VPUX_THROW_UNLESS(nceResults.size() == 1 || nceResults.size() == 2,
                      "NCEClusterTaskOp should have exact 1 or 2 output buffers, but got {0}", nceResults.size());

    mlir::DenseSet<mlir::Operation*> userTasks;

    // Find DPU->DPU buffers that can be swizzled
    for (auto* user : nceDataResult.getUsers()) {
        auto* userTask = user;
        userTasks.insert(userTask);
        if (auto nceClusterUser = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(user)) {
            userTask = nceClusterUser.getInnerTaskOp();
        }

        auto userNCETaskOp = mlir::dyn_cast_or_null<VPUIP::NCEClusterTaskOp>(userTask);
        if (userNCETaskOp == nullptr) {
            _log.nest().trace("Cannot swizzle activation buffers because one of consumers is not "
                              "VPUIP.NCEClusterTaskOp, user NCEOp - '{0}'",
                              userTask->getLoc());
            return false;
        }

        // In theory this is impossible by design, but better to double check
        if (maybeNceSMResult != nullptr &&
            VPUIP::getTopBufferOfNCEClusterTiling(userNCETaskOp, userNCETaskOp.input_sparsity_map()) !=
                    maybeNceSMResult) {
            _log.nest().trace("Cannot swizzle activation buffers because sparsity map is not directly consumed by "
                              "VPUIP.NCEClusterTaskOp");
            return false;
        }
    }

    for (auto outBuff : outputBuffs) {
        auto* bufOp = outBuff.getDefiningOp();

        if (!mlir::isa_and_nonnull<mlir::memref::AllocOp, VPURT::AllocDistributed>(bufOp)) {
            _log.nest().trace("Cannot swizzle activation buffers because buffer '{0}' is not defined by allocation op",
                              outBuff);
            return false;
        }
    }

    // Check if adding activation swizzling will not increase operation memory demand beyond CMX size
    if (!checkCMXUsage(op, outputBuffs)) {
        _log.nest().trace("Do not enable activation swizzling because of increase in memory demand beyond CMX size");
        return false;
    }

    // Check if adding activation swizzling on output of this op will not increase memory demand
    // beyond CMX size of user operations
    for (auto* userTask : userTasks) {
        if (!checkCMXUsage(userTask, nceResults)) {
            _log.nest().trace("Do not enable activation swizzling because of increase in memory demand beyond CMX size "
                              "of user task - '{0}'",
                              userTask->getLoc());
            return false;
        }
    }

    // Set information about input swizzling for user tasks
    for (auto* user : userTasks) {
        auto* userTask = user;
        if (auto nceClusterUser = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(user)) {
            userTask = nceClusterUser.getInnerTaskOp();
        }

        auto userNceOp = mlir::dyn_cast_or_null<VPUIP::NCEClusterTaskOp>(userTask);
        auto userSwizzSettingsIt = _opsSwizzlingFlagsMap.find(userNceOp);
        VPUX_THROW_WHEN(userSwizzSettingsIt == _opsSwizzlingFlagsMap.end(),
                        "Not found swizzling settings for given NCEOp - {0}", userNceOp->getLoc());

        // Before making final decision, check if user NCEOps which have fused constants
        // and have constant that is related to NCEOp input (e.g. activation_window)
        // has it also swizzled
        if (userNceOp.activation_window() != nullptr && userNceOp->hasAttr(vpux::ConstantFusing::constantsFused)) {
            // If fused constant of user task is not swizzled then output cannot be swizzled
            if (!userSwizzSettingsIt->second.weightInput) {
                _log.nest().trace(
                        "Do not enable activation swizzling because user task which has fused constant and has "
                        "activation window does not have constants swizzled, user task - '{0}'",
                        userNceOp->getLoc());
                return false;
            }
        }

        userSwizzSettingsIt->second.activationInput = true;
    }

    _log.nest().trace("NCEOp is eligible for swizzling");
    return true;
}

void Swizzling::activationBufferSwizzling(mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp nceOp) {
    SmallVector<mlir::Type> newClusterResultTypes;
    VPUIP::NCEClusterTilingOp nceClusterTilingOp = nullptr;

    auto getResultIndex = [&](mlir::Value outputBuffer) {
        // data always has index 0. Profiling isn't enabled yet, so sparsityMap's index 1
        auto type = outputBuffer.getType().cast<vpux::NDTypeInterface>();
        return type.getElemTypeSize() == Bit(1) ? 1 : 0;
    };

    for (auto bufVal : {nceOp.output_buff(), nceOp.output_sparsity_map_buff()}) {
        if (bufVal == nullptr) {
            continue;
        }

        mlir::Operation* sourceAllocOp = VPUIP::getTopBufferOfNCEClusterTiling(nceOp, bufVal).getDefiningOp();

        if (!mlir::isa_and_nonnull<mlir::memref::AllocOp, VPURT::AllocDistributed>(sourceAllocOp)) {
            _log.trace("Cannot swizzle output buffer of '{0}', since it's not memref.Alloc or VPURT.AllocDistrubted",
                       nceOp->getLoc());
            return;
        }

        auto swizzlingSchemeAttr = createSwizzlingSchemeAttr(_ctx, _archKind, SWIZZLING_KEY_5);
        auto addressAlignment = vpux::getAddressAlignmentForSwizzling(swizzlingSchemeAttr.getKey().getInt(), _archKind);
        auto addressAlignmentAttr = getIntAttr(_ctx, addressAlignment);

        auto origType = (*sourceAllocOp->getResultTypes().begin()).cast<vpux::NDTypeInterface>();

        const auto outputIndex = getResultIndex(bufVal);

        _log.trace("Enable swizzling for {0}th output buffer of NCE task - '{1}'", outputIndex, nceOp->getLoc());

        builder.setInsertionPoint(sourceAllocOp);
        auto newLoc = appendLoc(sourceAllocOp->getLoc(), "_alloc_swizzling");

        if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(sourceAllocOp)) {
            // Create new MemRefType which as part of layout will have swizzling set
            auto newType = getMemRefType(origType.getShape(), origType.getElementType(), origType.getDimsOrder(),
                                         origType.getMemSpace(), StridesRef(), swizzlingSchemeAttr,
                                         VPUIP::getCompressionSchemeAttr(origType));

            // Create new allocation with swizzling enabled and required alignment setting
            auto newAlloc =
                    builder.create<VPURT::Alloc>(newLoc, newType, addressAlignmentAttr, swizzlingSchemeAttr.getKey());
            allocOp->replaceAllUsesWith(newAlloc);

            _swizzledNceOperands.insert(newAlloc.buffer());

            _opsToRemove.push_back(allocOp.getOperation());

            nceOp.getResult(outputIndex).setType(newType);
        } else if (auto allocOp = mlir::dyn_cast<VPURT::AllocDistributed>(sourceAllocOp)) {
            auto distributedType = origType.dyn_cast<VPUIP::DistributedBufferType>();

            // Create new DistributedBufferType which as part of layout will have swizzling set
            auto newType = getDistributedBufferTypeWithSwizzling(distributedType, swizzlingSchemeAttr);

            // Create new allocation with swizzling enabled and required alignment setting
            auto newAlloc = builder.create<VPURT::AllocDistributed>(newLoc, newType, addressAlignmentAttr,
                                                                    swizzlingSchemeAttr.getKey());
            allocOp->replaceAllUsesWith(newAlloc);

            _swizzledNceOperands.insert(newAlloc.buffer());

            _opsToRemove.push_back(allocOp.getOperation());

            nceOp.getResult(outputIndex).setType(newType.getCompactType());

            nceClusterTilingOp = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
            VPUX_THROW_WHEN(nceClusterTilingOp == nullptr, "No NCEClusterTilingOp found");
            newClusterResultTypes.push_back(newType);
        }
    }

    _opsToRemove.push_back(nceOp.getOperation());
    if (nceClusterTilingOp != nullptr) {
        // Create new NCEClusterTask wrapped in NCEClusterTiling with updated result type
        auto newNceClusterTilingOp =
                buildNewNceClusterTilingOp(builder, nceClusterTilingOp, nceOp.getOperation(), newClusterResultTypes);

        for (auto bufVal : {nceOp.output_buff(), nceOp.output_sparsity_map_buff()}) {
            if (bufVal == nullptr) {
                continue;
            }
            const auto outputIndex = getResultIndex(bufVal);
            updateNceClusterTilingOpResultUsers(newNceClusterTilingOp, outputIndex);
        }

        _opsToRemove.push_back(nceClusterTilingOp.getOperation());
    }
}

//
// safeRunOnFunc
//

// TODO: #71565
void Swizzling::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    _archKind = VPU::getArch(module);
    _cmxSize = VPU::getTotalCMXSize(module).count();
    _ctx = &getContext();

    if (!_enableActivationSwizzling && !_enableWeightSwizzling) {
        _log.trace("Swizzling is disabled");
        return;
    }

    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&func.getBody().front().front(), &builderLog);

    // Iterate IR twice. First to check what NCEOps can have constats swizzled,
    // the second time check which NCEOps can have their output swizzled.
    // Second iteration is separated because when determining if NCEOp output can be swizzled
    // information about const swizzling is used. Some constants like activation_window impact
    // NCEOps activation input and both need to have matching swizzling setting.
    // In the end HW requires to have matching swizzling setting on NCE operands which
    // are consumed/produced by same reader/writer:
    // activation reader:
    // - input
    // - input_sparisty_map
    // - activation_window
    // weights reader
    // - weights
    // - weights_table
    // - weights_sparisty_map
    // output writer:
    // - output
    // - output_sparisty_map
    func->walk([&](VPUIP::NCEClusterTaskOp nceOp) {
        _opsSwizzlingFlagsMap[nceOp] = OpSwizzlingFlags();
        if (_enableWeightSwizzling && canSwizzleWeights(nceOp)) {
            _opsSwizzlingFlagsMap[nceOp].weightInput = true;
        }
    });

    if (_enableActivationSwizzling) {
        func->walk([&](VPUIP::NCEClusterTaskOp nceOp) {
            if (canSwizzleActivation(nceOp)) {
                _opsSwizzlingFlagsMap[nceOp].activationOutput = true;
            }
        });
    }

    // Check if in cases where fused constant is to be swizzled together with activation_window
    // and operation input is not swizzled then constant swizzling needs to be disabled
    for (auto& opsSwizzlingFlags : _opsSwizzlingFlagsMap) {
        auto nceOp = opsSwizzlingFlags.first;
        auto swizzFlags = opsSwizzlingFlags.second;
        if (!swizzFlags.activationInput && swizzFlags.weightInput &&
            nceOp->hasAttr(vpux::ConstantFusing::constantsFused) && nceOp.activation_window() != nullptr) {
            opsSwizzlingFlags.second.weightInput = false;
            _log.trace("Cannot swizzle weights of '{0}', since it is fused constant with activation_windows and there "
                       "is no swizzling on input",
                       nceOp->getLoc());
        }
    }

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

            auto input1NceTask = input1.getDefiningOp<VPUIP::NCEClusterTaskOp>();
            if (auto input1NceClusterTiling = input1.getDefiningOp<VPUIP::NCEClusterTilingOp>()) {
                input1NceTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(input1NceClusterTiling.getInnerTaskOp());
            }

            auto input2NceTask = input2.getDefiningOp<VPUIP::NCEClusterTaskOp>();
            if (auto input2NceClusterTiling = input2.getDefiningOp<VPUIP::NCEClusterTilingOp>()) {
                input2NceTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(input2NceClusterTiling.getInnerTaskOp());
            }

            bool input1SwizzlingFlag = false;
            auto input1SwizzlingFlagItr = _opsSwizzlingFlagsMap.end();
            if (input1NceTask) {
                input1SwizzlingFlagItr = _opsSwizzlingFlagsMap.find(input1NceTask);
                input1SwizzlingFlag = ((input1SwizzlingFlagItr != _opsSwizzlingFlagsMap.end()) &&
                                       input1SwizzlingFlagItr->second.activationOutput);
            }

            bool input2SwizzlingFlag = false;
            auto input2SwizzlingFlagItr = _opsSwizzlingFlagsMap.end();
            if (input2NceTask) {
                input2SwizzlingFlagItr = _opsSwizzlingFlagsMap.find(input2NceTask);
                input2SwizzlingFlag = ((input2SwizzlingFlagItr != _opsSwizzlingFlagsMap.end()) &&
                                       input2SwizzlingFlagItr->second.activationOutput);
            }

            bool outputSwizzlingFlag = false;
            auto outputSwizzlingFlagItr = _opsSwizzlingFlagsMap.end();
            bool inplaceEltwise = false;
            if (nceOp.is_inplace().value_or(false)) {
                inplaceEltwise = true;
                outputSwizzlingFlagItr = _opsSwizzlingFlagsMap.find(nceOp);
                outputSwizzlingFlag = ((outputSwizzlingFlagItr != _opsSwizzlingFlagsMap.end()) &&
                                       outputSwizzlingFlagItr->second.activationOutput);
            }

            if (input1SwizzlingFlag != input2SwizzlingFlag ||
                (inplaceEltwise &&
                 (input1SwizzlingFlag != outputSwizzlingFlag || input2SwizzlingFlag != outputSwizzlingFlag))) {
                _log.trace("Mismatch of swizzling setting of eltwise inputs, eltwise op - '{0}'", nceOp->getLoc());
                if (input1SwizzlingFlag) {
                    input1SwizzlingFlagItr->second.activationOutput = false;
                    _log.nest().trace("Swizzling cannot be enabled on op - '{0}'", input1NceTask->getLoc());
                    anyEltwiseBuffersUpdated = true;
                } else if (input2SwizzlingFlag) {
                    input2SwizzlingFlagItr->second.activationOutput = false;
                    _log.nest().trace("Swizzling cannot be enabled on op - '{0}'", input2NceTask->getLoc());
                    anyEltwiseBuffersUpdated = true;
                } else if (outputSwizzlingFlag) {
                    outputSwizzlingFlagItr->second.activationOutput = false;
                    _log.nest().trace("Swizzling cannot be enabled on eltwise - '{0}'", nceOp->getLoc());
                    anyEltwiseBuffersUpdated = true;
                }
            }
        });
    } while (anyEltwiseBuffersUpdated);

    // After identifying operations which can have swizzling applied
    // perform actual enabling. Code before only made necessary checks
    // where swizzling can be applied and in some cases due to limitations
    // (e.g. eltwise input swizzling mismatch) swizzling enabling was reverted.
    for (auto opSwizzlingFlag : _opsSwizzlingFlagsMap) {
        auto nceOp = opSwizzlingFlag.first;
        auto flags = opSwizzlingFlag.second;
        if (flags.weightInput) {
            constantBufferSwizzling(builder, nceOp, nceOp.weights());
            constantBufferSwizzling(builder, nceOp, nceOp.weights_sparsity_map());
            constantBufferSwizzling(builder, nceOp, nceOp.weight_table());
        }
        if (flags.activationInput) {
            // Special action need to be taken in case operation has activation_window, which
            // is a constant but related to activation input. If nceOp has fused constants
            // it was already swizzled as part of rest of constants
            if (nceOp.activation_window() != nullptr) {
                constantBufferSwizzling(builder, nceOp, nceOp.activation_window());
            }
        }

        if (flags.activationOutput) {
            activationBufferSwizzling(builder, nceOp);
        }
    }

    for (auto opToRemove : llvm::make_early_inc_range(_opsToRemove)) {
        if (opToRemove->use_empty()) {
            opToRemove->erase();
        }
    }
}

}  // namespace

//
// createSwizzlingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createSwizzlingPass(const bool enableWeightSwizzling,
                                                             const bool enableActivationSwizzling, Logger log) {
    return std::make_unique<Swizzling>(enableWeightSwizzling, enableActivationSwizzling, log);
}
