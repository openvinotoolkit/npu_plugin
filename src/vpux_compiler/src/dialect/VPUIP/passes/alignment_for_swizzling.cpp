//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/constant_fusion.hpp"

#include "vpux/compiler/core/aliases_info.hpp"

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/hw_settings.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/swizzling.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/PatternMatch.h>

#include <mlir/Pass/PassManager.h>

#include <algorithm>

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
    int64_t _cmxSize;
    // Store information about NCEClusterTask (or NCEClusterTiling) operands which
    // got swizzled
    mlir::DenseSet<mlir::Value> _swizzledNceOperands;
    void safeRunOnFunc() final;
    void activationBufferSwizzling(mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp nceOp);
    void constantBufferSwizzling(mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp nceOp, mlir::Value cst);
    void addPaddingAttrToBuffers(mlir::OpBuilder& builder, VPUIP::CopyOp& copyOp, mlir::memref::AllocOp& allocOp);
    template <typename InAllocOp, typename OutAllocOp>
    void addSwizzlingAttributesToBuffer(mlir::OpBuilder& builder, InAllocOp inAllocOp, mlir::Type newType);
    bool canSwizzleWeights(VPUIP::NCEClusterTaskOp nceOp);
    bool canSwizzleActivation(VPUIP::NCEClusterTaskOp nceOp);
    bool checkCMXUsage(mlir::Operation* op, mlir::DenseSet<mlir::Value> newBufsToSwizzle);
    SmallVector<mlir::Operation*> _opsToRemove;
    SmallVector<VPUIP::NCEClusterTaskOp> _opsForActivationSwizzling;
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

void adjustReturnTypesForInputChain(mlir::Value value, int64_t swizzlingKey) {
    auto adjustedType = setSwizzlingKey(value.getType(), swizzlingKey);
    value.setType(adjustedType);

    auto sourceOp = value.getDefiningOp();
    if (VPUIP::isPureViewOp(sourceOp)) {
        auto viewOp = mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(sourceOp);
        if (viewOp == nullptr) {
            return;
        }
        auto viewSource = viewOp.getViewSource();
        adjustReturnTypesForInputChain(viewSource, swizzlingKey);
    }
}

int64_t getRequiredPadding(NDTypeInterface type) {
    auto totalSize = type.getTotalAllocSize().count();
    auto elementSize = type.getElemTypeSize().count();
    ShapeRef shapeRef = type.getShape();
    if (totalSize % SWIZZLING_SIZE_ALIGNMENT == 0) {
        return 0;
    }

    auto newSize = alignSizeForSwizzling(totalSize);

    auto totalElements = (newSize * CHAR_BIT) / elementSize;
    auto paddingValue = shapeRef.front();
    auto padding = abs(((totalElements * paddingValue) / type.getNumElements()) - paddingValue);
    // At this point we know some padding will be required
    // Hence a cautionary check to see if the padding is non-zero
    return padding ? padding : ++padding;
}

VPUIP::NCEClusterTilingOp buildNewNceClusterTilingOp(mlir::OpBuilder& builder,
                                                     VPUIP::NCEClusterTilingOp oldNceClusterTilingOp,
                                                     mlir::Operation* innerOpToClone, mlir::Type resultType) {
    SmallVector<mlir::Value> newClusterTilingWithCopyOpOperands = oldNceClusterTilingOp->getOperands();
    SmallVector<mlir::Type> newClusterTilingWithCopyOpResultTypes = {resultType};

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
            oldNceClusterTilingOp->getLoc(), newClusterTilingWithCopyOpResultTypes, newClusterTilingWithCopyOpOperands,
            bodyBuilder);

    oldNceClusterTilingOp->replaceAllUsesWith(newClusterTilingWithCopyOp);

    return newClusterTilingWithCopyOp;
}

// After NCEClusterTiling operation was recreated and it is producing new result type
// it might be used as an operand by another NCEClusterTiling operation which will need to have its
// argument type updated
void updateNceClusterTilingOpResultUsers(VPUIP::NCEClusterTilingOp nceClusterTilingOp) {
    const auto result = nceClusterTilingOp.getResult(0);
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
                                                                   mlir::IntegerAttr swizzlingAttr) {
    const auto ctx = origDistType.getContext();
    const auto order = origDistType.getDimsOrder();
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));

    const auto layoutAttr = VPUIP::MemRefAttr::get(orderAttr, nullptr, swizzlingAttr, ctx);

    return VPUIP::DistributedBufferType::get(ctx, origDistType.getShape().raw(), origDistType.getElementType(),
                                             layoutAttr, origDistType.getMemSpace(), origDistType.getDistribution());
}

bool isPaddingRequired(Const::DeclareOp decOp, VPUIP::DistributedBufferType distributedType) {
    const auto distributionAttr = distributedType.getDistribution();
    const auto numClusters = distributionAttr.num_clusters().getInt();

    const auto perClusterShapes = distributedType.getPerClusterComputeShapes();
    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(numClusters),
                      "Number of shapes '{0}' and clusters '{1}' are mismatch", perClusterShapes.size(), numClusters);

    const auto perClusterOffsets = distributedType.getPerClusterComputeShapeOffsets();
    VPUX_THROW_UNLESS(perClusterOffsets.size() == checked_cast<size_t>(numClusters),
                      "Number of offsets '{0}' and clusters '{1}' are mismatch", perClusterOffsets.size(), numClusters);

    auto type = decOp.getType().cast<NDTypeInterface>();
    auto elemType = type.getElementType();

    for (auto i = 0; i < numClusters; i++) {
        mlir::Type newType = nullptr;
        if (auto qType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
            const auto newQType = tileScalesAndZP(qType, perClusterShapes[i], perClusterOffsets[i]);
            newType = type.changeShapeElemType(perClusterShapes[i], newQType);
        } else {
            newType = type.changeShape(perClusterShapes[i]);
        }

        if (getRequiredPadding(newType) != 0) {
            return true;
        }
    }

    return false;
}

int64_t getRequiredPadding(Const::DeclareOp decOp) {
    auto type = decOp.getType().cast<NDTypeInterface>();
    return getRequiredPadding(type);
}

// Check if for a given operation adding swizzling for provided buffers will not cause in increase
// of memory demand beyond CMX size
bool AlignmentForSwizzling::checkCMXUsage(mlir::Operation* op, mlir::DenseSet<mlir::Value> newBufsToSwizzle) {
    mlir::DenseSet<mlir::Value> operands(op->getOperands().begin(), op->getOperands().end());

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
            buffSizeAndAlignment.push_back(std::make_pair(alignSizeForSwizzling(totalSize),
                                                          vpux::getAlignmentForSwizzling(vpux::SWIZZLING_KEY_5)));
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
// 2. Is padding required <- This case will be enabled with E#48057
bool AlignmentForSwizzling::canSwizzleWeights(VPUIP::NCEClusterTaskOp nceOp) {
    auto isTiled = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>() != nullptr;
    auto isFused = nceOp->hasAttr(vpux::ConstantFusing::constantsFused);
    auto weights = VPUIP::getTopBufferOfNCEClusterTiling(nceOp, nceOp.weights());
    auto weightsSM = VPUIP::getTopBufferOfNCEClusterTiling(nceOp, nceOp.weights_sparsity_map());
    auto weightTable = VPUIP::getTopBufferOfNCEClusterTiling(nceOp, nceOp.weight_table());

    // Swizzling for ELTWISE is handled with activation swizzling
    if (weights == nullptr || weightTable == nullptr || nceOp.task_type() == VPUIP::NCETaskType::ELTWISE) {
        return false;
    }

    // WA for disabling swizzling for compressed conv layer
    // E#56431 will remove workaround for a long term fix
    if (auto shapeCastOp = weights.getDefiningOp<VPUIP::ShapeCastOp>()) {
        return false;
    }

    if (isTiled) {
        auto checkDistributedContent = [](VPUIP::NCEClusterTilingOp clusterTilingOp) {
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
            if (isPaddingRequired(decOp, distrType)) {
                return false;
            }

            return true;
        };

        auto shapeCastOp = weights.getDefiningOp<VPUIP::ShapeCastOp>();

        auto bufferTilingOp = shapeCastOp ? shapeCastOp.source().getDefiningOp<VPUIP::NCEClusterTilingOp>()
                                          : weights.getDefiningOp<VPUIP::NCEClusterTilingOp>();
        if (!checkDistributedContent(bufferTilingOp)) {
            return false;
        }

        // Temporarily disable swizzling for WT buffer size not multiple of 512
        // TODO E#48057 Resize these buffers
        auto wtBufferTilingOp = weightTable.getDefiningOp<VPUIP::NCEClusterTilingOp>();
        if (!checkDistributedContent(wtBufferTilingOp)) {
            return false;
        }

        if (weightsSM != nullptr) {
            auto weightsSMClusterTilingCopyOp = weightsSM.getDefiningOp<VPUIP::NCEClusterTilingOp>();
            if (!checkDistributedContent(weightsSMClusterTilingCopyOp)) {
                return false;
            }
        }
    } else if (isFused) {
        VPUIP::CopyOp copyOp = nullptr;
        auto decOp = vpux::ConstantFusing::getConstAndCopyOp(weights, copyOp);
        if (decOp == nullptr) {
            return false;
        }
        auto requiredPadding = getRequiredPadding(decOp);
        if (requiredPadding) {
            return false;
        }
        return true;
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

        auto decOp = copyOp.input().getDefiningOp<Const::DeclareOp>();
        auto* allocOp = copyOp.output_buff().getDefiningOp();

        if (!mlir::isa_and_nonnull<mlir::memref::AllocOp>(allocOp)) {
            return false;
        }

        auto requiredPadding = getRequiredPadding(decOp);
        if (requiredPadding) {
            // TODO E#48057 Resize these buffers
            return false;
        }

        if (weightsSM != nullptr) {
            auto weightsSMCopyOp = weightsSM.getDefiningOp<VPUIP::CopyOp>();
            if (weightsSMCopyOp == nullptr) {
                return false;
            }

            auto weightsSMDecOp = weightsSMCopyOp.input().getDefiningOp<Const::DeclareOp>();
            if (getRequiredPadding(weightsSMDecOp)) {
                return false;
            }
        }
    }

    mlir::DenseSet<mlir::Value> operands = {weights, weightTable};
    if (weightsSM != nullptr) {
        operands.insert(weightsSM);
    }

    // Check if adding constants swizzling will not increase operation memory demand beyond CMX size
    if (!checkCMXUsage(isTiled ? nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>() : nceOp, operands)) {
        _log.trace("Do not enable weights swizzling because of increase in memory demand beyond CMX size, op - '{0}'",
                   nceOp->getLoc());
        return false;
    }

    return true;
}

template <typename InAllocOp, typename OutAllocOp>
void AlignmentForSwizzling::addSwizzlingAttributesToBuffer(mlir::OpBuilder& builder, InAllocOp inAllocOp,
                                                           mlir::Type newType) {
    auto swizzlingAttr = builder.getI64IntegerAttr(vpux::SWIZZLING_KEY_5);
    auto alignmentAttr = builder.getI64IntegerAttr(vpux::getAlignmentForSwizzling(vpux::SWIZZLING_KEY_5));

    builder.setInsertionPoint(inAllocOp);

    auto origLoc = inAllocOp->getLoc();
    auto newLoc = appendLoc(origLoc, "_alloc_swizzling");

    auto outAllocOp = builder.create<OutAllocOp>(newLoc, newType, alignmentAttr, swizzlingAttr);

    inAllocOp->replaceAllUsesWith(outAllocOp);
    inAllocOp->erase();
    _swizzledNceOperands.insert(outAllocOp->getResult(0));
}

void AlignmentForSwizzling::addPaddingAttrToBuffers(mlir::OpBuilder& builder, VPUIP::CopyOp& copyOp,
                                                    mlir::memref::AllocOp& allocOp) {
    builder.setInsertionPoint(copyOp);

    auto decOp = copyOp.input().getDefiningOp<Const::DeclareOp>();
    auto requiredPadding = getRequiredPadding(decOp);

    if (!requiredPadding) {
        return;
    }

    _log.trace("Padding required for op '{0}'", decOp->getLoc());
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
    copyOp = newCopyOp;
}

bool isTypeSwizzled(mlir::Type type) {
    VPUIP::MemRefAttr memRefAttr;
    if (auto bufferMemRefType = type.cast<mlir::MemRefType>()) {
        memRefAttr = bufferMemRefType.getLayout().dyn_cast<VPUIP::MemRefAttr>();
    } else if (auto distBufferType = type.cast<VPUIP::DistributedBufferType>()) {
        memRefAttr = distBufferType.getLayout().dyn_cast<VPUIP::MemRefAttr>();
    }

    if (memRefAttr && memRefAttr.swizzlingKey()) {
        return true;
    }

    return false;
}

void AlignmentForSwizzling::constantBufferSwizzling(mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp nceOp,
                                                    mlir::Value cst) {
    if (cst == nullptr) {
        return;
    }

    auto isTiled = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>() != nullptr;
    auto isFused = nceOp->hasAttr(vpux::ConstantFusing::constantsFused);
    auto constant = VPUIP::getTopBufferOfNCEClusterTiling(nceOp, cst);
    auto swizzlingAttr = builder.getI64IntegerAttr(SWIZZLING_KEY_5);
    auto shapeCastOp = constant.getDefiningOp<VPUIP::ShapeCastOp>();
    if (isFused || shapeCastOp) {
        adjustReturnTypesForInputChain(constant, SWIZZLING_KEY_5);
    }

    if (isTiled) {
        auto clusterTilingWithCopyOp = shapeCastOp != nullptr
                                               ? shapeCastOp.source().getDefiningOp<VPUIP::NCEClusterTilingOp>()
                                               : constant.getDefiningOp<VPUIP::NCEClusterTilingOp>();
        auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(clusterTilingWithCopyOp.getInnerTaskOp());
        auto distributedBuffer = clusterTilingWithCopyOp.output_buffs()[0].getDefiningOp<VPURT::AllocDistributed>();

        _log.trace("Enable swizzling for distributed constant buffer of NCE task - '{0}'", nceOp->getLoc());

        auto origType = distributedBuffer.getType().cast<VPUIP::DistributedBufferType>();

        // Create new DistributedBufferType which as part of layout will have swizzling set
        auto newType = getDistributedBufferTypeWithSwizzling(origType, swizzlingAttr);

        // Create new allocation with swizzling enabled and required alignment setting
        addSwizzlingAttributesToBuffer<VPURT::AllocDistributed, VPURT::AllocDistributed>(builder, distributedBuffer,
                                                                                         newType);

        // Create new CopyOp with correct result type
        builder.setInsertionPointAfter(copyOp);
        auto tempCopyOp = builder.create<VPUIP::CopyOp>(copyOp.getLoc(), newType.getCompactType(), copyOp.input(),
                                                        copyOp.output_buff());
        copyOp->replaceAllUsesWith(tempCopyOp);
        copyOp->erase();

        auto newNceClusterTilingOp =
                buildNewNceClusterTilingOp(builder, clusterTilingWithCopyOp, tempCopyOp.getOperation(), newType);

        updateNceClusterTilingOpResultUsers(newNceClusterTilingOp);

        tempCopyOp->erase();
        clusterTilingWithCopyOp->erase();
    } else {
        auto copyOp = constant.getDefiningOp<VPUIP::CopyOp>();
        if (isFused) {
            std::ignore = vpux::ConstantFusing::getConstAndCopyOp(constant, copyOp);
        }

        if (isTypeSwizzled(copyOp.output_buff().getType())) {
            // In case of constant folding buffer might have already been swizzled
            return;
        }

        auto allocOp = copyOp.output_buff().getDefiningOp<mlir::memref::AllocOp>();
        VPUX_THROW_WHEN(allocOp == nullptr, "Allocation operation was not identified");

        _log.trace("Enable swizzling for constant buffer of NCE task - '{0}'", nceOp->getLoc());
        addPaddingAttrToBuffers(builder, copyOp, allocOp);

        auto origType = allocOp.getType().cast<vpux::NDTypeInterface>();
        auto newType = getMemRefType(origType.getShape(), origType.getElementType(), origType.getDimsOrder(),
                                     swizzlingAttr, origType.getMemSpace());

        addSwizzlingAttributesToBuffer<mlir::memref::AllocOp, VPURT::Alloc>(builder, allocOp, newType);

        // Create new CopyOp with updated type
        builder.setInsertionPoint(copyOp);
        auto newCopyOp = builder.create<VPUIP::CopyOp>(copyOp.getLoc(), copyOp.input(), copyOp.output_buff());
        copyOp->replaceAllUsesWith(newCopyOp);
        copyOp->erase();
    }
}

bool AlignmentForSwizzling::canSwizzleActivation(VPUIP::NCEClusterTaskOp nceOp) {
    mlir::Value nceResult = nceOp.output();
    mlir::Value outputBuff = nceOp.output_buff();
    auto* op = nceOp.getOperation();
    // Check if wrapped in NCEClusterTiling
    if (auto nceClusterOp = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>()) {
        nceResult = nceClusterOp->getResult(0);
        outputBuff = nceClusterOp.output_buffs()[0];
        op = nceClusterOp.getOperation();
    }

    mlir::DenseSet<mlir::Operation*> userTasks;

    // Find DPU->DPU buffers that can be swizzled
    for (auto* user : nceResult.getUsers()) {
        auto* userTask = user;
        userTasks.insert(userTask);
        if (auto nceClusterUser = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(user)) {
            userTask = nceClusterUser.getInnerTaskOp();
        }

        if (!mlir::isa<VPUIP::NCEClusterTaskOp>(userTask)) {
            return false;
        }
    }

    auto* bufOp = outputBuff.getDefiningOp();

    if (!mlir::isa_and_nonnull<mlir::memref::AllocOp, VPURT::AllocDistributed>(bufOp)) {
        return false;
    }

    // Check if adding activation swizzling will not increase operation memory demand beyond CMX size
    if (!checkCMXUsage(op, {outputBuff})) {
        _log.trace(
                "Do not enable activation swizzling because of increase in memory demand beyond CMX size, op - '{0}'",
                nceOp->getLoc());
        return false;
    }

    // Check if adding activation swizzling on output of this op will not increase memory demand
    // beyond CMX size of user operations
    for (auto* userTask : userTasks) {
        if (!checkCMXUsage(userTask, {nceResult})) {
            _log.trace("Do not enable activation swizzling because of increase in memory demand beyond CMX size of "
                       "user task, op - '{0}', user task - '{1}'",
                       nceOp->getLoc(), userTask->getLoc());
            return false;
        }
    }

    return true;
}

void AlignmentForSwizzling::activationBufferSwizzling(mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp nceOp) {
    mlir::Value nceResult = nceOp.output();
    mlir::Value outputBuff = nceOp.output_buff();
    // Check if wrapped in NCEClusterTiling
    if (auto nceClusterOp = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>()) {
        nceResult = nceClusterOp->getResult(0);
        outputBuff = nceClusterOp.output_buffs()[0];
    }

    auto* bufOp = outputBuff.getDefiningOp();

    if (!mlir::isa_and_nonnull<mlir::memref::AllocOp, VPURT::AllocDistributed>(bufOp)) {
        return;
    }

    auto swizzlingAttr = builder.getI64IntegerAttr(SWIZZLING_KEY_5);
    auto alignmentAttr = builder.getI64IntegerAttr(getAlignmentForSwizzling(SWIZZLING_KEY_5));

    auto origType = (*bufOp->getResultTypes().begin()).cast<vpux::NDTypeInterface>();

    _log.trace("Enable swizzling for output buffer of NCE task - '{0}'", nceOp->getLoc());

    builder.setInsertionPoint(bufOp);
    auto newLoc = appendLoc(bufOp->getLoc(), "_alloc_swizzling");

    if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(bufOp)) {
        // Create new MemRefType which as part of layout will have swizzling set
        auto newType = getMemRefType(origType.getShape(), origType.getElementType(), origType.getDimsOrder(),
                                     swizzlingAttr, origType.getMemSpace());

        // Create new allocation with swizzling enabled and required alignment setting
        auto newAlloc = builder.create<VPURT::Alloc>(newLoc, newType, alignmentAttr, swizzlingAttr);
        allocOp->replaceAllUsesWith(newAlloc);

        _swizzledNceOperands.insert(newAlloc.buffer());

        _opsToRemove.push_back(allocOp.getOperation());

        nceOp.getResult(0).setType(newType);
    } else if (auto allocOp = mlir::dyn_cast<VPURT::AllocDistributed>(bufOp)) {
        auto distributedType = origType.dyn_cast<VPUIP::DistributedBufferType>();

        // Create new DistributedBufferType which as part of layout will have swizzling set
        auto newType = getDistributedBufferTypeWithSwizzling(distributedType, swizzlingAttr);

        // Create new allocation with swizzling enabled and required alignment setting
        auto newAlloc = builder.create<VPURT::AllocDistributed>(newLoc, newType, alignmentAttr, swizzlingAttr);
        allocOp->replaceAllUsesWith(newAlloc);

        _swizzledNceOperands.insert(newAlloc.buffer());

        _opsToRemove.push_back(allocOp.getOperation());

        nceOp.getResult(0).setType(newType.getCompactType());

        auto nceClusterTilingOp = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
        VPUX_THROW_WHEN(nceClusterTilingOp == nullptr, "No NCEClusterTiling op located");

        // Create new NCEClusterTask wrapped in NCEClusterTiling with updated result type
        auto newNceClusterTilingOp =
                buildNewNceClusterTilingOp(builder, nceClusterTilingOp, nceOp.getOperation(), newType);

        updateNceClusterTilingOpResultUsers(newNceClusterTilingOp);

        _opsToRemove.push_back(nceOp.getOperation());
        _opsToRemove.push_back(nceClusterTilingOp.getOperation());
    }
}

//
// safeRunOnFunc
//

void AlignmentForSwizzling::safeRunOnFunc() {
    auto func = getFunction();

    auto module = func->getParentOfType<mlir::ModuleOp>();

    _cmxSize = VPU::getTotalCMXSize(module).count();

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
                constantBufferSwizzling(builder, nceOp, nceOp.weights_sparsity_map());
                constantBufferSwizzling(builder, nceOp, nceOp.weight_table());
            }
        }
        if (_enableActivationSwizzling) {
            if (canSwizzleActivation(nceOp)) {
                _opsForActivationSwizzling.push_back(nceOp);
            }
        }
    });

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
            auto input1NceTaskItr = _opsForActivationSwizzling.end();
            if (input1NceTask) {
                input1NceTaskItr =
                        std::find(_opsForActivationSwizzling.begin(), _opsForActivationSwizzling.end(), input1NceTask);
                input1SwizzlingFlag = (input1NceTaskItr != _opsForActivationSwizzling.end());
            }

            bool input2SwizzlingFlag = false;
            auto input2NceTaskItr = _opsForActivationSwizzling.end();
            if (input2NceTask) {
                input2NceTaskItr =
                        std::find(_opsForActivationSwizzling.begin(), _opsForActivationSwizzling.end(), input2NceTask);
                input2SwizzlingFlag = (input2NceTaskItr != _opsForActivationSwizzling.end());
            }

            if (input1SwizzlingFlag != input2SwizzlingFlag) {
                _log.trace("Mismatch of swizzling setting of eltwise inputs, eltwise op - '{0}'", nceOp->getLoc());
                if (input1SwizzlingFlag) {
                    _opsForActivationSwizzling.erase(input1NceTaskItr);
                    _log.nest().trace("Swizzling cannot be enabled on op - '{0}'", input1NceTask->getLoc());
                } else if (input2SwizzlingFlag) {
                    _opsForActivationSwizzling.erase(input2NceTaskItr);
                    _log.nest().trace("Swizzling cannot be enabled on op - '{0}'", input2NceTask->getLoc());
                }
            }
        });
    } while (anyEltwiseBuffersUpdated);

    // After identifying all qualified for swizzling ops and removing those which
    // due to eltwise input swizzling mismatch cannot be enabled, perform
    // actual swizzling enabling
    for (auto nceOp : llvm::make_early_inc_range(_opsForActivationSwizzling)) {
        activationBufferSwizzling(builder, nceOp);
    }

    for (auto opToRemove : llvm::make_early_inc_range(_opsToRemove)) {
        opToRemove->erase();
    }
}

}  // namespace

//
// createAlignmentForSwizzling
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createAlignmentForSwizzling(bool enableWeightSwizzling,
                                                                     bool enableActivationSwizzling, Logger log) {
    return std::make_unique<AlignmentForSwizzling>(enableWeightSwizzling, enableActivationSwizzling, log);
}
