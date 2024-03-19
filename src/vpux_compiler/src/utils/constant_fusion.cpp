//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/constant_fusion.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/utils/IE/float16.hpp"

using namespace vpux;

// This function is a recursive helper implementation of getConstAndDma
// It keeps on parsing the parent op and looks for the DeclareOp
// Once found stores the Op and returns the delcare Op
Const::DeclareOp getConstAndDmaRecImpl(mlir::BlockArgument arg, mlir::async::ExecuteOp execParentOp,
                                       mlir::Operation** constOp) {
    if (arg == nullptr || execParentOp == nullptr) {
        return nullptr;
    }

    // Adjust the index by adding dependencies size
    auto dependenciesSize = execParentOp.getDependencies().size();
    auto indexOfFusedConstant = arg.getArgNumber() + static_cast<int32_t>(dependenciesSize);

    // GoTo parent of the arg
    auto tempExecOp = execParentOp->getOperand(indexOfFusedConstant).getDefiningOp<mlir::async::ExecuteOp>();
    auto* tempBodyBlock = tempExecOp.getBody();
    for (auto& tempOp : tempBodyBlock->getOperations()) {
        auto tilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(tempOp);
        auto* op = tilingOp ? tilingOp.getInnerTaskOp() : &tempOp;
        if (!mlir::isa<VPUIP::CopyOp, VPUIP::NNDMAOp>(op)) {
            continue;
        }

        auto type = op->getResult(0).getType();
        if (auto ndType = type.cast<vpux::NDTypeInterface>()) {
            // For constant fusion this should always be U8 or F16
            if (!ndType.getElementType().isUnsignedInteger(8) && !ndType.getElementType().isF16()) {
                continue;
            }

            auto cstValue = mlir::dyn_cast<VPUIP::LayerOpInterface>(op).getInputs()[0];
            if (tilingOp) {
                auto blkArg = mlir::dyn_cast<VPUIP::LayerOpInterface>(op).getInputs()[0].cast<mlir::BlockArgument>();
                cstValue = tilingOp->getOperand(blkArg.getArgNumber());
            }

            if (auto constDeclareOp = cstValue.getDefiningOp<Const::DeclareOp>()) {
                *constOp = op;
                return constDeclareOp;
            }

            // Op is produced by other operation. By checking other users of this buffer
            // identify the one with const as input which would be the initial op loading searched constant
            auto lookUp =
                    tilingOp ? tilingOp.getOperand(0) : mlir::dyn_cast<VPUIP::LayerOpInterface>(op).getInputs()[0];
            for (auto user : lookUp.getUsers()) {
                if (auto userTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(user)) {
                    auto newDecOp = userTilingOp.getOperand(0).getDefiningOp<Const::DeclareOp>();
                    if (newDecOp != nullptr) {
                        *constOp = userTilingOp.getInnerTaskOp();
                        return newDecOp;
                    }
                }
                if (mlir::isa<VPUIP::CopyOp, VPUIP::NNDMAOp>(user)) {
                    if (auto newDecOp = mlir::dyn_cast<VPUIP::LayerOpInterface>(user)
                                                .getInputs()[0]
                                                .getDefiningOp<Const::DeclareOp>()) {
                        *constOp = user;
                        return newDecOp;
                    }
                }
            }

            // Op wrapped in async.execute has input but not found in this block
            // continue traversing by checking producer/parent of this argument
            arg = mlir::dyn_cast<VPUIP::LayerOpInterface>(op).getInputs()[0].dyn_cast<mlir::BlockArgument>();
            execParentOp = op->getParentOfType<mlir::async::ExecuteOp>();
            return getConstAndDmaRecImpl(arg, execParentOp, constOp);
        }
    }
    return nullptr;
}

// Get the underlying Declare and Copy Op for the constant passed
// If not found on the first level recursively parse the parents of the Op until a DeclareOp is found
Const::DeclareOp ConstantFusing::getConstAndDma(VPUIP::NCEClusterTaskOp nceOp, mlir::Value constant,
                                                mlir::Operation** constOp) {
    Const::DeclareOp constDeclareOp = nullptr;
    VPUIP::ViewOp viewOp = nullptr;

    if (constant == nullptr) {
        return nullptr;
    }

    auto iArg = constant.dyn_cast<mlir::BlockArgument>();
    if (iArg != nullptr) {
        auto parentTilingOp = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
        viewOp = parentTilingOp->getOperand(iArg.getArgNumber()).getDefiningOp<VPUIP::ViewOp>();
        VPUX_THROW_UNLESS(viewOp != nullptr, "Tiled Constant found without a ViewOp");
    } else {
        viewOp = constant.getDefiningOp<VPUIP::ViewOp>();
        VPUX_THROW_UNLESS(viewOp != nullptr, "Constant found without a ViewOp");
    }

    auto subViewOp = viewOp.getSource().getDefiningOp<VPUIP::SubViewOp>();
    VPUX_THROW_UNLESS(subViewOp != nullptr, "SubViewOp expected as source for ViewOp for tensor fusion");
    mlir::Value source = subViewOp.getSource();

    if (mlir::BlockArgument arg = source.dyn_cast<mlir::BlockArgument>()) {
        // Op wrapped in async.execute has input continue traversing by checking producer of this argument
        auto execParentOp = subViewOp->getParentOfType<mlir::async::ExecuteOp>();
        return getConstAndDmaRecImpl(arg, execParentOp, constOp);
    }

    if (auto declareBuffer = source.getDefiningOp<VPURT::DeclareBufferOp>()) {
        for (auto user : declareBuffer->getUsers()) {
            if (auto clusterOp = mlir::dyn_cast_or_null<VPUIP::NCEClusterTilingOp>(user)) {
                *constOp = clusterOp.getInnerTaskOp();
                constDeclareOp = clusterOp.getOperand(0).getDefiningOp<Const::DeclareOp>();
                break;
            }
        }
    }

    if (auto allocDistributed = source.getDefiningOp<VPURT::AllocDistributed>()) {
        for (auto user : allocDistributed->getUsers()) {
            if (auto clusterOp = mlir::dyn_cast_or_null<VPUIP::NCEClusterTilingOp>(user)) {
                *constOp = clusterOp.getInnerTaskOp();
                constDeclareOp = clusterOp.getOperand(0).getDefiningOp<Const::DeclareOp>();
                break;
            }
        }
    }

    if (auto* op = source.getDefiningOp()) {
        constDeclareOp = mlir::dyn_cast<VPUIP::LayerOpInterface>(op).getInputs()[0].getDefiningOp<Const::DeclareOp>();

        while (constDeclareOp == nullptr) {
            op = mlir::dyn_cast<VPUIP::LayerOpInterface>(op).getInputs()[0].getDefiningOp();
            VPUX_THROW_UNLESS(op != nullptr, "Next CopyOp or NNDMAOp as source operation expected");

            constDeclareOp =
                    mlir::dyn_cast<VPUIP::LayerOpInterface>(op).getInputs()[0].getDefiningOp<Const::DeclareOp>();
        }
        *constOp = op;
    }

    if (auto clusterOp = source.getDefiningOp<VPUIP::NCEClusterTilingOp>()) {
        VPUX_THROW_WHEN(clusterOp.getInputs().empty(), "NCEClusterTiling op has no inputs - '{0}'",
                        clusterOp->getLoc());
        constDeclareOp = clusterOp.getInputs()[0].getDefiningOp<Const::DeclareOp>();

        while (constDeclareOp == nullptr) {
            clusterOp = clusterOp.getInputs()[0].getDefiningOp<VPUIP::NCEClusterTilingOp>();
            VPUX_THROW_UNLESS(clusterOp != nullptr, "Next NCEClusterTiling as source operation expected");

            VPUX_THROW_WHEN(clusterOp.getInputs().empty(), "NCEClusterTiling op has no inputs - '{0}'",
                            clusterOp->getLoc());
            constDeclareOp = clusterOp.getInputs()[0].getDefiningOp<Const::DeclareOp>();
        }
        *constOp = clusterOp.getInnerTaskOp();
    }
    return constDeclareOp;
}

int32_t ConstantFusing::getOffsetForConstant(VPUIP::NCEClusterTaskOp& nceOp, mlir::Value constant) {
    int32_t offset = 0;
    VPUIP::ViewOp viewOp = nullptr;
    if (constant == nullptr) {
        return offset;
    }

    auto arg = constant.dyn_cast<mlir::BlockArgument>();
    if (arg != nullptr) {
        auto execParentOp = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
        viewOp = execParentOp->getOperand(arg.getArgNumber()).getDefiningOp<VPUIP::ViewOp>();
        VPUX_THROW_UNLESS(viewOp != nullptr, "Tiled Constant found without a ViewOp");
    } else {
        viewOp = constant.getDefiningOp<VPUIP::ViewOp>();
        VPUX_THROW_UNLESS(viewOp != nullptr, "Getting Offset: Constant found without a ViewOp");
    }

    auto subViewOp = viewOp.getSource().getDefiningOp<VPUIP::SubViewOp>();
    VPUX_THROW_UNLESS(subViewOp != nullptr, "SubViewOp expected as source for ViewOp for tensor fusion");

    auto offsets = subViewOp.getStaticOffsets();
    return parseIntArrayAttr<int32_t>(offsets).back();
}

VPUIP::DistributedBufferType ConstantFusing::getDistributedBufferType(VPUIP::DistributedBufferType origDistType,
                                                                      Const::DeclareOp declOp,
                                                                      mlir::PatternRewriter& rewriter) {
    auto typeInterface = declOp.getOutput().getType().cast<vpux::NDTypeInterface>();

    const auto ctx = typeInterface.getContext();
    const auto order = typeInterface.getDimsOrder();
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
    const auto strides = typeInterface.getStrides();
    const Bit elemSize = typeInterface.getElemTypeSize();

    const auto elemStrides = to_small_vector(strides | transformed([&](Bit stride) {
                                                 return stride.count() / elemSize.count();
                                             }));

    const auto stridesAttr = getIntArrayAttr(ctx, elemStrides);
    const auto layoutAttr = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                                  /*allocSize=*/nullptr, ctx);

    vpux::IndexedSymbolAttr memKindAttr =
            IndexedSymbolAttr::get(rewriter.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN));

    // Create updated distributedTensorAttr, remove alignment as the fused buffer is a flat buffer
    auto origDistributedTensorAttr = origDistType.getDistribution();

    if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(origDistributedTensorAttr)) {
        VPUX_THROW_WHEN(origDistributedTensorAttr.getMode().getValue() != VPU::DistributionMode::DUPLICATED,
                        "DistributedBuffer for fused constant has mode different from DUPLICATED, type = {0}",
                        origDistType);

        auto newDistribution =
                VPU::getNonOverlappedDistributedAttr(typeInterface.getShape(), origDistributedTensorAttr.getMode(),
                                                     nullptr, origDistributedTensorAttr.getNumClusters(), nullptr,
                                                     origDistributedTensorAttr.getUniformDistributedSegments(), ctx);

        return VPUIP::DistributedBufferType::get(ctx, typeInterface.getShape().raw(), typeInterface.getElementType(),
                                                 layoutAttr, memKindAttr, newDistribution);
    }

    auto distributedTensorAttr = VPU::DistributedTensorAttr::get(
            ctx, origDistributedTensorAttr.getMode(), origDistributedTensorAttr.getNumTiles(), nullptr, nullptr,
            nullptr, origDistributedTensorAttr.getNumClusters(), nullptr,
            origDistributedTensorAttr.getUniformDistributedSegments(), origDistributedTensorAttr.getComputeShapes(),
            origDistributedTensorAttr.getComputeOffsets(), origDistributedTensorAttr.getMemoryShapes(),
            origDistributedTensorAttr.getMemoryOffsets(), origDistributedTensorAttr.getEqualMemoryAndComputeView());

    return VPUIP::DistributedBufferType::get(ctx, typeInterface.getShape().raw(), typeInterface.getElementType(),
                                             layoutAttr, memKindAttr, distributedTensorAttr);
}

void ConstantFusing::getCopyAndDeclareOpForFusion(VPUIP::NCEClusterTaskOp& nceOp, mlir::Value constant,
                                                  VPUIP::CopyOp& copyOp, Const::DeclareOp& declareOp,
                                                  VPURT::AllocDistributed& foundAllocDistributed,
                                                  VPUIP::NCEClusterTilingOp& tilingOp) {
    if (constant == nullptr) {
        return;
    }
    auto arg = constant.dyn_cast<mlir::BlockArgument>();

    // Op is Tiled
    if (arg != nullptr) {
        auto execParentOp = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
        tilingOp = execParentOp->getOperand(arg.getArgNumber()).getDefiningOp<VPUIP::NCEClusterTilingOp>();

        // This could be the case when the parent is not a tiled op directly instead a
        // ShapeCastOp preceded by a NCEClusterTilingOp don't fuse such constants for now
        if (tilingOp == nullptr) {
            return;
        }

        // Get distribution mode
        foundAllocDistributed = tilingOp.getOutputBuffs()[0].getDefiningOp<VPURT::AllocDistributed>();
        auto inputType = foundAllocDistributed.getType().dyn_cast<VPUIP::DistributedBufferType>();
        auto isDuplicated = inputType.getDistribution().getMode().getValue() == VPU::DistributionMode::DUPLICATED;

        // Only Fuse if the constants are broadcasted/duplicated
        if (isDuplicated) {
            copyOp = tilingOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
            declareOp = tilingOp.getOperand(0).getDefiningOp<Const::DeclareOp>();
        }
    } else {
        // If the constant isn't a block arg the parent Op is not tiled so just get the declare Op
        auto tempCopyOp = constant.getDefiningOp<VPUIP::CopyOp>();
        if (tempCopyOp != nullptr) {
            copyOp = tempCopyOp;
            declareOp = copyOp.getInput().getDefiningOp<Const::DeclareOp>();

            while (declareOp == nullptr) {
                copyOp = copyOp.getInput().getDefiningOp<VPUIP::CopyOp>();
                // If this is the case then the constant is not spilled, To be handled with E#45105
                if (copyOp == nullptr) {
                    // Return nullptr, will skip fusion for this layer
                    break;
                }

                declareOp = copyOp.getInput().getDefiningOp<Const::DeclareOp>();
            }
        }
    }
}
