//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/constant_fusion.hpp"
#include "vpux/utils/IE/float16.hpp"

using namespace vpux;

void ConstantFusing::convertInputToI32(Const::details::ContentRange<uint8_t>& inValues,
                                       std::vector<int32_t>& outValues) {
    auto size = inValues.size();
    VPUX_THROW_UNLESS(size % 4 == 0, "size of inValues expected to be multiple of 4 but found {0}", size);
    for (auto i = 0; i < static_cast<int>(size); i += 4) {
        const uint32_t i32 =
                (inValues[i + 0] << 0) | (inValues[i + 1] << 8) | (inValues[i + 2] << 16) | (inValues[i + 3] << 24);
        outValues.push_back(i32);
    }
}

// This function is a recursive helper implementation of getConstAndCopyOp
// It keeps on parsing the parent op and looks for the DeclareOp
// Once found stores the copyOp and returns the delcare Op
Const::DeclareOp getConstAndCopyOpRecImpl(mlir::BlockArgument arg, mlir::async::ExecuteOp execParentOp,
                                          VPUIP::CopyOp& constCopyOp) {
    if (arg == nullptr || execParentOp == nullptr) {
        return nullptr;
    }

    // Adjust the index by adding dependencies size
    auto dependenciesSize = execParentOp.dependencies().size();
    auto indexOfFusedConstant = arg.getArgNumber() + static_cast<int32_t>(dependenciesSize);

    // GoTo parent of the arg
    auto tempExecOp = execParentOp->getOperand(indexOfFusedConstant).getDefiningOp<mlir::async::ExecuteOp>();
    auto* tempBodyBlock = &tempExecOp.body().front();
    for (auto& tempOp : tempBodyBlock->getOperations()) {
        auto tilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(tempOp);
        auto copyOp = tilingOp ? mlir::dyn_cast<VPUIP::CopyOp>(tilingOp.getInnerTaskOp())
                               : mlir::dyn_cast<VPUIP::CopyOp>(tempOp);
        if (copyOp == nullptr) {
            continue;
        }

        auto type = copyOp.getType();
        if (auto ndType = type.cast<vpux::NDTypeInterface>()) {
            // For constant fusion this should always be U8
            if (!ndType.getElementType().isUnsignedInteger(8)) {
                continue;
            }

            auto cstValue = copyOp.input();
            if (tilingOp) {
                auto blkArg = copyOp.input().cast<mlir::BlockArgument>();
                cstValue = tilingOp->getOperand(blkArg.getArgNumber());
            }

            if (auto constDeclareOp = cstValue.getDefiningOp<Const::DeclareOp>()) {
                constCopyOp = copyOp;
                return constDeclareOp;
            }

            // CopyOp is produced by other operation. By checking other users of this buffer
            // identify the one with const as input which would be the initial copyOp loading searched constant
            auto lookUp = tilingOp ? tilingOp.getOperand(0) : copyOp.input();
            for (auto user : lookUp.getUsers()) {
                if (auto userTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(user)) {
                    auto newDecOp = userTilingOp.getOperand(0).getDefiningOp<Const::DeclareOp>();
                    if (newDecOp != nullptr) {
                        constCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(userTilingOp.getInnerTaskOp());
                        return newDecOp;
                    }
                }
                if (auto iCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(user)) {
                    if (auto newDecOp = iCopyOp.input().getDefiningOp<Const::DeclareOp>()) {
                        constCopyOp = iCopyOp;
                        return newDecOp;
                    }
                }
            }

            // CopyOp wrapped in async.execute has input but not found in this block
            // continue traversing by checking producer/parent of this argument
            arg = copyOp.input().dyn_cast<mlir::BlockArgument>();
            execParentOp = copyOp->getParentOfType<mlir::async::ExecuteOp>();
            return getConstAndCopyOpRecImpl(arg, execParentOp, constCopyOp);
        }
    }
    return nullptr;
}

// Get the underlying Declare and Copy Op for the constant passed
// If not found on the first level recursively parse the parents of the Op until a DeclareOp is found
Const::DeclareOp ConstantFusing::getConstAndCopyOp(VPUIP::NCEClusterTaskOp nceOp, mlir::Value constant,
                                                   VPUIP::CopyOp& constCopyOp) {
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

    auto subViewOp = viewOp.source().getDefiningOp<VPUIP::SubViewOp>();
    VPUX_THROW_UNLESS(subViewOp != nullptr, "SubViewOp expected as source for ViewOp for tensor fusion");
    mlir::Value source = subViewOp.source();

    if (mlir::BlockArgument arg = source.dyn_cast<mlir::BlockArgument>()) {
        // CopyOp wrapped in async.execute has input continue traversing by checking producer of this argument
        auto execParentOp = subViewOp->getParentOfType<mlir::async::ExecuteOp>();
        return getConstAndCopyOpRecImpl(arg, execParentOp, constCopyOp);
    }

    if (auto declareBuffer = source.getDefiningOp<VPURT::DeclareBufferOp>()) {
        for (auto user : declareBuffer->getUsers()) {
            if (auto clusterOp = mlir::dyn_cast_or_null<VPUIP::NCEClusterTilingOp>(user)) {
                constCopyOp = clusterOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
                constDeclareOp = clusterOp.getOperand(0).getDefiningOp<Const::DeclareOp>();
                break;
            }
        }
    }

    if (auto allocDistributed = source.getDefiningOp<VPURT::AllocDistributed>()) {
        for (auto user : allocDistributed->getUsers()) {
            if (auto clusterOp = mlir::dyn_cast_or_null<VPUIP::NCEClusterTilingOp>(user)) {
                constCopyOp = clusterOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
                constDeclareOp = clusterOp.getOperand(0).getDefiningOp<Const::DeclareOp>();
                break;
            }
        }
    }

    if (auto copyOp = source.getDefiningOp<VPUIP::CopyOp>()) {
        constDeclareOp = copyOp.input().getDefiningOp<Const::DeclareOp>();

        while (constDeclareOp == nullptr) {
            copyOp = copyOp.input().getDefiningOp<VPUIP::CopyOp>();
            VPUX_THROW_UNLESS(copyOp != nullptr, "Next CopyOp as source operation expected");

            constDeclareOp = copyOp.input().getDefiningOp<Const::DeclareOp>();
        }
        constCopyOp = copyOp;
    }

    if (auto clusterOp = source.getDefiningOp<VPUIP::NCEClusterTilingOp>()) {
        VPUX_THROW_WHEN(clusterOp.inputs().empty(), "NCEClusterTiling op has no inputs - '{0}'", clusterOp->getLoc());
        constDeclareOp = clusterOp.inputs()[0].getDefiningOp<Const::DeclareOp>();

        while (constDeclareOp == nullptr) {
            clusterOp = clusterOp.inputs()[0].getDefiningOp<VPUIP::NCEClusterTilingOp>();
            VPUX_THROW_UNLESS(clusterOp != nullptr, "Next NCEClusterTiling as source operation expected");

            VPUX_THROW_WHEN(clusterOp.inputs().empty(), "NCEClusterTiling op has no inputs - '{0}'",
                            clusterOp->getLoc());
            constDeclareOp = clusterOp.inputs()[0].getDefiningOp<Const::DeclareOp>();
        }
        constCopyOp = clusterOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
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

    auto subViewOp = viewOp.source().getDefiningOp<VPUIP::SubViewOp>();
    VPUX_THROW_UNLESS(subViewOp != nullptr, "SubViewOp expected as source for ViewOp for tensor fusion");

    auto offsets = subViewOp.static_offsets();
    return parseIntArrayAttr<int32_t>(offsets).back();
}

VPUIP::DistributedBufferType ConstantFusing::getDistributedBufferType(VPUIP::DistributedBufferType origDistType,
                                                                      Const::DeclareOp declOp,
                                                                      mlir::PatternRewriter& rewriter) {
    auto typeInterface = declOp.output().getType().cast<vpux::NDTypeInterface>();

    const auto ctx = typeInterface.getContext();
    const auto order = typeInterface.getDimsOrder();
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
    const auto strides = typeInterface.getStrides();
    const auto elemSize = typeInterface.getElemTypeSize();

    const auto elemStrides = to_small_vector(strides | transformed([&](Bit stride) {
                                                 return stride.count() / elemSize.count();
                                             }));

    const auto stridesAttr = getIntArrayAttr(ctx, elemStrides);
    const auto layoutAttr = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, /*swizzlingScheme=*/nullptr, nullptr,
                                                   /*allocSize=*/nullptr, ctx);

    vpux::IndexedSymbolAttr memKindAttr =
            IndexedSymbolAttr::get(rewriter.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN));

    // Create updated distributedTensorAttr, remove alignment as the fused buffer is a flat buffer
    auto origDistributedTensorAttr = origDistType.getDistribution();
    auto distributedTensorAttr = VPU::DistributedTensorAttr::get(
            origDistributedTensorAttr.mode(), origDistributedTensorAttr.num_tiles(), nullptr, nullptr, nullptr,
            origDistributedTensorAttr.num_clusters(), nullptr, origDistributedTensorAttr.uniform_distributed_segments(),
            origDistributedTensorAttr.compute_shapes(), origDistributedTensorAttr.compute_offsets(),
            origDistributedTensorAttr.equal_memory_and_compute_view(), ctx);

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
        foundAllocDistributed = tilingOp.output_buffs()[0].getDefiningOp<VPURT::AllocDistributed>();
        auto inputType = foundAllocDistributed.getType().dyn_cast<VPUIP::DistributedBufferType>();
        auto isDuplicated = inputType.getDistribution().mode().getValue() == VPU::DistributionMode::DUPLICATED;

        // Only Fuse if the constants are broadcasted/duplcicated
        if (isDuplicated) {
            copyOp = tilingOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
            declareOp = tilingOp.getOperand(0).getDefiningOp<Const::DeclareOp>();
        }
    } else {
        // If the constant isn't a block arg the parent Op is not tiled so just get the declare Op
        auto tempCopyOp = constant.getDefiningOp<VPUIP::CopyOp>();
        if (tempCopyOp != nullptr) {
            copyOp = tempCopyOp;
            declareOp = copyOp.input().getDefiningOp<Const::DeclareOp>();

            while (declareOp == nullptr) {
                // If this is the case then the constant is not spilled, To be handled with E#45105
                if (VPUIP::isPureViewOp(copyOp.input().getDefiningOp())) {
                    // Return nullptr, will skip fusion for this layer
                    break;
                }
                copyOp = copyOp.input().getDefiningOp<VPUIP::CopyOp>();
                VPUX_THROW_UNLESS(copyOp != nullptr, "Next CopyOp as source operation expected");

                declareOp = copyOp.input().getDefiningOp<Const::DeclareOp>();
            }
        }
    }
}
