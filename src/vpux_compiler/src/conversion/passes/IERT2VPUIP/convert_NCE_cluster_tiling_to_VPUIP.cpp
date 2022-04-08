//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BlockAndValueMapping.h>

using namespace vpux;

namespace {

using GroupedAllocMap = DenseMap<mlir::Operation*, SmallVector<mlir::Operation*>>;

//
// NCEClusterTilingRewriter
//

class NCEClusterTilingRewriter final : public mlir::OpConversionPattern<VPU::NCEClusterTilingOp> {
public:
    NCEClusterTilingRewriter(mlir::TypeConverter& converter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::NCEClusterTilingOp>(converter, ctx), _ctx(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEClusterTilingOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    void findGroupedAllocOps(mlir::Block& block, GroupedAllocMap& groupedAllocOps,
                             SmallVector<mlir::Operation*>& skipAllocOps) const;

private:
    mlir::MLIRContext* _ctx;
    Logger _log;
};

// Find the mapping from the initial grouping ops to their source allocations found inside the block body
void NCEClusterTilingRewriter::findGroupedAllocOps(mlir::Block& block, GroupedAllocMap& groupedAllocOps,
                                                   SmallVector<mlir::Operation*>& skipAllocOps) const {
    block.walk([&](vpux::GroupedViewOpInterface groupedViewOp) {
        SmallVector<mlir::Value> newOperands;
        for (auto source : groupedViewOp.getViewSources()) {
            if (source == nullptr) {
                continue;
            }
            auto allocOp = source.getDefiningOp();
            if (!isBufAllocOp(allocOp)) {
                continue;
            }
            groupedAllocOps[groupedViewOp].push_back(allocOp);
            skipAllocOps.push_back(allocOp);
        }

        const auto allocationsIt = groupedAllocOps.find(groupedViewOp);
        const auto hasAllocations = allocationsIt != groupedAllocOps.end();
        VPUX_THROW_WHEN(hasAllocations && groupedViewOp->getNumOperands() != allocationsIt->second.size(),
                        "Not all operands of the grouped view operation are allocations. Expected {0}, got {1}",
                        groupedViewOp->getNumOperands(), allocationsIt->second.size());

        if (hasAllocations) {
            return;
        }

        // Cover the case where the source values are aliases to the output buffers of an operation in the body
        // Currently this only happens for VPUIP::NCEClusterTask
        for (auto source : groupedViewOp.getViewSources()) {
            auto producerOp = source.getDefiningOp<VPUIP::NCEClusterTaskOp>();
            if (producerOp == nullptr) {
                continue;
            }
            const auto resultIt = llvm::find(producerOp->getResults(), source);
            const auto resultIdx = std::distance(producerOp->getResults().begin(), resultIt);
            auto operand = producerOp.getViewSource(resultIdx);
            auto operandParentOp = operand.getDefiningOp();
            if (!isBufAllocOp(operandParentOp)) {
                _log.nest().trace("Parent op '{0}' of operand '{1}' is not an allocation op", operand, operandParentOp);
                continue;
            }
            if (operandParentOp->getParentRegion() != groupedViewOp->getParentRegion()) {
                _log.nest().trace("Parent op '{0}' of operand '{1}' is found in a different region", operand,
                                  operandParentOp);
                continue;
            }
            groupedAllocOps[groupedViewOp].push_back(operandParentOp);
            skipAllocOps.push_back(operandParentOp);
        }
    });
}

mlir::LogicalResult NCEClusterTilingRewriter::matchAndRewrite(VPU::NCEClusterTilingOp origOp, OpAdaptor newArgs,
                                                              mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    VPUX_THROW_UNLESS(origOp.results().size() == 1,
                      "Currently only single output NCEClusterTiling operation is supported, got {0} outputs",
                      origOp.results().size());

    // Initialize VPUIP::NCEClusterTiling operands with operands of VPU::NCEClusterTiling op
    // Output buffs will be added based on identified allocation operations
    SmallVector<mlir::Value> inputsOutputOperands = newArgs.operands();
    SmallVector<mlir::Value> origOutputBuffers;

    const auto clusterTilingBufferType = typeConverter->convertType(origOp.results()[0].getType());

    auto outputDataDistType = clusterTilingBufferType.dyn_cast<VPUIP::DistributedBufferType>();
    if (auto sparseType = clusterTilingBufferType.dyn_cast<VPUIP::SparseBufferType>()) {
        outputDataDistType = sparseType.getData().dyn_cast<VPUIP::DistributedBufferType>();
    }

    auto& origOpBodyBlock = origOp.body().front();

    GroupedAllocMap groupedAllocOps;
    SmallVector<mlir::Operation*> skipAllocOps;
    findGroupedAllocOps(origOpBodyBlock, groupedAllocOps, skipAllocOps);

    rewriter.setInsertionPoint(origOp);

    // The goal is to move the allocation operations outside the NCEClusterTiling body.
    // In case the resulting type of NCEClusterTiling is a distributed type, these allocations
    // should also produce distributed types now that they are outside.
    // This function will ensure the correct buffer type is allocated.
    const auto createNewAllocOp = [&](mlir::Operation* op) {
        const auto outputType = op->getResult(0).getType();
        if (outputDataDistType != nullptr) {
            const auto ndOutputType = outputType.cast<vpux::NDTypeInterface>();
            const auto layout = mlir::AffineMapAttr::get(ndOutputType.getDimsOrder().toAffineMap(_ctx));
            auto distributedBufferType = VPUIP::DistributedBufferType::get(
                    _ctx, ndOutputType.getShape().raw(), ndOutputType.getElementType(), layout,
                    ndOutputType.getMemSpace(), outputDataDistType.getDistribution());
            return rewriter.create<VPURT::AllocDistributed>(op->getLoc(), distributedBufferType, nullptr, nullptr)
                    .getOperation();
        }
        const auto memrefOutputType = outputType.cast<mlir::MemRefType>();
        return rewriter.create<mlir::memref::AllocOp>(op->getLoc(), memrefOutputType).getOperation();
    };

    SmallVector<mlir::Value> newOutputOperands;

    // Move the allocation ops outside NCEClusterTiling
    origOpBodyBlock.walk([&](mlir::Operation* op) {
        if (llvm::find(skipAllocOps, op) != skipAllocOps.end()) {
            return;
        }

        auto groupedAllocOp = groupedAllocOps.find(op);
        if (groupedAllocOp != groupedAllocOps.end()) {
            auto groupedViewOp = mlir::cast<vpux::GroupedViewOpInterface>(groupedAllocOp->first);
            const auto expectsGroupedBuffers = isBufAllocOp(groupedViewOp->getOperand(0).getDefiningOp());

            SmallVector<mlir::Value> newGroupedOpOperands;
            for (auto allocOp : groupedAllocOp->second) {
                auto newAllocOp = createNewAllocOp(allocOp);
                _log.nest().trace("Created alloc op '{0}'", *newAllocOp);

                newGroupedOpOperands.push_back(newAllocOp->getResult(0));
                if (!expectsGroupedBuffers) {
                    origOutputBuffers.push_back(allocOp->getResult(0));
                }
            }

            mlir::BlockAndValueMapping mapper;
            mapper.map(groupedViewOp.getViewSources(), newGroupedOpOperands);
            auto newGroupedViewOp = rewriter.clone(*groupedViewOp.getOperation(), mapper);
            vpux::inferReturnTypes(newGroupedViewOp, vpux::InferShapedTypeMode::ALL);
            _log.nest().trace("Created grouped view op '{0}'", *newGroupedViewOp);

            newOutputOperands.push_back(newGroupedViewOp->getResult(0));
            if (expectsGroupedBuffers) {
                origOutputBuffers.push_back(groupedViewOp->getResult(0));
            }
            return;
        }

        if (!isBufAllocOp(op)) {
            return;
        }
        auto newAllocOp = createNewAllocOp(op);
        _log.nest().trace("Created alloc op '{0}'", *newAllocOp);
        newOutputOperands.push_back(newAllocOp->getResult(0));
        origOutputBuffers.push_back(op->getResult(0));
    });

    inputsOutputOperands.append(newOutputOperands);

    VPUX_THROW_UNLESS(newOutputOperands.size() == 1,
                      "Currently only a single output operand in NCEClusterTiling is supported, got {0}",
                      newOutputOperands.size());

    // When creating a new VPUIP::NCEClusterTiling operation, copy the whole body of original the VPU::NCEClusterTiling
    // op except the allocation related operations which were already copied outside before and which will be passed
    // as new operands (output_buffs)
    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        // VPUIP::NCEClusterTask works with individual buffers, which requires an ungrouping operation
        // to be created now that VPUIP::NCEClusterTiling receives the sparse buffer as an operand
        mlir::Operation* ungroupOp = nullptr;
        if (origOutputBuffers.size() > 1 && newOutputOperands.size() == 1) {
            auto outputOperand = newOperands.back();
            if (outputOperand.getType().isa<VPUIP::SparseBufferType>()) {
                ungroupOp = builder.create<VPUIP::UngroupSparseBufferOp>(loc, outputOperand);
                _log.nest().trace("Created ungroup op '{0}'", *ungroupOp);
            }
        }

        auto origArguments = origOpBodyBlock.getArguments();

        mlir::BlockAndValueMapping mapper;
        mapper.map(origArguments, newOperands.take_front(origArguments.size()));
        if (ungroupOp != nullptr) {
            mapper.map(origOutputBuffers, ungroupOp->getResults());
        } else {
            mapper.map(origOutputBuffers, newOperands.take_back(origOutputBuffers.size()));
        }

        for (auto& op : origOpBodyBlock.getOperations()) {
            if (!mlir::isa<VPU::YieldOp>(op) && !isBufAllocOp(&op)) {
                if (mlir::isa<vpux::GroupedViewOpInterface>(op) && isBufAllocOp(op.getOperand(0).getDefiningOp())) {
                    _log.nest().trace("Skipping grouped view op that was moved outside '{0}'", op);
                    continue;
                }
                _log.nest().trace("Cloning op '{0}'", op);
                builder.clone(op, mapper);
            }
        }
    };

    auto clusterTilingOp = rewriter.create<VPUIP::NCEClusterTilingOp>(origOp->getLoc(), clusterTilingBufferType,
                                                                      inputsOutputOperands, bodyBuilder);

    rewriter.replaceOp(origOp, clusterTilingOp.results());

    return mlir::success();
}

//
// ConvertNCEClusterTilingToVPUIPPass
//

class ConvertNCEClusterTilingToVPUIPPass final :
        public ConvertNCEClusterTilingToVPUIPBase<ConvertNCEClusterTilingToVPUIPPass> {
public:
    explicit ConvertNCEClusterTilingToVPUIPPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertNCEClusterTilingToVPUIPPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    vpux::BufferizeTypeConverter typeConverter;

    mlir::ConversionTarget target(ctx);
    target.addIllegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addIllegalOp<VPU::NCEClusterTilingOp>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addLegalOp<mlir::memref::AllocOp>();
    vpux::populateBufferizeMaterializationLegality(target);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<NCEClusterTilingRewriter>(typeConverter, &ctx, _log);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertNCEClusterTilingToVPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertNCEClusterTilingToVPUIPPass(Logger log) {
    return std::make_unique<ConvertNCEClusterTilingToVPUIPPass>(log);
}
