//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/core/static_allocation.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// AssignTensorOffsetsDDRPass
//

class AssignTensorOffsetsDDRPass final : public VPUIP::AssignTensorOffsetsDDRBase<AssignTensorOffsetsDDRPass> {
public:
    explicit AssignTensorOffsetsDDRPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    void runOnOperation() final;

public:
    class AllocRewrite;
    class DeallocRewrite;

private:
    void passBody();

private:
    Logger _log;
};

void AssignTensorOffsetsDDRPass::runOnOperation() {
    try {
        passBody();
    } catch (const std::exception& e) {
        printTo(getOperation().emitError(), "{0} failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

//
// AllocRewrite
//

class AssignTensorOffsetsDDRPass::AllocRewrite final : public mlir::OpRewritePattern<mlir::AllocOp> {
public:
    AllocRewrite(StaticAllocation& allocInfo, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::AllocOp>(ctx), _allocInfo(allocInfo), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::AllocOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    std::reference_wrapper<StaticAllocation> _allocInfo;
    Logger _log;
};

mlir::LogicalResult AssignTensorOffsetsDDRPass::AllocRewrite::matchAndRewrite(mlir::AllocOp origOp,
                                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Alloc Operation '{0}'", origOp);

    auto val = origOp.memref();

    auto offset = _allocInfo.get().getValOffset(val);
    if (!offset.hasValue()) {
        _log.error("Value '{0}' was not allocated", val);
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPUIP::DeclareTensorOp>(origOp, val.getType(), VPUIP::MemoryLocation::VPU_DDR_Heap,
                                                        offset.getValue());

    _log.trace("Replaced with statically allocated VPUIP.DeclareTensorOp (offset = {0})", offset);

    return mlir::success();
}

//
// DeallocRewrite
//

class AssignTensorOffsetsDDRPass::DeallocRewrite final : public mlir::OpConversionPattern<mlir::DeallocOp> {
public:
    DeallocRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpConversionPattern<mlir::DeallocOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::DeallocOp origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AssignTensorOffsetsDDRPass::DeallocRewrite::matchAndRewrite(
        mlir::DeallocOp origOp, ArrayRef<mlir::Value> newOperands, mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Dealloc Operation '{0}'", origOp);

    VPUX_THROW_UNLESS(newOperands.size() == 1, "Got wrong newOperands count {0}", newOperands.size());

    auto* producer = newOperands[0].getDefiningOp();
    if (!mlir::isa<VPUIP::DeclareTensorOp>(producer)) {
        _log.error("Value '{0}' was produced by unsupported Operation '{1}'", newOperands[0], producer->getName());
        return mlir::failure();
    }

    rewriter.eraseOp(origOp);

    _log.trace("Erase Dealloc Operation");

    return mlir::success();
}

//
// passBody
//

void AssignTensorOffsetsDDRPass::passBody() {
    auto& ctx = getContext();
    auto module = getOperation();

    auto& allocInfo = getAnalysis<StaticAllocation>();

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addIllegalOp<mlir::AllocOp, mlir::DeallocOp>();

    mlir::OwningRewritePatternList patterns;
    patterns.insert<AllocRewrite>(allocInfo, &ctx, _log.nest());
    patterns.insert<DeallocRewrite>(&ctx, _log.nest());

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        _log.error("Failed to replace Alloc/Dealloc Operations");
        signalPassFailure();
    }

    VPUIP::GraphOp graphOp;
    mlir::FuncOp graphFunc;
    if (mlir::failed(VPUIP::GraphOp::getFromModule(module, graphOp, graphFunc))) {
        _log.error("Failed to get VPUIP.Graph Operation from module");
        signalPassFailure();
        return;
    }

    auto oldResources = graphOp.resourcesAttr();

    auto oldMemSizes = oldResources.memory_sizes();
    SmallVector<mlir::Attribute, 2> newMemSizes;

    for (auto memMap : oldMemSizes.getAsRange<VPUIP::MemoryMappingAttr>()) {
        if (memMap.item().getValue() != VPUIP::PhysicalMemory::DDR) {
            newMemSizes.push_back(memMap);
        }
    }

    newMemSizes.push_back(VPUIP::MemoryMappingAttr::get(
            VPUIP::PhysicalMemoryAttr::get(VPUIP::PhysicalMemory::DDR, module.getContext()),
            getInt64Attr(module.getContext(), allocInfo.maxAllocatedSize()), module.getContext()));

    auto newResources =
            VPUIP::ResourcesAttr::get(oldResources.processor_allocation(), oldResources.processor_frequencies(),
                                      mlir::ArrayAttr::get(newMemSizes, module.getContext()),
                                      oldResources.memory_bandwidth(), module.getContext());

    graphOp.resourcesAttr(newResources);
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createAssignTensorOffsetsDDRPass(Logger log) {
    return std::make_unique<AssignTensorOffsetsDDRPass>(log);
}
