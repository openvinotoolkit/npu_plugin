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

#include "vpux/compiler/dialect/IERT/passes.hpp"

#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/core/static_allocation.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/linear_scan.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// StaticAllocationPass
//

class StaticAllocationPass final : public IERT::StaticAllocationBase<StaticAllocationPass> {
public:
    StaticAllocationPass(mlir::Attribute memSpace, Logger log);

public:
    void runOnOperation() final;

public:
    class AllocRewrite;
    class DeallocRewrite;

private:
    void passBody();

private:
    mlir::Attribute _memSpace;
    Logger _log;
};

StaticAllocationPass::StaticAllocationPass(mlir::Attribute memSpace, Logger log): _memSpace(memSpace), _log(log) {
    _log.setName(Base::getArgumentName());
}

void StaticAllocationPass::runOnOperation() {
    try {
        auto& ctx = getContext();

        if (_memSpace == nullptr) {
            if (!memSpaceName.getValue().empty()) {
                _memSpace = mlir::StringAttr::get(&ctx, memSpaceName.getValue());
            }
        }

        passBody();
    } catch (const std::exception& e) {
        (void)errorAt(getOperation(), "{0} failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

//
// AllocRewrite
//

class StaticAllocationPass::AllocRewrite final : public mlir::OpRewritePattern<mlir::memref::AllocOp> {
public:
    AllocRewrite(StaticAllocation& allocInfo, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::memref::AllocOp>(ctx), _allocInfo(allocInfo), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::memref::AllocOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    std::reference_wrapper<StaticAllocation> _allocInfo;
    Logger _log;
};

mlir::LogicalResult StaticAllocationPass::AllocRewrite::matchAndRewrite(mlir::memref::AllocOp origOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Alloc Operation '{0}'", origOp->getLoc());

    const auto val = origOp.memref();

    const auto offset = _allocInfo.get().getValOffset(val);
    if (!offset.hasValue()) {
        _log.error("Value '{0}' was not allocated", val.getLoc());
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IERT::StaticAllocOp>(origOp, val.getType(), offset.getValue());

    _log.trace("Replaced with statically allocated VPUIP.DeclareTensorOp (offset = {0})", offset);

    return mlir::success();
}

//
// DeallocRewrite
//

class StaticAllocationPass::DeallocRewrite final : public mlir::OpConversionPattern<mlir::memref::DeallocOp> {
public:
    DeallocRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<mlir::memref::DeallocOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::memref::DeallocOp origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult StaticAllocationPass::DeallocRewrite::matchAndRewrite(
        mlir::memref::DeallocOp origOp, ArrayRef<mlir::Value> newOperands,
        mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Dealloc Operation '{0}'", origOp->getLoc());

    VPUX_THROW_UNLESS(newOperands.size() == 1, "Got wrong newOperands count {0}", newOperands.size());

    auto* producer = newOperands[0].getDefiningOp();
    if (producer == nullptr) {
        _log.error("Value '{0}' has no producer", newOperands[0].getLoc());
        return mlir::failure();
    }
    if (!mlir::isa<IERT::StaticAllocOp>(producer)) {
        _log.error("Value '{0}' was produced by unsupported Operation '{1}'", newOperands[0].getLoc(),
                   producer->getLoc());
        return mlir::failure();
    }

    rewriter.eraseOp(origOp);

    _log.trace("Erase Dealloc Operation");

    return mlir::success();
}

//
// passBody
//

void StaticAllocationPass::passBody() {
    auto& ctx = getContext();
    auto module = getOperation();

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    auto resources = IERT::RunTimeResourcesOp::getFromModule(module);
    if (resources == nullptr) {
        _log.error("The pass '{0}' requires run-time resources information", getName());
        signalPassFailure();
        return;
    }

    auto available = resources.getAvailableMemory(_memSpace);
    if (available == nullptr) {
        _log.error("The memory space '{0}' is not available", _memSpace);
        signalPassFailure();
        return;
    }

    const auto maxSize = available.size();
    const uint64_t alignment = 64;  // TODO: extract from run-time resources information

    StaticAllocation allocInfo(netFunc, _memSpace, maxSize, alignment);

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<IERT::IERTDialect>();
    target.addDynamicallyLegalOp<mlir::memref::AllocOp>([&](mlir::memref::AllocOp op) {
        const auto type = op.memref().getType().dyn_cast<mlir::MemRefType>();
        return type == nullptr || type.getMemorySpace() != _memSpace;
    });
    target.addDynamicallyLegalOp<mlir::memref::DeallocOp>([&](mlir::memref::DeallocOp op) {
        const auto type = op.memref().getType().dyn_cast<mlir::MemRefType>();
        return type == nullptr || type.getMemorySpace() != _memSpace;
    });

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.insert<AllocRewrite>(allocInfo, &ctx, _log.nest());
    patterns.insert<DeallocRewrite>(&ctx, _log.nest());

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        _log.error("Failed to replace Alloc/Dealloc Operations");
        signalPassFailure();
        return;
    }

    resources.setUsedMemory(_memSpace, allocInfo.maxAllocatedSize());
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IERT::createStaticAllocationPass(mlir::Attribute memSpace, Logger log) {
    return std::make_unique<StaticAllocationPass>(memSpace, log);
}
