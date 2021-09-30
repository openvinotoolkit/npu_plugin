//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IERT/passes.hpp"

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/core/list_scheduler.hpp"
#include "vpux/compiler/core/mem_live_range_info.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/linear_scan.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/DenseSet.h>

using namespace vpux;

namespace {

//
// AllocRewrite
//

class AllocRewrite final : public mlir::OpRewritePattern<mlir::memref::AllocOp> {
public:
    AllocRewrite(LinearScanHandler& allocInfo, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::memref::AllocOp>(ctx), _allocInfo(allocInfo), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::memref::AllocOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    LinearScanHandler& _allocInfo;
    Logger _log;
};

mlir::LogicalResult AllocRewrite::matchAndRewrite(mlir::memref::AllocOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Alloc Operation '{0}'", origOp->getLoc());

    const auto val = origOp.memref();

    for (auto* user : origOp->getUsers()) {
        if (auto iface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
            if (iface.getEffectOnValue<mlir::MemoryEffects::Free>(val)) {
                return errorAt(origOp, "IR with explicit deallocation operations is not supported");
            }
        }
    }

    const auto offset = checked_cast<int64_t>(_allocInfo.getAddress(val));

    auto startA = _allocInfo.getAddress(val);
    auto size = _allocInfo.getSize(val);
    auto endA = startA + size;

    Logger::global().error("allocated: {0} {1}-{2} ({3})", origOp->getLoc(), startA, endA, size);

    _log.trace("Replace with statically allocated VPUIP.DeclareTensorOp (offset = {0})", offset);
    rewriter.replaceOpWithNewOp<IERT::StaticAllocOp>(origOp, val.getType(), offset);

    return mlir::success();
}

//
// ListSchedulingPass
//

class ListSchedulingPass final : public IERT::ListSchedulingBase<ListSchedulingPass> {
public:
    ListSchedulingPass(IERT::AttrCreateFunc memSpaceCb, Logger log);

public:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnModule() final;

private:
    IERT::AttrCreateFunc _memSpaceCb;
    mlir::Attribute _memSpace;
};

ListSchedulingPass::ListSchedulingPass(IERT::AttrCreateFunc memSpaceCb, Logger log)
        : _memSpaceCb(std::move(memSpaceCb)) {
    Base::initLogger(log, Base::getArgumentName());
}

mlir::LogicalResult ListSchedulingPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    _memSpace = _memSpaceCb(ctx, memSpaceName.getValue());

    if (_memSpace == nullptr) {
        return mlir::failure();
    }

    return mlir::success();
}

void ListSchedulingPass::safeRunOnModule() {
    auto& ctx = getContext();
    auto module = getOperation();

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    // linear scan
    auto resources = IERT::RunTimeResourcesOp::getFromModule(module);
    auto available = resources.getAvailableMemory(_memSpace);
    const auto maxSize = available.size();
    const uint64_t alignment = 64;

    LinearScan<mlir::Value, LinearScanHandler> scan(maxSize.count(), alignment);
    auto& liveRangeInfo = getChildAnalysis<MemLiveRangeInfo>(netFunc);
    auto& depsInfo = getChildAnalysis<AsyncDepsInfo>(netFunc);

    // list scheduler
    ListScheduler scheduler(_memSpace, liveRangeInfo, depsInfo, scan);
    scheduler.generateSchedule();

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<IERT::IERTDialect>();
    target.addDynamicallyLegalOp<mlir::memref::AllocOp>([&](mlir::memref::AllocOp op) {
        const auto type = op.memref().getType().dyn_cast<mlir::MemRefType>();
        return type == nullptr || type.getMemorySpace() != _memSpace;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AllocRewrite>(scan.handler(), &ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        _log.error("Failed to replace Alloc/Dealloc Operations");
        signalPassFailure();
        return;
    }

    resources.setUsedMemory(_memSpace, scan.handler().maxAllocatedSize());
}

}  // namespace

//
// createListSchedulingPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createListSchedulingPass(AttrCreateFunc memSpaceCb, Logger log) {
    return std::make_unique<ListSchedulingPass>(std::move(memSpaceCb), log);
}