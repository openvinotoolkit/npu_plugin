//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

using namespace vpux;

namespace {

//
// TimestampRewrite
//

class TimestampRewrite final : public mlir::OpRewritePattern<VPUIP::TimestampOp> {
public:
    TimestampRewrite(mlir::MLIRContext* ctx, vpux::VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<VPUIP::TimestampOp>(ctx), _log(log), _arch(arch) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::TimestampOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    vpux::VPU::ArchKind _arch;
};

mlir::LogicalResult TimestampRewrite::matchAndRewrite(VPUIP::TimestampOp origOp,
                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Timestamp Operation '{0}'", origOp->getLoc());

    auto origType = origOp.getType().cast<vpux::NDTypeInterface>();
    VPUX_THROW_UNLESS(origType.getNumElements() == 1, "Got wrong elements number for TimestampOp");

    const auto timerType = origType.changeMemSpace(VPU::MemoryKind::Register);

    uint32_t hwAddress = 0;
    int64_t dmaPort = 0;
    auto execOp = origOp->getParentOfType<mlir::async::ExecuteOp>();
    if (execOp != nullptr && VPUIP::VPUIPDialect::hasExecutorInstanceMask(execOp)) {
        auto portMask = parseIntArrayAttr<int64_t>(VPUIP::VPUIPDialect::getExecutorInstanceMask(execOp));
        // Only use port number if one is configured, else assume it will be later unrolled
        // into per port operations
        // TODO: Maybe dmaPort from NNDMAOp should also hold array of ports
        if (portMask.size() == 1) {
            dmaPort = portMask[0];
        }
    }

    switch (_arch) {
    case VPU::ArchKind::VPUX30XX:
    case VPU::ArchKind::VPUX311X:
        hwAddress = VPUIP::HW_TIMER_ABSOLUTE_ADDR_30XX;
        VPUX_THROW_UNLESS(origType.getElementType() == getUInt32Type(getContext()),
                          "Got wrong element type for TimestampOp");
        break;
    case VPU::ArchKind::VPUX37XX:
        hwAddress = VPUIP::HW_TIMER_ABSOLUTE_ADDR_37XX;
        VPUX_THROW_UNLESS(origType.getElementType() == getUInt64Type(getContext()),
                          "Got wrong element type for TimestampOp");
        break;
    default:
        VPUX_THROW("Unsuported architecture for TimestampRewrite");
    }

    auto bufferOp = rewriter.create<VPURT::DeclareBufferOp>(origOp->getLoc(), timerType, VPURT::BufferSection::Register,
                                                            hwAddress);

    rewriter.replaceOpWithNewOp<VPUIP::NNDMAOp>(origOp, bufferOp.buffer(), origOp.output_buff(), dmaPort);

    _log.trace("Replaced with 'VPURT::DeclareBufferOp'");

    return mlir::success();
}

//
// CopyOpRewrite
//

class CopyOpRewrite final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    CopyOpRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::CopyOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CopyOpRewrite::matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found CopyOp Operation '{0}'", origOp->getLoc());

    int64_t dmaPort = 0;
    auto execOp = origOp->getParentOfType<mlir::async::ExecuteOp>();
    if (execOp != nullptr && VPUIP::VPUIPDialect::hasExecutorInstanceMask(execOp)) {
        auto portMask = parseIntArrayAttr<int64_t>(VPUIP::VPUIPDialect::getExecutorInstanceMask(execOp));
        // Only use port number if one is configured, else assume it will be later unrolled
        // into per port operations
        // TODO: Maybe dmaPort from NNDMAOp should also hold array of ports
        if (portMask.size() == 1) {
            dmaPort = portMask[0];
        }
    }

    rewriter.replaceOpWithNewOp<VPUIP::NNDMAOp>(origOp, origOp.input(), origOp.output_buff(), dmaPort,
                                                origOp.channelTypeAttr(), false, false, origOp.spillIdAttr(), nullptr);

    return mlir::success();
}

//
// ConvertTransferOpsToDMAsPass
//

class ConvertTransferOpsToDMAsPass final : public VPUIP::ConvertTransferOpsToDMAsBase<ConvertTransferOpsToDMAsPass> {
public:
    explicit ConvertTransferOpsToDMAsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertTransferOpsToDMAsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<VPUIP::CopyOp>();
    target.addIllegalOp<VPUIP::TimestampOp>();
    target.addLegalOp<VPUIP::NNDMAOp>();
    target.addLegalOp<VPURT::DeclareBufferOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<TimestampRewrite>(&ctx, arch, _log);
    patterns.insert<CopyOpRewrite>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertTransferOpsToDMAsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertTransferOpsToDMAsPass(Logger log) {
    return std::make_unique<ConvertTransferOpsToDMAsPass>(log);
}
