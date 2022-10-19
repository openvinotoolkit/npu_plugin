//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BlockAndValueMapping.h>

using namespace vpux;

namespace {

//
// NCEClusterTilingRewriter
//

class NCEClusterTilingRewriter final : public mlir::OpConversionPattern<VPU::NCEClusterTilingOp> {
public:
    NCEClusterTilingRewriter(mlir::TypeConverter& converter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::NCEClusterTilingOp>(converter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEClusterTilingOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult NCEClusterTilingRewriter::matchAndRewrite(VPU::NCEClusterTilingOp origOp, OpAdaptor newArgs,
                                                              mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    VPUX_THROW_UNLESS(origOp.results().size() == 1,
                      "Currently only single output NCEClusterTiling operation is supported, got {0} outputs",
                      origOp.results().size());

    // Initialize VPUIP::NCEClusterTiling operands with operands of
    // VPU::NCEClusterTiling op. Output buffs will be added based on identified
    // allocation operations
    SmallVector<mlir::Value> inputsOutputOperands = newArgs.operands();
    SmallVector<mlir::Value> origOpBuffers;

    const auto clusterTilingBufferType = typeConverter->convertType(origOp.results()[0].getType());

    auto& origOpBodyBlock = origOp.body().front();

    // Traverse body of VPU::NCEClusterTiling op and for each buffer
    // allocation operation copy it outside and store information about
    // new buffers
    origOpBodyBlock.walk([&](mlir::Operation* op) {
        if (!isBufAllocOp(op)) {
            return;
        }
        rewriter.setInsertionPoint(origOp);

        mlir::Value newAllocBuff;
        if (clusterTilingBufferType.isa<VPUIP::DistributedBufferType>()) {
            newAllocBuff =
                    rewriter.create<VPURT::AllocDistributed>(op->getLoc(), clusterTilingBufferType, nullptr, nullptr)
                            .getResult();
        } else {
            newAllocBuff = rewriter.create<mlir::memref::AllocOp>(op->getLoc(),
                                                                  clusterTilingBufferType.cast<mlir::MemRefType>())
                                   .getResult();
        }
        inputsOutputOperands.push_back(newAllocBuff);
        origOpBuffers.push_back(op->getResult(0));
    });

    VPUX_THROW_UNLESS(origOpBuffers.size() == 1,
                      "Currently only single allocation in NCEClusterTiling is supported, got {0} allocations",
                      origOpBuffers.size());

    // When creating new VPUIP::NCEClusterTiling operation copy whole body of original VPU::NCEClusterTiling
    // op except allocation related operations which were already copied outside before and which will be
    // passed as new operands (output_buffs)
    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        std::ignore = loc;
        mlir::BlockAndValueMapping mapper;

        auto origArguments = origOpBodyBlock.getArguments();
        mapper.map(origArguments, newOperands.take_front(origArguments.size()));
        mapper.map(origOpBuffers, newOperands.take_back(origOpBuffers.size()));

        for (auto& op : origOpBodyBlock.getOperations()) {
            if (!mlir::isa<VPU::YieldOp>(op) && !isBufAllocOp(&op)) {
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
