//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConvertGatherToSlicePass
//

class ConvertGatherToSlicePass final : public IE::ConvertGatherToSliceBase<ConvertGatherToSlicePass> {
public:
    explicit ConvertGatherToSlicePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class GatherConverter;

private:
    void safeRunOnFunc() final;
};

//
// GatherConverter
//

class ConvertGatherToSlicePass::GatherConverter final : public mlir::OpRewritePattern<IE::GatherOp> {
public:
    GatherConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::GatherOp>(ctx), _log(log) {
        setDebugName("ConvertGatherToSlicePass::GatherConverter");
    }

    mlir::LogicalResult matchAndRewrite(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertGatherToSlicePass::GatherConverter::matchAndRewrite(IE::GatherOp gatherOp,
                                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Gather Op: {0}", gatherOp);
    auto* ctx = rewriter.getContext();

    auto indices = gatherOp.indices().getDefiningOp<Const::DeclareOp>();
    const auto indicesContent = indices.content();
    int64_t indicesVal = indicesContent.getSplatValue<int64_t>();

    const auto axisVal = gatherOp.axis_value().getValue();

    const auto inType = gatherOp.input().getType().cast<NDTypeInterface>();
    const auto inputShape = inType.getShape();
    auto static_offsets = SmallVector<int64_t>(inputShape.size(), 0);
    static_offsets[axisVal] = indicesVal;

    SmallVector<int64_t> static_sizes(inputShape.begin(), inputShape.end());
    static_sizes[axisVal] = 1;

    const auto sliceOpLoc = appendLoc(gatherOp.getLoc(), "_slice");
    auto sliceOp = rewriter.create<IE::SliceOp>(sliceOpLoc, gatherOp.input(), getIntArrayAttr(ctx, static_offsets),
                                                getIntArrayAttr(ctx, static_sizes));

    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(gatherOp, sliceOp.result(), nullptr, false,
                                               getIntArrayAttr(ctx, getShape(gatherOp.output())));

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertGatherToSlicePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto module = getOperation();
    const auto arch = VPU::getArch(module);
    if (arch == VPU::ArchKind::VPUX37XX) {
        _log.trace("Slice is not enabled for VPUX37XX device. ConvertGatherToSlicePass is disabled. Got: {0}", arch);
        return;
    }

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::GatherOp>([&](IE::GatherOp gatherOp) {
        const auto batchDims = gatherOp.batch_dims();

        auto indices = gatherOp.indices().getDefiningOp<Const::DeclareOp>();
        if (indices == nullptr) {
            return true;
        }

        const auto indicesContent = indices.content();
        return !(indicesContent.getType().getNumElements() == 1 && batchDims == 0 && gatherOp.axis_value().hasValue());
    });
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ReshapeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<GatherConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertGatherToSlicePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertGatherToSlicePass(Logger log) {
    return std::make_unique<ConvertGatherToSlicePass>(log);
}
