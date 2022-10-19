//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/IE/loop.hpp"

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

mlir::Value createSubAvgPool(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter, ArrayRef<int64_t> begins,
                             ArrayRef<int64_t> ends, mlir::ArrayAttr avgPoolOpKernelAttr,
                             mlir::ArrayAttr avgPoolOpStridesAttr) {
    mlir::MLIRContext* ctx = origOp->getContext();
    const auto beginMask = getIntArrayAttr(ctx, makeArrayRef({1, 1, 0, 0}));
    const auto endMask = getIntArrayAttr(ctx, makeArrayRef({1, 1, 0, 0}));
    const auto newAxisMask = getIntArrayAttr(ctx, makeArrayRef({0, 0, 0, 0}));
    const auto shrinkAxisMask = getIntArrayAttr(ctx, makeArrayRef({0, 0, 0, 0}));
    const auto ellipsisMask = getIntArrayAttr(ctx, makeArrayRef({0, 0, 0, 0}));
    const auto stridesAttr = getIntArrayAttr(ctx, makeArrayRef({1, 1, 1, 1}));
    const auto beginsAttr = getIntArrayAttr(ctx, begins);
    const auto endsAttr = getIntArrayAttr(ctx, ends);

    auto stridedSliceOp = rewriter.create<IE::StridedSliceOp>(origOp.getLoc(), origOp.input(), nullptr, nullptr,
                                                              nullptr, beginsAttr, endsAttr, stridesAttr, beginMask,
                                                              endMask, newAxisMask, shrinkAxisMask, ellipsisMask);

    const auto zeroPadAttr = getIntArrayAttr(ctx, makeArrayRef({0, 0}));

    auto avgPoolOp = rewriter.create<IE::AvgPoolOp>(origOp->getLoc(), stridedSliceOp.output(), avgPoolOpKernelAttr,
                                                    avgPoolOpStridesAttr, zeroPadAttr, zeroPadAttr,
                                                    origOp.rounding_typeAttr(), nullptr, origOp.post_opAttr());
    return avgPoolOp.output();
}

//
// AveragePoolRewriter
//

class AveragePoolRewriter final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    AveragePoolRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _log(log) {
        setDebugName("AveragePoolRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AveragePoolRewriter::matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got AveragePool layer at '{1}'", getDebugName(), origOp->getLoc());
    auto nestLog = _log.nest();
    auto* ctx = origOp->getContext();

    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padValue = padsBegin[Dims4D::PadsBegin::Top.ind()];  // same on all sides
    const auto shape = getShape(origOp.output());
    const auto width = shape[Dims4D::Act::W];
    const auto height = shape[Dims4D::Act::H];
    const auto origStrides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    const auto kernelHeight = kernelSize[Dims4D::Kernel::Y.ind()];
    const auto kernelWidth = kernelSize[Dims4D::Kernel::X.ind()];
    SmallVector<mlir::Value> inputs;

    nestLog.trace("Create AvgPool center op without any padding");

    const auto zeroPadAttr = getIntArrayAttr(ctx, makeArrayRef({0, 0}));

    auto centerOp = rewriter.create<IE::AvgPoolOp>(origOp->getLoc(), origOp.input(), origOp.kernel_size(),
                                                   origOp.strides(), zeroPadAttr, zeroPadAttr,
                                                   origOp.rounding_typeAttr(), nullptr, origOp.post_opAttr());

    inputs.push_back(centerOp.output());

    nestLog.trace("Create SubAvgPool operation for top left corner");

    const auto cornerKernelH = kernelHeight - padValue;
    const auto cornerKernelW = kernelWidth - padValue;
    const auto cornerKernelAttr = getIntArrayAttr(ctx, makeArrayRef({cornerKernelH, cornerKernelW}));
    const auto cornerStridesAttr = getIntArrayAttr(ctx, makeArrayRef({1, 1}));

    inputs.push_back(createSubAvgPool(origOp, rewriter, /*begins=*/{0, 0, 0, 0},
                                      /*ends=*/{0, 0, cornerKernelH, cornerKernelW}, cornerKernelAttr,
                                      cornerStridesAttr));

    nestLog.trace("Create SubAvgPool operation for top right corner");

    inputs.push_back(createSubAvgPool(origOp, rewriter, /*begins=*/{0, 0, 0, width - cornerKernelW},
                                      /*ends=*/{0, 0, cornerKernelH, width}, cornerKernelAttr, cornerStridesAttr));

    nestLog.trace("Create SubAvgPool operation for bottom right corner");

    inputs.push_back(createSubAvgPool(origOp, rewriter,
                                      /*begins=*/{0, 0, height - cornerKernelH, width - cornerKernelW},
                                      /*ends=*/{0, 0, height, width}, cornerKernelAttr, cornerStridesAttr));

    nestLog.trace("Create SubAvgPool operation for bottom left corner");

    inputs.push_back(createSubAvgPool(origOp, rewriter, /*begins=*/{0, 0, height - cornerKernelH, 0},
                                      /*ends=*/{0, 0, height, cornerKernelW}, cornerKernelAttr, cornerStridesAttr));

    nestLog.trace("Create SubAvgPool operation for left side");

    const auto verticalKernelH = kernelHeight;
    const auto verticalKernelW = kernelWidth - padValue;
    SmallVector<int64_t> verticalStrides({1, origStrides[Dims4D::Strides::Y.ind()]});
    auto verticalStridesAttr = getIntArrayAttr(ctx, makeArrayRef({verticalStrides[0], verticalStrides[1]}));
    auto verticalKernelAttr = getIntArrayAttr(ctx, makeArrayRef({verticalKernelH, verticalKernelW}));

    inputs.push_back(createSubAvgPool(origOp, rewriter, /*begins=*/{0, 0, 0, 0},
                                      /*ends=*/{0, 0, height, verticalKernelW}, verticalKernelAttr,
                                      verticalStridesAttr));

    nestLog.trace("Create SubAvgPool operation for right side");

    inputs.push_back(createSubAvgPool(origOp, rewriter, /*begins=*/{0, 0, 0, width - verticalKernelW},
                                      /*ends=*/{0, 0, height, width}, verticalKernelAttr, verticalStridesAttr));

    nestLog.trace("Create SubAvgPool operation for top side");

    const auto horizontalKernelH = kernelHeight - padValue;
    const auto horizontalKernelW = kernelWidth;
    SmallVector<int64_t> horizontalStrides({origStrides[Dims4D::Strides::X.ind()], 1});
    auto horizontalStridesAttr = getIntArrayAttr(ctx, makeArrayRef({horizontalStrides[0], horizontalStrides[1]}));
    auto horizontalKernelAttr = getIntArrayAttr(ctx, makeArrayRef({horizontalKernelH, horizontalKernelW}));

    inputs.push_back(createSubAvgPool(origOp, rewriter, /*begins=*/{0, 0, 0, 0},
                                      /*ends=*/{0, 0, horizontalKernelH, width}, horizontalKernelAttr,
                                      horizontalStridesAttr));

    nestLog.trace("Create SubAvgPool operation for bottom side");

    inputs.push_back(createSubAvgPool(origOp, rewriter, /*begins=*/{0, 0, height - horizontalKernelH, 0},
                                      /*ends=*/{0, 0, height, width}, horizontalKernelAttr, horizontalStridesAttr));
    // offsets for concatenate subAvgPool
    SmallVector<SmallVector<int64_t>> staticOffsets = {{0, 0, 1, 1},                       // center op
                                                       {0, 0, 0, 0},                       // top left op
                                                       {0, 0, 0, (width - 1)},             // top right op
                                                       {0, 0, (height - 1), (width - 1)},  // bottom right op
                                                       {0, 0, (height - 1), 0},            // bottom left op
                                                       {0, 0, 1, 0},                       // left op
                                                       {0, 0, 1, (width - 1)},             // right op
                                                       {0, 0, 0, 1},                       // top op
                                                       {0, 0, (height - 1), 1}};           // bottom op
    mlir::ArrayAttr staticOffsetsAttr = getIntArrayOfArray(ctx, staticOffsets);

    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, origOp.getType(), inputs, staticOffsetsAttr);

    return mlir::success();
}

//
// HandleExcludePadForAvgPoolPass
//

class HandleExcludePadForAvgPoolPass final : public IE::HandleExcludePadForAvgPoolBase<HandleExcludePadForAvgPoolPass> {
public:
    explicit HandleExcludePadForAvgPoolPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void HandleExcludePadForAvgPoolPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::AvgPoolOp>([&](IE::AvgPoolOp op) {
        if (!op.exclude_pads()) {
            return true;
        }
        const auto padsBegin = parseIntArrayAttr<int64_t>(op.pads_begin());
        const auto padsEnd = parseIntArrayAttr<int64_t>(op.pads_end());
        const auto zeros = SmallVector<int64_t>{0, 0};
        if (padsBegin == zeros && padsEnd == zeros) {
            return true;
        }
        const auto ones = SmallVector<int64_t>{1, 1};
        const auto strides = parseIntArrayAttr<int64_t>(op.strides());
        return !(padsBegin == ones && padsEnd == ones && strides == ones);
    });

    target.addLegalOp<IE::StridedSliceOp>();
    target.addLegalOp<IE::ConcatOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<AveragePoolRewriter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// HandleExcludePadForAvgPoolPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createHandleExcludePadForAvgPoolPass(Logger log) {
    return std::make_unique<HandleExcludePadForAvgPoolPass>(log);
}
