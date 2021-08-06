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

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <ngraph_ops/convolution_ie.hpp>

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <ngraph/slice_plan.hpp>

using namespace vpux;

namespace {

//
// ResolveStridedSlicePass
//

class ResolveStridedSlicePass final : public IE::ResolveStridedSliceBase<ResolveStridedSlicePass> {
public:
    explicit ResolveStridedSlicePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class SlicePlanning;

private:
    void safeRunOnFunc() final;
};

//
// SlicePlanning
//

class ResolveStridedSlicePass::SlicePlanning final : public mlir::OpRewritePattern<IE::StridedSliceOp> {
public:
    SlicePlanning(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::StridedSliceOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::StridedSliceOp origOp, mlir::PatternRewriter& rewriter) const final;
    static ngraph::SlicePlan getSlicePlan(IE::StridedSliceOp origOp);

private:
    Logger _log;
};

ngraph::SlicePlan ResolveStridedSlicePass::SlicePlanning::getSlicePlan(IE::StridedSliceOp origOp) {
    const auto getAxisSetArr = [](mlir::ArrayAttr attr) {
        ngraph::AxisSet axis_set;

        const auto arr = parseIntArrayAttr<int64_t>(attr);
        for (const auto& p : arr | indexed) {
            if (p.value() == 1) {
                axis_set.emplace(p.index());
            }
        }

        return axis_set;
    };

    VPUX_THROW_UNLESS(origOp.begins_attr().hasValue(), "begins_attr is null");
    VPUX_THROW_UNLESS(origOp.ends_attr().hasValue(), "ends_attr is null");
    VPUX_THROW_UNLESS(origOp.strides_attr().hasValue(), "strides_attr is null");

    const auto beginsVec = to_std_vector(parseIntArrayAttr<int64_t>(origOp.begins_attr().getValue()));
    const auto endsVec = to_std_vector(parseIntArrayAttr<int64_t>(origOp.ends_attr().getValue()));
    const auto stridesVec = to_std_vector(parseIntArrayAttr<int64_t>(origOp.strides_attr().getValue()));

    const auto beginMask = getAxisSetArr(origOp.begin_mask());
    const auto endMask = getAxisSetArr(origOp.end_mask());
    const auto newAxisMask = getAxisSetArr(origOp.new_axis_mask());
    const auto shrinkAxisMask = getAxisSetArr(origOp.shrink_axis_mask());
    const auto ellipsisMask = getAxisSetArr(origOp.ellipsis_mask());

    const auto inDataType = origOp.input().getType().cast<mlir::ShapedType>();
    const auto inDataShape = inDataType.getShape();

    return ngraph::make_slice_plan(ngraph::Shape(inDataShape.begin(), inDataShape.end()), beginsVec, endsVec,
                                   stridesVec, beginMask, endMask, newAxisMask, shrinkAxisMask, ellipsisMask);
}

mlir::LogicalResult ResolveStridedSlicePass::SlicePlanning::matchAndRewrite(IE::StridedSliceOp origOp,
                                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::StridedSlice Operation '{0}'", origOp->getLoc());

    const auto plan = getSlicePlan(origOp);

    const auto beginAttr = getIntArrayAttr(getContext(), plan.begins);
    const auto endsAttr = getIntArrayAttr(getContext(), plan.ends);
    const auto stridesAttr = getIntArrayAttr(getContext(), plan.strides);
    const auto zeroesArrayAttr = getIntArrayAttr(getContext(), llvm::SmallVector<int64_t>(plan.begins.size(), 0));

    auto newOp = rewriter.create<IE::StridedSliceOp>(
            origOp->getLoc(), origOp.input(), origOp.begins(), origOp.ends(), origOp.strides(), beginAttr, endsAttr,
            stridesAttr, zeroesArrayAttr, zeroesArrayAttr, zeroesArrayAttr, zeroesArrayAttr, zeroesArrayAttr);

    const auto outputShapeAttr = getIntArrayAttr(getContext(), plan.reshape_out_shape);
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newOp.output(), nullptr, false, outputShapeAttr);

    _log.trace("Replaced with 'IE::StridedSlice' -> 'IE::Reshape'");

    return mlir::success();
}

//
// safeRunOnFunc
//

void ResolveStridedSlicePass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isLegalOp = [&](IE::StridedSliceOp slice) {
        auto isZero = [](auto val) {
            return val == 0;
        };
        auto isPositive = [](auto val) {
            return val >= 0;
        };

        VPUX_THROW_UNLESS(slice.begins_attr().hasValue(), "begins_attr is null");
        VPUX_THROW_UNLESS(slice.ends_attr().hasValue(), "ends_attr is null");

        return llvm::all_of(parseIntArrayAttr<int64_t>(slice.new_axis_mask()), isZero) &&
               llvm::all_of(parseIntArrayAttr<int64_t>(slice.shrink_axis_mask()), isZero) &&
               llvm::all_of(parseIntArrayAttr<int64_t>(slice.ellipsis_mask()), isZero) &&
               llvm::all_of(parseIntArrayAttr<int64_t>(slice.begin_mask()), isZero) &&
               llvm::all_of(parseIntArrayAttr<int64_t>(slice.end_mask()), isZero) &&
               llvm::all_of(parseIntArrayAttr<int64_t>(slice.begins_attr().getValue()), isPositive) &&
               llvm::all_of(parseIntArrayAttr<int64_t>(slice.ends_attr().getValue()), isPositive);
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::StridedSliceOp>(isLegalOp);
    target.addLegalOp<IE::ReshapeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<SlicePlanning>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createResolveStridedSlicePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createResolveStridedSlicePass(Logger log) {
    return std::make_unique<ResolveStridedSlicePass>(log);
}
