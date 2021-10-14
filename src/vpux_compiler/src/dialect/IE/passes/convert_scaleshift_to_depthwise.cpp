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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/arch.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConvertScaleShiftToDepthwisePass
//

class ConvertScaleShiftToDWPass final : public IE::ConvertScaleShiftToDWBase<ConvertScaleShiftToDWPass> {
public:
    explicit ConvertScaleShiftToDWPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class ScaleShiftOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// ScaleShiftOpConverter
//

class ConvertScaleShiftToDWPass::ScaleShiftOpConverter final : public mlir::OpRewritePattern<IE::ScaleShiftOp> {
public:
    ScaleShiftOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ScaleShiftOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScaleShiftOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertScaleShiftToDWPass::ScaleShiftOpConverter::matchAndRewrite(
        IE::ScaleShiftOp origOp, mlir::PatternRewriter& rewriter) const {
    if (origOp.weights() == nullptr)
        return matchFailed(_log, rewriter, origOp, "Failed to convert ScaleShift to DW, since there are no weights");

    if (origOp.weights().getDefiningOp<Const::DeclareOp>() == nullptr)
        return matchFailed(_log, rewriter, origOp, "Failed to convert ScaleShift to DW, since it is Eltwise Multiply");

    const SmallVector<int32_t> strides = {1, 1};
    const SmallVector<int32_t> padBegin = {0, 0};
    const SmallVector<int32_t> padEnd = {0, 0};
    const SmallVector<int32_t> dilations = {1, 1};

    const int64_t kernelSize = 1;

    auto dilationsAttr = getIntArrayAttr(origOp.getContext(), dilations);
    auto stridesAttr = getIntArrayAttr(origOp.getContext(), strides);
    auto padBeginAttr = getIntArrayAttr(origOp.getContext(), padBegin);
    auto padEndAttr = getIntArrayAttr(origOp.getContext(), padEnd);

    auto outShape = getShape(origOp.output()).toValues();
    auto groupAttr = getIntAttr(origOp.getContext(), outShape[IE::Dims4D::Act::C]);

    const auto elemType = origOp.weights().getType().cast<mlir::ShapedType>().getElementType();
    const SmallVector<int64_t> weightShape = {outShape[IE::Dims4D::Act::C], 1, kernelSize, kernelSize};
    const auto dataStorageType = mlir::RankedTensorType::get(weightShape, elemType);

    auto multiply = origOp.weights().getDefiningOp<Const::DeclareOp>();
    const auto reshapedMultiply = multiply.contentAttr().reshape(Shape(weightShape));

    auto dwConvFilter = rewriter.create<Const::DeclareOp>(origOp.getLoc(), dataStorageType, reshapedMultiply);

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(origOp, origOp.input(), dwConvFilter.output(), origOp.biases(),
                                                        stridesAttr, padBeginAttr, padEndAttr, dilationsAttr, groupAttr,
                                                        nullptr);

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertScaleShiftToDWPass::safeRunOnFunc() {
    auto func = getFunction();

    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ScaleShiftOpConverter>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertScaleShiftToDWPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertScaleShiftToDWPass(Logger log) {
    return std::make_unique<ConvertScaleShiftToDWPass>(log);
}
