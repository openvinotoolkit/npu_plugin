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
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <ngraph_ops/convolution_ie.hpp>

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConvertAvgPoolToDWConvPass
//

class ConvertAvgPoolToDWConvPass final : public IE::ConvertAvgPoolToDWConvBase<ConvertAvgPoolToDWConvPass> {
public:
    explicit ConvertAvgPoolToDWConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class AvgPoolOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// AvgPoolOpConverter
//

class ConvertAvgPoolToDWConvPass::AvgPoolOpConverter final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    AvgPoolOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

static mlir::DenseElementsAttr buildWeightData(const mlir::RankedTensorType dataStorageType, const int64_t kernelY,
                                               const int64_t kernelX) {
    const float weightRealVal = 1.0f / static_cast<float>(kernelY * kernelX);
    const auto elemType = dataStorageType.getElementType();
    if (elemType.isF32()) {
        return mlir::DenseElementsAttr::get(dataStorageType, weightRealVal);
    } else if (elemType.isF16()) {
        const ngraph::float16 weightHalfVal = weightRealVal;
        return mlir::DenseElementsAttr::get(dataStorageType, weightHalfVal);
    }
    return nullptr;
}

mlir::LogicalResult ConvertAvgPoolToDWConvPass::AvgPoolOpConverter::matchAndRewrite(
        IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto outputShape = getShape(origOp.output());
    const auto OC = outputShape[IE::Dims4D::Act::C];
    const auto kernel = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    const auto kernelY = kernel[0];
    const auto kernelX = kernel[1];

    const auto elemType = origOp.input().getType().cast<mlir::ShapedType>().getElementType();
    const SmallVector<int64_t> weightShape = {OC, 1, 1, kernelY, kernelX};
    const auto dataStorageType = mlir::RankedTensorType::get(weightShape, elemType);
    const auto dataAttr = buildWeightData(dataStorageType, kernelY, kernelX);
    // Must never fail, given the 'RankedTensorOf<[F16, F32]>:$input,' declaration.
    VPUX_THROW_UNLESS(dataAttr != nullptr,
                      "Average pool has incompatible data type {0}, only float16 or float32 are supported", elemType);

    auto dwConvFilter =
            rewriter.create<Const::DeclareOp>(origOp.getLoc(), dataStorageType, Const::ContentAttr::get(dataAttr));
    const SmallVector<int32_t> dilations = {1, 1};
    auto dilationsAttr = getIntArrayAttr(origOp.getContext(), dilations);

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
            origOp, origOp.input(), dwConvFilter.output(), /*bias=*/nullptr, origOp.stridesAttr(),
            origOp.pads_beginAttr(), origOp.pads_endAttr(), dilationsAttr,
            /*groups=*/nullptr, /*post_opAttr=*/nullptr, /*clip_opAttr=*/nullptr);

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertAvgPoolToDWConvPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isLegal = [&](IE::AvgPoolOp origOp) {
        const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.kernel_size());
        const auto KY = kernelSize[0];
        const auto KX = kernelSize[1];

        const auto kernelStrides = parseIntArrayAttr<int64_t>(origOp.strides());
        const auto SY = kernelStrides[0];
        const auto SX = kernelStrides[1];

        const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
        const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
        const auto padTop = padsBegin[0];
        const auto padBottom = padsEnd[0];
        const auto padLeft = padsBegin[1];
        const auto padRight = padsEnd[1];

        // The logic is reversed here. If AvgPoolOp can be represented as an NCE task, it becomes illegal.
        return mlir::failed(VPUIP::NCEInvariant::verifyKernel(origOp->getLoc(), KY, KX, SY, SX, padTop, padBottom,
                                                              padLeft, padRight));
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::AvgPoolOp>(isLegal);
    target.addLegalOp<IE::GroupConvolutionOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<AvgPoolOpConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertAvgPoolToDWConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertAvgPoolToDWConvPass(Logger log) {
    return std::make_unique<ConvertAvgPoolToDWConvPass>(log);
}
