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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <ngraph/coordinate_diff.hpp>
#include <ngraph/op/op.hpp>

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConvertScalarToTensorPass
//

class ConvertScalarToTensorPass final : public IERT::ConvertScalarToTensorBase<ConvertScalarToTensorPass> {
public:
    explicit ConvertScalarToTensorPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class ScalarConverter;

private:
    void safeRunOnFunc() final;
};

//
// ScalarConverter
//

class ConvertScalarToTensorPass::ScalarConverter final : public mlir::OpRewritePattern<IERT::GatherOp> {
public:
    ScalarConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IERT::GatherOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::GatherOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertScalarToTensorPass::ScalarConverter::matchAndRewrite(IERT::GatherOp origOp,
                                                                                mlir::PatternRewriter& rewriter) const {
    _log.debug("Need to convert indices to tensor");

    const std::array<int64_t, 1> indicesShape = {1};
    const auto indicesType = origOp.indices().getType().cast<vpux::NDTypeInterface>();
    const auto newIndicesType = indicesType.changeShape(ShapeRef(indicesShape));

    auto newIndices = rewriter.create<IERT::GenericReshapeOp>(origOp->getLoc(), newIndicesType, origOp.indices());

    rewriter.replaceOpWithNewOp<IERT::GatherOp>(origOp, origOp.getType(), origOp.input(), newIndices.output(), nullptr,
                                                origOp.output_buff(), origOp.axis_valueAttr(), origOp.batch_dims());

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertScalarToTensorPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);

    const auto isLegalOp = [](IERT::GatherOp op) {
        bool isTensor = (op.indices().getType().cast<mlir::MemRefType>().getRank() != 0);
        return isTensor;
    };

    target.addDynamicallyLegalOp<IERT::GatherOp>(isLegalOp);
    target.addLegalOp<IERT::GenericReshapeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ScalarConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        _log.debug("Failed to replace indices from scalar to tensor");
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertScalarToTensorPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createConvertScalarToTensorPass(Logger log) {
    return std::make_unique<ConvertScalarToTensorPass>(log);
}
