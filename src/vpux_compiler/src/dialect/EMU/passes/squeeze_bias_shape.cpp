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
#include "vpux/compiler/dialect/EMU/passes.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/EMU/ops.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// SqueezeBiasShapePass
//

class SqueezeBiasShapePass final : public EMU::SqueezeBiasShapeBase<SqueezeBiasShapePass> {
public:
    explicit SqueezeBiasShapePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// BiasShapeConverter
//
template <class ConcreteOp>
class BiasShapeConverter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    BiasShapeConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult BiasShapeConverter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                    mlir::PatternRewriter& rewriter) const {

    constexpr size_t BIAS_OPERAND = 2;
    const auto bias = origOp.bias();
    if (bias == nullptr) {
        return mlir::failure();
    }
    auto biasConst = bias.template getDefiningOp<Const::DeclareOp>();
    if (biasConst == nullptr) {
        return mlir::failure();
    }
    VPUX_THROW_UNLESS(bias == origOp.getOperand(BIAS_OPERAND), "Bias operand is not at the expected index");

    rewriter.startRootUpdate(origOp);

    const auto origBiasType = bias.getType().template cast<mlir::ShapedType>();
    const auto origBiasShape = origBiasType.getShape();

    const auto newBiasShape = ShapeRef({origBiasShape[1]});
    const auto newBiasType = changeShape(origBiasType, newBiasShape);
    const auto newBiasConstAttr = biasConst.contentAttr().reshape(newBiasShape);
    auto newBias = rewriter.replaceOpWithNewOp<Const::DeclareOp>(biasConst, newBiasType, newBiasConstAttr).output();

    // Updating bias operand
    origOp.setOperand(BIAS_OPERAND, newBias);

    rewriter.finalizeRootUpdate(origOp);

    return mlir::success();
}

template <class ConcreteOp>
bool isLegalOp(ConcreteOp op)
{
    return op.bias().getType().template cast<mlir::ShapedType>().getShape().size() == 1;
}

//
// safeRunOnFunc
//

void SqueezeBiasShapePass::safeRunOnFunc() {

    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::ConvolutionOp>([](IE::ConvolutionOp op)
        {return isLegalOp<IE::ConvolutionOp>(op);});
    target.addDynamicallyLegalOp<IE::FullyConnectedOp>([](IE::FullyConnectedOp op)
        {return isLegalOp<IE::FullyConnectedOp>(op);});
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<BiasShapeConverter<IE::ConvolutionOp>>(&ctx, _log);
    patterns.insert<BiasShapeConverter<IE::FullyConnectedOp>>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }

}

}  // namespace

//
// createSqueezeBiasShapePass
//

std::unique_ptr<mlir::Pass> vpux::EMU::createSqueezeBiasShapePass(Logger log) {
    return std::make_unique<SqueezeBiasShapePass>(log);
}
