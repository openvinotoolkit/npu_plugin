//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/EMU/passes.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
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
    BiasShapeConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult BiasShapeConverter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    const auto bias = origOp.bias();
    if (bias == nullptr) {
        return mlir::failure();
    }
    auto biasConst = bias.template getDefiningOp<Const::DeclareOp>();
    if (biasConst == nullptr) {
        return mlir::failure();
    }

    rewriter.startRootUpdate(origOp);

    const auto origBiasType = bias.getType().template cast<vpux::NDTypeInterface>();
    const auto origBiasShape = origBiasType.getShape();

    const auto newBiasShape = Shape({origBiasShape[Dims4D::Act::C]});
    const auto newBiasType = origBiasType.changeShape(newBiasShape);
    const auto newBiasConstAttr = biasConst.contentAttr().reshape(newBiasShape);
    auto newBias = rewriter.replaceOpWithNewOp<Const::DeclareOp>(biasConst, newBiasType, newBiasConstAttr).output();

    // Updating bias operand
    origOp.biasMutable().assign(newBias);

    rewriter.finalizeRootUpdate(origOp);

    return mlir::success();
}

template <class ConcreteOp>
bool isLegalOp(ConcreteOp op) {
    return op.bias().getType().template cast<vpux::NDTypeInterface>().getShape().size() == 1;
}

//
// safeRunOnFunc
//

void SqueezeBiasShapePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<VPU::ConvolutionOp>(&isLegalOp<VPU::ConvolutionOp>);
    target.addDynamicallyLegalOp<VPU::GroupConvolutionOp>(&isLegalOp<VPU::GroupConvolutionOp>);
    target.addDynamicallyLegalOp<VPU::FullyConnectedOp>(&isLegalOp<VPU::FullyConnectedOp>);
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<BiasShapeConverter<VPU::ConvolutionOp>>(&ctx, _log);
    patterns.add<BiasShapeConverter<VPU::GroupConvolutionOp>>(&ctx, _log);
    patterns.add<BiasShapeConverter<VPU::FullyConnectedOp>>(&ctx, _log);

    auto func = getOperation();
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
