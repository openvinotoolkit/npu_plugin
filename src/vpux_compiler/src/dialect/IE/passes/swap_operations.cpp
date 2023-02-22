//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/utils/transpose_op_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <functional>

using namespace vpux;

namespace {

const int64_t SUPPORTED_RANK = 4;
const int8_t CHANNEL_ALIGNMENT = 16;

mlir::Value alignConstant(mlir::PatternRewriter& rewriter, mlir::Operation* parent, mlir::Value constInput) {
    return llvm::TypeSwitch<mlir::Operation*, mlir::Value>(parent)
            .Case<IE::AffineReshapeOp, IE::ReshapeOp>([&](auto origOp) {
                const auto constInputShape = getShape(constInput);
                const auto parentInputDimC = getShape(origOp.input())[Dims4D::Act::C];
                if (constInputShape.totalSize() != parentInputDimC) {
                    return mlir::Value();
                }

                SmallVector<int64_t> constShape(constInputShape.size(), 1);
                constShape[Dims4D::Act::C.ind()] = parentInputDimC;

                return rewriter.createOrFold<IE::ReshapeOp>(
                        origOp->getLoc(), constInput, nullptr, false,
                        getIntArrayAttr(origOp->getContext(), makeArrayRef(constShape)));
            })
            .Case<IE::TransposeOp>([&](auto origOp) {
                const auto dstOrder = IE::deduceInverseOrder(origOp);
                const auto dstPerm = dstOrder.toAffineMap(origOp->getContext());
                const auto dstOrderAttr = mlir::AffineMapAttr::get(dstPerm);

                return rewriter.createOrFold<IE::TransposeOp>(origOp->getLoc(), constInput, nullptr, dstOrderAttr);
            })
            .Default([](mlir::Operation* op) -> mlir::Value {
                VPUX_THROW("Unsupported operation '{0}' at '{1}'", op->getName(), op->getLoc());
            });
}

//
// SwapWithBias
//

class SwapWithBias final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    SwapWithBias(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AddOp>(ctx), _log(log) {
        setDebugName("SwapWithBias");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapWithBias::matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Found Add operation {1}", getDebugName(), origOp);

    const auto in1Type = origOp.input1().getType().cast<vpux::NDTypeInterface>();
    const auto in2Type = origOp.input2().getType().cast<vpux::NDTypeInterface>();
    const auto outShapeRes = origOp.output().getType().cast<vpux::NDTypeInterface>();

    if (in1Type == in2Type) {
        _log.trace("[{0}] Don't swap operations with Eltwise {1}", getDebugName(), origOp);
        return mlir::failure();
    }

    bool lhsIsActivation = (in1Type == outShapeRes);
    auto activationInput = lhsIsActivation ? origOp.input1() : origOp.input2();
    auto biasInput = lhsIsActivation ? origOp.input2() : origOp.input1();

    auto isInputConst = [&](mlir::Value input) -> bool {
        if (auto constOp = input.getDefiningOp<Const::DeclareOp>()) {
            return true;
        }
        if (auto fqOp = input.getDefiningOp<IE::FakeQuantizeOp>()) {
            if (auto constOp = fqOp.input().getDefiningOp<Const::DeclareOp>()) {
                return true;
            }
        }
        return false;
    };
    if (!isInputConst(biasInput)) {
        _log.trace("[{0}] Add operation {1} is not bias", getDebugName(), origOp);
        return mlir::failure();
    }

    auto parentOp = activationInput.getDefiningOp();

    if (parentOp == nullptr) {
        return mlir::failure();
    }

    if (!mlir::isa<IE::ElemTypeInfoOpInterface>(parentOp)) {
        _log.trace("[{0}] Swapped operation {1} doesn't implement ElemTypeInfoOpInterface interface", getDebugName(),
                   *parentOp);
        return mlir::failure();
    }

    if (mlir::isa<IE::ReorderOp>(parentOp)) {
        _log.trace("[{0}] Don't swap ReorderOp {1}", getDebugName(), *parentOp);
        return mlir::failure();
    }

    if (!parentOp->hasOneUse()) {
        _log.trace("[{0}] Swapped operation {1} has more than one use", getDebugName(), *parentOp);
        return mlir::failure();
    }

    auto parentInput = parentOp->getOperand(0);
    if (parentInput.getType().cast<vpux::NDTypeInterface>().getRank() != SUPPORTED_RANK) {
        _log.trace("[{0}] Swapped operation doesn't have rank {1}", getDebugName(), SUPPORTED_RANK);
        return mlir::failure();
    }

    // TODO: E#68168 check the layout info as we did for Sigmod/Relu/Tanh
    auto newConstant = alignConstant(rewriter, parentOp, biasInput);
    if (!newConstant) {
        _log.trace("[{0}] Swapped operation {1} fails to align constant", getDebugName(), *parentOp);
        return mlir::failure();
    }

    auto newAdd =
            rewriter.create<IE::AddOp>(origOp.getLoc(), parentInput, newConstant, origOp.auto_broadcast(), nullptr);

    mlir::BlockAndValueMapping mapper;
    mapper.map(parentOp->getOperands(), newAdd->getResults());
    auto* newParent = rewriter.clone(*parentOp, mapper);
    vpux::inferReturnTypes(newParent, vpux::InferShapedTypeMode::ALL);

    rewriter.replaceOp(origOp, newParent->getResults());

    return mlir::success();
}

//
// SwapWithActivation
//

template <class Activation>
class SwapWithActivation final : public mlir::OpRewritePattern<Activation> {
public:
    SwapWithActivation(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<Activation>(ctx), _log(log) {
        this->setDebugName("SwapWithActivation");
    }

public:
    mlir::LogicalResult matchAndRewrite(Activation origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class Activation>
mlir::LogicalResult SwapWithActivation<Activation>::matchAndRewrite(Activation origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Found activation function {1}", this->getDebugName(), origOp);

    auto parentOp = origOp.input().getDefiningOp();

    if (parentOp == nullptr) {
        return mlir::failure();
    }

    // TODO: E#18651 Support ElemTypeInfoOpInterface for Slice in the correct way
    if (!mlir::isa<IE::ElemTypeInfoOpInterface>(parentOp) || mlir::isa<IE::SliceOp>(parentOp)) {
        _log.trace("[{0}] Swapped operation {1} doesn't implement ElemTypeInfoOpInterface interface {0}",
                   this->getDebugName(), parentOp);
        return mlir::failure();
    }

    if (!parentOp->hasOneUse()) {
        _log.trace("[{0}] Swapped operation {1} has more than one use", this->getDebugName(), parentOp);
        return mlir::failure();
    }

    mlir::Value parentInput = parentOp->getOperand(0);
    if (parentInput.getType().cast<vpux::NDTypeInterface>().getRank() != SUPPORTED_RANK) {
        _log.trace("[{0}] Swapped operation doesn't have rank {1}", this->getDebugName(), SUPPORTED_RANK);
        return mlir::failure();
    }

    mlir::Value origOperand = origOp->getResult(0);
    const auto origOrder = origOperand.getType().cast<vpux::NDTypeInterface>().getDimsOrder();
    const auto parentOrder = parentInput.getType().cast<vpux::NDTypeInterface>().getDimsOrder();
    bool updateOutOrder = false;

    if (origOrder != parentOrder) {
        auto origLayoutOp = mlir::dyn_cast<IE::LayoutInfoOpInterface>(origOperand.getDefiningOp());
        if (origLayoutOp == nullptr) {
            return mlir::failure();
        }

        auto orderInfo = origLayoutOp.getLayoutInfo();
        orderInfo.setInput(0, parentOrder);
        origLayoutOp.inferLayoutInfo(orderInfo);
        if (orderInfo.getInput(0) != parentOrder) {
            return mlir::failure();
        }
        if (orderInfo.getOutput(0) != parentOrder) {
            return mlir::failure();
        }
        updateOutOrder = true;
    }

    mlir::BlockAndValueMapping mapper;
    mapper.map(origOp->getOperands(), parentOp->getOperands());
    auto* newActivation = rewriter.clone(*origOp, mapper);
    vpux::inferReturnTypes(newActivation, vpux::InferShapedTypeMode::ALL);

    // TODO: E#68166 check IE operation's inferReturnTypes method in terms of layout,
    // currently almost all set to default value.
    if (updateOutOrder) {
        mlir::Value newActivationOperand = newActivation->getResult(0);
        const auto newActOrigType = newActivationOperand.getType().cast<vpux::NDTypeInterface>();
        if (newActOrigType.getDimsOrder() != parentOrder) {
            const auto newType = newActOrigType.changeDimsOrder(parentOrder);
            newActivationOperand.setType(newType);
        }
    }

    mapper.clear();
    mapper.map(parentOp->getOperands(), newActivation->getResults());
    auto* newParent = rewriter.clone(*parentOp, mapper);
    vpux::inferReturnTypes(newParent, vpux::InferShapedTypeMode::ALL);

    rewriter.replaceOp(origOp, newParent->getResults());

    return mlir::success();
}

class SwapTanhSlice final : public mlir::OpRewritePattern<IE::TanhOp> {
public:
    SwapTanhSlice(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TanhOp>(ctx), _log(log) {
        this->setDebugName("SwapOperationsPass::SwapTanhSlice");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::TanhOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapTanhSlice::matchAndRewrite(IE::TanhOp originOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", originOp->getName(), originOp->getLoc());
    auto sliceOp = originOp.input().getDefiningOp<IE::SliceOp>();
    if (sliceOp == nullptr) {
        return mlir::failure();
    }

    auto oldSliceType = sliceOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto oldLayerType = originOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto newType = oldLayerType.changeShape(sliceOp.source().getType().cast<vpux::NDTypeInterface>().getShape());

    const auto oldSliceShape = oldSliceType.getShape();
    const auto newLayerShape = newType.getShape();

    // Move tanH only when the slice is due to channel alignment X % 16 != 0
    if (oldSliceShape[Dims4D::Act::C] % CHANNEL_ALIGNMENT == 0) {
        return mlir::failure();
    }

    // In case when actual number of channels is less than 1/2 of the aligned channel value
    // Such cases avoid moving TanH as it would be computationally expensive operation and does not offer any gain
    // e.g. Actual channels: 3 Aligned Channels 16, we don't want to compute TanH with 16 Channels for such case
    if (oldSliceShape[Dims4D::Act::C] < newLayerShape[Dims4D::Act::C] / 2) {
        return mlir::failure();
    }

    auto newOp = rewriter.create<IE::TanhOp>(originOp.getLoc(), newType, sliceOp.source());
    auto newSlice = rewriter.replaceOpWithNewOp<IE::SliceOp>(originOp, newOp->getResult(0),
                                                             sliceOp.static_offsetsAttr(), sliceOp.static_sizesAttr());

    newSlice->getResult(0).setType(oldSliceType);

    return mlir::success();
}

class SwapOperationsPass final : public IE::SwapOperationsBase<SwapOperationsPass> {
public:
    explicit SwapOperationsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SwapOperationsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwapWithActivation<IE::ReLUOp>>(&ctx, _log.nest());
    patterns.add<SwapWithActivation<IE::SigmoidOp>>(&ctx, _log.nest());
    patterns.add<SwapWithActivation<IE::TanhOp>>(&ctx, _log.nest());
    patterns.add<SwapWithBias>(&ctx, _log.nest());
    // TODO: E#18651 Support ElemTypeInfoOpInterface for Slice
    patterns.add<SwapTanhSlice>(&ctx, _log.nest());

    auto func = getFunction();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSwapOperationsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSwapOperationsPass(Logger log) {
    return std::make_unique<SwapOperationsPass>(log);
}
