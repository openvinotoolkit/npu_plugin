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

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// AdjustLayoutsPass
//

class AdjustLayoutsPass final : public IERT::AdjustLayoutsBase<AdjustLayoutsPass> {
public:
    explicit AdjustLayoutsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class Impl;

private:
    void safeRunOnFunc() final;
};

//
// Impl
//

class AdjustLayoutsPass::Impl final : public mlir::OpInterfaceRewritePattern<LayerInterface> {
public:
    Impl(mlir::MLIRContext* ctx, Logger log, const IERT::LayerInfoDialectInterface* layerInfo)
            : mlir::OpInterfaceRewritePattern<LayerInterface>(ctx), _log(log), _layerInfo(layerInfo) {
    }

public:
    mlir::LogicalResult matchAndRewrite(LayerInterface layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    void insertReorderForInput(LayerInterface op, mlir::OpOperand& input, DimsOrder dstOrder,
                               mlir::PatternRewriter& rewriter) const;
    void insertReorderForOutput(LayerInterface op, mlir::Value output, mlir::OpOperand& output_buff, DimsOrder dstOrder,
                                mlir::PatternRewriter& rewriter) const;

    void setNewType(mlir::Value operand, DimsOrder newOrder) const;

private:
    Logger _log;
    const IERT::LayerInfoDialectInterface* _layerInfo;
};

//
// insertReorderForInput
//

void AdjustLayoutsPass::Impl::insertReorderForInput(LayerInterface op, mlir::OpOperand& input, DimsOrder dstOrder,
                                                    mlir::PatternRewriter& rewriter) const {
    _log.nest(2).trace("Insert ReorderOp for input[{0}]", input.getOperandNumber());

    const auto inputVal = input.get();
    auto origType = inputVal.getType().cast<mlir::MemRefType>();
    auto newType = mlir::MemRefType::get(origType.getShape(), origType.getElementType(),
                                         dstOrder.toAffineMap(getContext()), origType.getMemorySpace());

    _log.nest(2).trace("Create Reorder: '{0}' -> '{1}'", DimsOrder::fromValue(inputVal), dstOrder);
    auto allocOp = rewriter.create<mlir::memref::AllocOp>(op->getLoc(), newType);
    auto reorderOp = rewriter.create<IERT::ReorderOp>(op->getLoc(), inputVal, allocOp);
    input.set(reorderOp.output());
}

//
// insertReorderForOutput
//

void AdjustLayoutsPass::Impl::insertReorderForOutput(LayerInterface op, mlir::Value output,
                                                     mlir::OpOperand& output_buff, DimsOrder dstOrder,
                                                     mlir::PatternRewriter& rewriter) const {
    _log.nest(2).trace("Insert ReorderOp for output {0}", output.getType());
    auto allocOp = rewriter.create<mlir::memref::AllocOp>(op->getLoc(), output.getType().cast<mlir::MemRefType>());

    rewriter.setInsertionPointAfter(op);

    _log.nest(2).trace("Create Reorder: '{0}' -> '{1}'", DimsOrder::fromValue(output), dstOrder);
    auto reorderOp = rewriter.create<IERT::ReorderOp>(op->getLoc(), output, output_buff.get());

    output_buff.set(allocOp);
    output.replaceAllUsesExcept(reorderOp.output(), llvm::SmallPtrSet<mlir::Operation*, 1>{reorderOp});
}

//
// setNewType
//

void AdjustLayoutsPass::Impl::setNewType(mlir::Value operand, DimsOrder newOrder) const {
    const auto origType = operand.getType().cast<mlir::MemRefType>();
    const auto newType = mlir::MemRefType::get(origType.getShape(), origType.getElementType(),
                                               newOrder.toAffineMap(getContext()), origType.getMemorySpace());

    operand.setType(newType);
}

//
// matchAndRewrite
//

mlir::LogicalResult AdjustLayoutsPass::Impl::matchAndRewrite(LayerInterface layerOp,
                                                             mlir::PatternRewriter& rewriter) const {
    if (mlir::isa<IERT::ReorderOp, IERT::ConcatViewOp>(layerOp) || !layerOp->hasTrait<RTLayer>()) {
        return mlir::failure();
    }

    _log.trace("Process operation {0}", layerOp->getName());

    auto orderInfo = layerOp.getDataOrderInfo();
    _log.nest(2).trace("Current layouts: {0}", orderInfo);
    if (_layerInfo->isSupportedLayout(layerOp, orderInfo).succeeded()) {
        _log.nest(2).trace("Current layouts are supported.");
        return mlir::failure();
    }

    _log.nest(2).trace("Operation has unsupported layout. Required layouts: {0}", orderInfo);

    const auto inputs = layerOp.getInOpOperands();
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto& input = inputs[i];

        if (!orderInfo.hasInput(i)) {
            continue;
        }

        const auto inOrder = DimsOrder::fromValue(input.get());
        const auto supportedLayout = orderInfo.getInput(i);
        if (inOrder != supportedLayout) {
            insertReorderForInput(layerOp, input, supportedLayout, rewriter);
            continue;
        }
    }

    const auto output_buffs = layerOp.getOutOpOperands();
    for (size_t i = 0; i < output_buffs.size(); ++i) {
        auto& output_buff = output_buffs[i];

        if (!orderInfo.hasOutput(i)) {
            continue;
        }

        const auto currOrder = DimsOrder::fromValue(output_buff.get());
        const auto supportedLayout = orderInfo.getOutput(i);
        if (currOrder != supportedLayout) {
            const auto output = layerOp->getResult(checked_cast<unsigned>(i));

            setNewType(output, supportedLayout);
            insertReorderForOutput(layerOp, output, output_buff, currOrder, rewriter);
        }
    }

    return mlir::success();
}

//
// safeRunOnFunc
//

void AdjustLayoutsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    const auto dialect = ctx.getOrLoadDialect<IERT::IERTDialect>();
    VPUX_THROW_UNLESS(dialect != nullptr, "IERT Dialect was not loaded");
    const auto layerInfo = dialect->getRegisteredInterface<IERT::LayerInfoDialectInterface>();
    VPUX_THROW_UNLESS(layerInfo != nullptr, "LayerInfoDialect is not registered");

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<Impl>(&ctx, _log, layerInfo);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createAdjustLayoutsPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createAdjustLayoutsPass(Logger log) {
    return std::make_unique<AdjustLayoutsPass>(log);
}
