//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/utils/adjust_layout_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ReshapeGroupConvInput
//

class ReshapeGroupConvInput final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    ReshapeGroupConvInput(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp convOp, mlir::PatternRewriter& rewriter) const final;

private:
    IE::ShapeCastOp buildReshape(mlir::Location loc, mlir::Value input, ArrayRef<int64_t> targetShape,
                                 mlir::PatternRewriter& rewriter) const;

    IE::ShapeCastOp reshapeOutput(IE::GroupConvolutionOp origOp, mlir::Value convOutput,
                                  mlir::PatternRewriter& rewriter) const;
    mlir::Value broadcastConst(mlir::Value activation, int64_t factor, Dim onDim,
                               mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

mlir::Value ReshapeGroupConvInput::broadcastConst(mlir::Value activation, int64_t dimVal, Dim onDim,
                                                  mlir::PatternRewriter& rewriter) const {
    const auto origInShape = getShape(activation);
    const auto KX = 1;
    const auto KY = 1;
    const auto IC = (onDim == Dims4D::Filter::IC) ? dimVal : 1;
    const auto OC = (onDim == Dims4D::Filter::OC) ? dimVal : 1;

    const Shape weightShape = {OC, IC, KY, KX};

    auto cst = activation.getDefiningOp<Const::DeclareOp>();

    auto content = cst.getContent();
    Const::ContentAttr newContentAttr;
    if (content.isSplat()) {
        auto contentAttr = cst.getContentAttr();
        auto baseContent = contentAttr.getBaseContent();
        Shape realDataShape;
        if (auto denseBaseAttr = baseContent.dyn_cast<mlir::DenseElementsAttr>()) {
            newContentAttr = Const::ContentAttr::get(denseBaseAttr);
            realDataShape = denseBaseAttr.getType().getShape();
        } else if (auto opaqueBaseAttr = baseContent.dyn_cast<Const::OpaqueElementsAttr>()) {
            newContentAttr = Const::ContentAttr::get(opaqueBaseAttr);
            realDataShape = opaqueBaseAttr.getType().getShape();
        } else {
            VPUX_THROW("Got unsupported 'baseContent' in 'ContentAttr'");
        }
        auto newConstantShape = Shape(origInShape.size(), int64_t(1));
        newContentAttr = newContentAttr.reshape(newConstantShape);
        for (auto attr : contentAttr.getTransformations()) {
            if (attr.isa<Const::PadWithZeroAttr>() || attr.isa<Const::BroadcastAttr>() ||
                attr.isa<Const::ReshapeAttr>()) {
                // The const's shape will fully handled by this pass, the broadcast will be added,
                //   so ignore the origin broadcast, reshape and pad transformation
                continue;
            }
            newContentAttr = Const::ContentAttr::addTransformation(newContentAttr, attr);
        }
        newContentAttr = newContentAttr.broadcast(onDim, weightShape[onDim]);
    } else {
        auto broadcastDim = (OC > 1 ? Dims4D::Filter::OC : Dims4D::Filter::IC);
        newContentAttr = cst.getContentAttr();
        newContentAttr = newContentAttr.broadcast(broadcastDim, weightShape[broadcastDim]);
        newContentAttr = newContentAttr.reshape(weightShape);
    }
    return rewriter.create<Const::DeclareOp>(activation.getLoc(), newContentAttr.getType(), newContentAttr);
}

IE::ShapeCastOp ReshapeGroupConvInput::reshapeOutput(IE::GroupConvolutionOp origOp, mlir::Value convOutput,
                                                     mlir::PatternRewriter& rewriter) const {
    const auto origOutShape = getShape(origOp.output());
    const SmallVector<int64_t> targetShape = to_small_vector(origOutShape.raw());

    const auto reshapedLoc = appendLoc(origOp.getLoc(), "_output_reshape");
    return buildReshape(reshapedLoc, convOutput, makeArrayRef(targetShape), rewriter);
}

IE::ShapeCastOp ReshapeGroupConvInput::buildReshape(mlir::Location loc, mlir::Value input,
                                                    ArrayRef<int64_t> targetShape,
                                                    mlir::PatternRewriter& rewriter) const {
    const auto ctx = rewriter.getContext();
    const auto srcType = input.getType().cast<vpux::NDTypeInterface>();
    const auto dstType = srcType.changeShape(ShapeRef(targetShape));
    const auto targetShapeAttr = getIntArrayAttr(ctx, targetShape);
    auto inputShapeCastOp = rewriter.create<IE::ShapeCastOp>(loc, dstType, input, targetShapeAttr);
    return inputShapeCastOp;
}

mlir::LogicalResult ReshapeGroupConvInput::matchAndRewrite(IE::GroupConvolutionOp convOp,
                                                           mlir::PatternRewriter& rewriter) const {
    const auto alignedChannel = VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
    auto groups = convOp.groups().value();
    auto inNDInterface = convOp.input().getType().dyn_cast<vpux::NDTypeInterface>();
    auto inDimOrder = inNDInterface.getDimsOrder();
    SmallVector<int64_t> newInputShape;
    if (DimsOrder::NHWC != inDimOrder) {
        return mlir::failure();
    }

    auto isConstAndSameAsGroup = [groups](mlir::Value value) {
        auto cst = value.getDefiningOp<Const::DeclareOp>();
        if (!cst) {
            return false;
        }
        // Only handle below scenario:
        //  kernel size 1x1 and Group == Output Channel
        auto shapeSize = value.getType().cast<vpux::NDTypeInterface>().getNumElements();
        if (shapeSize != groups) {
            return false;
        }
        return true;
    };

    auto isSplat = [](mlir::Value val) {
        auto cst = val.getDefiningOp<Const::DeclareOp>();
        auto content = cst.getContent();
        return content.isSplat();
    };

    auto isElementWised = [isSplat](IE::GroupConvolutionOp op) {
        return isSplat(op.filter()) && (!op.bias() || isSplat(op.bias()));
    };

    const auto supportedGroupConv = [&](IE::GroupConvolutionOp layerOp) {
        auto outNDInterface = layerOp.output().getType().cast<vpux::NDTypeInterface>();
        auto outDimOrder = outNDInterface.getDimsOrder();
        if (inDimOrder != outDimOrder) {
            return false;
        }

        if (outNDInterface.getElementType() != inNDInterface.getElementType()) {
            return false;
        }

        auto strides = parseIntArrayAttr<int64_t>(layerOp.strides());
        auto withOutStrides = std::all_of(strides.begin(), strides.end(), [](auto val) {
            return val == 1;
        });
        if (!withOutStrides) {
            return false;
        }

        // Already aligned
        if (!(groups % alignedChannel)) {
            return false;
        }
        // check kernel size and is a depthwise convolution or not
        if (!isConstAndSameAsGroup(layerOp.filter())) {
            return false;
        }

        // check filter and bias const is same
        if (layerOp.bias() != nullptr) {
            if (!isConstAndSameAsGroup(layerOp.bias())) {
                return false;
            }
        }

        auto input = layerOp.input().getDefiningOp<IE::PermuteQuantizeOp>();
        if (input) {
            // Expand can be fused to PermuteQuantize, ignore
            if (!layerOp.output().hasOneUse()) {
                return false;
            }
            if (VPUIP::NCEInvariant::verifyKernel(*layerOp.output().getUsers().begin()).succeeded()) {
                const auto outputShape = getShape(layerOp.output());
                if (outputShape[Dims4D::Act::C] % alignedChannel) {
                    return false;
                }
            }
        }

        if (isElementWised(layerOp)) {
            auto newExpandedShapeResult =
                    getShapeCastExpandedShapeInDimC(layerOp, getShape(layerOp.input()), _log.nest());
            if (mlir::failed(newExpandedShapeResult)) {
                newExpandedShapeResult =
                        getShapeCastExpandedShapeCanNotAlign(layerOp, getShape(layerOp.input()), _log.nest());
                if (mlir::failed(newExpandedShapeResult)) {
                    return false;
                }
            }
            newInputShape = newExpandedShapeResult.value().raw();
            return true;
        }
        // Input can be reshaped
        auto newExpandedShapeResult =
                getShapeCastExpandedShapeKeepDimC(layerOp, getShape(layerOp.input()), _log.nest());
        if (mlir::failed(newExpandedShapeResult)) {
            return false;
        }
        newInputShape = newExpandedShapeResult.value().raw();
        return true;
    };
    if (!supportedGroupConv(convOp)) {
        return mlir::failure();
    }
    _log.trace("Adjust input/filter/bias shape for group convolution at '{0}'", convOp->getLoc());
    auto input = buildReshape(convOp.getLoc(), convOp.input(), newInputShape, rewriter);
    auto weights = broadcastConst(convOp.filter(), newInputShape[Dims4D::Act::C.ind()], Dims4D::Filter::OC, rewriter);
    auto bias = convOp.bias();
    if (bias) {
        bias = broadcastConst(convOp.bias(), newInputShape[Dims4D::Act::C.ind()], Dims4D::Filter::IC, rewriter);
    }
    auto newGroupConv = rewriter.create<IE::GroupConvolutionOp>(
            convOp.getLoc(), input, weights, bias, convOp.strides(), convOp.pads_begin(), convOp.pads_end(),
            convOp.dilations(), getIntAttr(rewriter.getContext(), newInputShape[Dims4D::Act::C.ind()]), nullptr);
    auto newInShape = getShape(input);
    auto origOutputType = convOp.getType().cast<vpux::NDTypeInterface>();
    newGroupConv.output().setType(origOutputType.changeShape(newInShape));
    auto output = reshapeOutput(convOp, newGroupConv, rewriter);
    rewriter.replaceOp(convOp, output.result());
    return mlir::success();
}

//
// AdjustGroupConvShape
//

class AdjustGroupConvShapePass final : public IE::AdjustGroupConvShapeBase<AdjustGroupConvShapePass> {
public:
    explicit AdjustGroupConvShapePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AdjustGroupConvShapePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ReshapeGroupConvInput>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createAdjustGroupConvShapePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustGroupConvShapePass(Logger log) {
    return std::make_unique<AdjustGroupConvShapePass>(log);
}
