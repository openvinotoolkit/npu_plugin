//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/groupconvolution_utils.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/PatternMatch.h>

#include <ngraph/coordinate.hpp>
#include <ngraph/op/max_pool.hpp>
#include <ngraph/validation_util.hpp>

using namespace vpux;

//
// FuseConvAndBias
//

namespace {

class FuseConvAndBias final : public mlir::OpRewritePattern<IE::ScaleShiftOp> {
public:
    using mlir::OpRewritePattern<IE::ScaleShiftOp>::OpRewritePattern;

    void initialize() {
        setDebugName("FuseConvAndBias");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScaleShiftOp biasOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseConvAndBias::matchAndRewrite(IE::ScaleShiftOp biasOp, mlir::PatternRewriter& rewriter) const {
    if (biasOp.weights() != nullptr) {
        return matchFailed(rewriter, biasOp, "ScaleShift has scales operand");
    }
    if (!biasOp.input().hasOneUse()) {
        return matchFailed(rewriter, biasOp, "ScaleShift is not the only user of its operand");
    }
    if (biasOp.biases() == nullptr || mlir::failed(IE::getConstParentOp(biasOp.biases()))) {
        return matchFailed(rewriter, biasOp, "ScaleShift has non constant biases");
    }

    auto* op = biasOp.input().getDefiningOp();
    if (op == nullptr || !mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp>(op)) {
        return matchFailed(rewriter, biasOp, "ScaleShift producer is not a Convolution layer");
    }

    // For those Convolutions/GroupConvolutions cannot convert to NCE task should not fuse ScaleShift as Bias.
    // Because SW kernel will not support any Post Ops.
    if (auto convOp = mlir::dyn_cast<IE::ConvolutionOp>(op)) {
        if (VPUIP::NCEInvariant::verifyKernel(convOp).failed()) {
            return matchFailed(rewriter, convOp, "Conv cannot convert to NCE, not fuse ScaleShift");
        }
    }
    if (auto grConvOp = mlir::dyn_cast<IE::GroupConvolutionOp>(op)) {
        if (VPUIP::NCEInvariant::verifyKernel(grConvOp).failed() &&
            mlir::failed(IE::canConvertGroupConvToConv(grConvOp))) {
            return matchFailed(rewriter, grConvOp, "GroupConv cannot convert to NCE, not fuse ScaleShift");
        }
    }

    if (op->getNumOperands() != 2) {
        return matchFailed(rewriter, biasOp, "ScaleShift producer already has fused biases");
    }

    const auto convOutShape = getShape(op->getOpResult(0));
    const auto biasShape = getShape(biasOp.biases());

    if (biasShape.size() != 4) {
        return matchFailed(rewriter, biasOp, "ScaleShift 'shift' operand shape doesn't match bias restrictions");
    }
    if (biasShape[Dims4D::Act::N] != 1) {
        return matchFailed(rewriter, biasOp, "ScaleShift 'shift' operand shape doesn't match bias restrictions");
    }
    if (biasShape[Dims4D::Act::C] != convOutShape[Dims4D::Act::C]) {
        return matchFailed(rewriter, biasOp, "ScaleShift 'shift' operand shape doesn't match bias restrictions");
    }
    if (biasShape[Dims4D::Act::H] != 1) {
        return matchFailed(rewriter, biasOp, "ScaleShift 'shift' operand shape doesn't match bias restrictions");
    }
    if (biasShape[Dims4D::Act::W] != 1) {
        return matchFailed(rewriter, biasOp, "ScaleShift 'shift' operand shape doesn't match bias restrictions");
    }

    auto* newConv = rewriter.clone(*op);
    newConv->insertOperands(newConv->getNumOperands(), biasOp.biases());

    rewriter.replaceOp(biasOp, newConv->getOpResults());

    return mlir::success();
}

}  // namespace

//
// Convolution
//

mlir::LogicalResult vpux::IE::ConvolutionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ConvolutionOpAdaptor conv(operands, attrs);
    if (mlir::failed(conv.verify(loc))) {
        return mlir::failure();
    }

    const auto inShape = conv.input().getType().cast<mlir::ShapedType>().getShape();
    const auto inType = conv.input().getType().cast<mlir::ShapedType>().getElementType();
    const auto filterShape = conv.filter().getType().cast<mlir::ShapedType>().getShape();

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(conv.pads_end());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(conv.pads_begin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(conv.strides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(conv.dilations());

    static const auto ChanDim = Dim(1);
    if (inShape[ChanDim.ind()] != filterShape[ChanDim.ind()]) {
        return errorAt(loc, "Channels count of input tensor shape and filter shape must be the same: {0} != {1}",
                       inShape[ChanDim.ind()], filterShape[ChanDim.ind()]);
    }

    const auto outputShape =
            ngraph::infer_convolution_forward(EmptyNode::instance(), ngraph::Shape(inShape.begin(), inShape.end()),
                                              ngraph::Strides(windowStrides.size(), 1),  // dummy data dilations
                                              ngraph::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
                                              ngraph::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
                                              ngraph::Shape(filterShape.begin(), filterShape.end()),
                                              ngraph::Strides(windowStrides.begin(), windowStrides.end()),
                                              ngraph::Strides(windowDilations.begin(), windowDilations.end()));

    const auto shapeI64 = to_small_vector(outputShape.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));
    inferredReturnShapes.emplace_back(shapeI64, inType);

    return mlir::success();
}

void vpux::IE::ConvolutionOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                          mlir::MLIRContext* context) {
    patterns.add<FuseConvAndBias>(context);
}

//
// GroupConvolution
//

mlir::LogicalResult vpux::IE::GroupConvolutionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::GroupConvolutionOpAdaptor conv(operands, attrs);
    if (mlir::failed(conv.verify(loc))) {
        return mlir::failure();
    }

    auto inShape = to_small_vector(conv.input().getType().cast<mlir::ShapedType>().getShape());
    const auto inType = conv.input().getType().cast<mlir::ShapedType>().getElementType();
    auto filterShape = to_small_vector(conv.filter().getType().cast<mlir::ShapedType>().getShape());

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(conv.pads_end());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(conv.pads_begin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(conv.strides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(conv.dilations());

    int64_t groups = 0;
    if (conv.groups().value_or(0) != 0) {
        if (filterShape.size() != inShape.size()) {
            return errorAt(loc, "Input size '{0}' does not match filter size '{1}'. (groups != 0)", inShape.size(),
                           filterShape.size());
        }

        groups = conv.groups().value();
    } else {
        if (filterShape.size() != inShape.size() + 1) {
            return errorAt(loc, "Input size '{0}' does not match filter size '{1}'. (groups == 0)", inShape.size() + 1,
                           filterShape.size());
        }

        groups = filterShape[0];

        // we need to adjust filters_shape to reuse helpers for normal convolution
        filterShape[1] *= groups;
        filterShape.erase(filterShape.begin());
    }

    inShape[1] /= groups;

    const auto outputShape =
            ngraph::infer_convolution_forward(EmptyNode::instance(), ngraph::Shape(inShape.begin(), inShape.end()),
                                              ngraph::Strides(windowStrides.size(), 1),  // dummy data dilations
                                              ngraph::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
                                              ngraph::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
                                              ngraph::Shape(filterShape.begin(), filterShape.end()),
                                              ngraph::Strides(windowStrides.begin(), windowStrides.end()),
                                              ngraph::Strides(windowDilations.begin(), windowDilations.end()));

    const auto shapeI64 = to_small_vector(outputShape.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));
    inferredReturnShapes.emplace_back(shapeI64, inType);

    return mlir::success();
}

namespace {

class GroupsToAttr final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    using mlir::OpRewritePattern<IE::GroupConvolutionOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp convOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult GroupsToAttr::matchAndRewrite(IE::GroupConvolutionOp convOp,
                                                  mlir::PatternRewriter& rewriter) const {
    if (convOp.groups().has_value()) {
        return mlir::failure();
    }

    auto filterShape = to_small_vector(convOp.filter().getType().cast<mlir::ShapedType>().getShape());

    const auto groups = filterShape[0];

    // Adjust filters_shape
    filterShape[1] *= groups;
    filterShape.erase(filterShape.begin());

    const auto filterShapeAttr = getIntArrayAttr(getContext(), filterShape);
    auto newFilter = rewriter.create<IE::ReshapeOp>(convOp->getLoc(), convOp.filter(), nullptr, false, filterShapeAttr);

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(convOp, convOp.input(), newFilter.output(), convOp.bias(),
                                                        convOp.stridesAttr(), convOp.pads_beginAttr(),
                                                        convOp.pads_endAttr(), convOp.dilationsAttr(),
                                                        getIntAttr(convOp.getContext(), groups), convOp.post_opAttr());

    return mlir::success();
}

}  // namespace

void vpux::IE::GroupConvolutionOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                               mlir::MLIRContext* context) {
    patterns.add<FuseConvAndBias>(context);
    patterns.add<GroupsToAttr>(context);
}
