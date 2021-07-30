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

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/IR/PatternMatch.h>

#include <ngraph/coordinate.hpp>
#include <ngraph/op/max_pool.hpp>
#include <ngraph/util.hpp>
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
        return matchFailed(rewriter, biasOp, "Failed to fuse ScaleShift, since it has scales");
    }
    if (!biasOp.input().hasOneUse()) {
        return matchFailed(rewriter, biasOp,
                           "Failed to fuse ScaleShift, since it is not the only user of its input Value");
    }

    auto* convOp = biasOp.input().getDefiningOp();
    if (convOp == nullptr || !mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp>(convOp)) {
        return matchFailed(rewriter, biasOp, "Failed to fuse ScaleShift, its producer is not a Convolution layer");
    }

    if (convOp->getNumOperands() != 2) {
        return matchFailed(rewriter, biasOp, "Failed to fuse ScaleShift, its producer already has fused biases");
    }

    const auto convOutShape = getShape(convOp->getOpResult(0));
    const auto biasShape = getShape(biasOp.biases());

    if (biasShape.size() != 4) {
        return matchFailed(rewriter, biasOp, "Failed to fuse ScaleShift, unsupported bias shape");
    }
    if (biasShape[IE::Dims4D::Act::N] != 1) {
        return matchFailed(rewriter, biasOp, "Failed to fuse ScaleShift, unsupported bias shape");
    }
    if (biasShape[IE::Dims4D::Act::C] != convOutShape[IE::Dims4D::Act::C]) {
        return matchFailed(rewriter, biasOp, "Failed to fuse ScaleShift, unsupported bias shape");
    }
    if (biasShape[IE::Dims4D::Act::H] != 1) {
        return matchFailed(rewriter, biasOp, "Failed to fuse ScaleShift, unsupported bias shape");
    }
    if (biasShape[IE::Dims4D::Act::W] != 1) {
        return matchFailed(rewriter, biasOp, "Failed to fuse ScaleShift, unsupported bias shape");
    }

    auto* dialect = biasOp->getDialect();
    VPUX_THROW_UNLESS(dialect != nullptr, "IE Dialect was not loaded");

    const auto layerInfo = dialect->getRegisteredInterface<IE::LayerInfoDialectInterface>();
    VPUX_THROW_UNLESS(layerInfo != nullptr, "LayerInfoDialect is not registered");

    if (!layerInfo->isSupportedPostProcessing(convOp, biasOp)) {
        return matchFailed(rewriter, biasOp, "Failed to fuse Bias into Convolution, it's not supported");
    }

    auto* newConv = rewriter.clone(*convOp);
    newConv->insertOperands(newConv->getNumOperands(), biasOp.biases());

    rewriter.replaceOp(biasOp, newConv->getOpResults());

    return mlir::success();
}

}  // namespace

//
// Convolution
//

mlir::LogicalResult vpux::IE::ConvolutionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

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
            ngraph::infer_convolution_forward(nullptr, ngraph::Shape(inShape.begin(), inShape.end()),
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
    patterns.insert<FuseConvAndBias>(context);
}

//
// GroupConvolution
//

mlir::LogicalResult vpux::IE::GroupConvolutionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

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
    if (conv.groups() != 0) {
        if (filterShape.size() != inShape.size()) {
            return errorAt(loc, "Wrong filter shape '{0}'", filterShape);
        }

        groups = conv.groups().getInt();
    } else {
        if (filterShape.size() != inShape.size() + 1) {
            return errorAt(loc, "Wrong filter shape '{0}'", filterShape);
        }

        groups = filterShape[0];

        // we need to adjust filters_shape to reuse helpers for normal convolution
        filterShape[1] *= groups;
        filterShape.erase(filterShape.begin());
    }

    inShape[1] /= groups;

    const auto outputShape =
            ngraph::infer_convolution_forward(nullptr, ngraph::Shape(inShape.begin(), inShape.end()),
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
    if (convOp.groups().hasValue()) {
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
    patterns.insert<FuseConvAndBias>(context);
    patterns.insert<GroupsToAttr>(context);
}
