//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
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

public:
    mlir::LogicalResult matchAndRewrite(IE::ScaleShiftOp biasOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseConvAndBias::matchAndRewrite(IE::ScaleShiftOp biasOp, mlir::PatternRewriter& rewriter) const {
    static const auto C = Dim(1);

    if (!biasOp.input().hasOneUse()) {
        return mlir::failure();
    }

    if (biasOp.weights()) {
        return mlir::failure();
    }

    auto convOp = mlir::dyn_cast_or_null<ConvolutionLayerInterface>(biasOp.input().getDefiningOp());
    if (convOp == nullptr) {
        return mlir::failure();
    }

    if (convOp->getNumOperands() != 2 || convOp.bias() != nullptr) {
        return mlir::failure();
    }

    auto convOutShape = getShape(convOp.output());
    auto biasShape = getShape(biasOp.biases());

    if (convOutShape.size() != 4) {
        return mlir::failure();
    }
    if (biasShape[C] != convOutShape[C]) {
        return mlir::failure();
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

    const auto dataPaddingBelow = parseIntArrayAttr(conv.pads_end());
    const auto dataPaddingAbove = parseIntArrayAttr(conv.pads_begin());
    const auto windowStrides = parseIntArrayAttr(conv.strides());
    const auto windowDilations = parseIntArrayAttr(conv.dilations());

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

    const auto dataPaddingBelow = parseIntArrayAttr(conv.pads_end());
    const auto dataPaddingAbove = parseIntArrayAttr(conv.pads_begin());
    const auto windowStrides = parseIntArrayAttr(conv.strides());
    const auto windowDilations = parseIntArrayAttr(conv.dilations());

    int64_t groups = 0;
    if (conv.groups() != 0) {
        if (filterShape.size() != 4) {
            return errorAt(loc, "Wrong filter shape '{0}'", filterShape);
        }

        groups = conv.groups().getInt();
    } else {
        if (filterShape.size() != 5) {
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

    const auto filterShapeType = mlir::RankedTensorType::get({checked_cast<int64_t>(filterShape.size())},
                                                             getSInt64Type(convOp->getContext()));
    const auto filterShapeAttr = mlir::DenseElementsAttr::get(filterShapeType, makeArrayRef(filterShape));
    auto filterShapeOp = rewriter.create<IE::ConstantOp>(convOp->getLoc(), filterShapeType, filterShapeAttr);

    auto newFilter = rewriter.createOrFold<IE::ReshapeOp>(convOp->getLoc(), convOp.filter(), filterShapeOp, false);

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
            convOp, convOp.input(), newFilter, convOp.bias(), convOp.stridesAttr(), convOp.pads_beginAttr(),
            convOp.pads_endAttr(), convOp.dilationsAttr(),
            getInt32Attr(convOp.getContext(), checked_cast<uint32_t>(groups)));

    return mlir::success();
}

}  // namespace

void vpux::IE::GroupConvolutionOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                               mlir::MLIRContext* context) {
    patterns.insert<FuseConvAndBias>(context);
    patterns.insert<GroupsToAttr>(context);
}
