//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
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

class FuseConvAndBias final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    using mlir::OpRewritePattern<IE::AddOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::AddOp biasOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseConvAndBias::matchAndRewrite(IE::AddOp biasOp, mlir::PatternRewriter& rewriter) const {
    static const auto N = Dim(0);
    static const auto C = Dim(1);
    static const auto H = Dim(2);
    static const auto W = Dim(3);

    if (!biasOp.input1().hasOneUse()) {
        return mlir::failure();
    }

    auto convOp = mlir::dyn_cast_or_null<ConvolutionLayerInterface>(biasOp.input1().getDefiningOp());
    if (convOp == nullptr) {
        return mlir::failure();
    }

    if (convOp->getNumOperands() != 2 || convOp.bias() != nullptr) {
        return mlir::failure();
    }

    auto convOutShape = getShape(convOp.output());
    auto biasShape = getShape(biasOp.input2());

    if (convOutShape.size() != 4 || biasShape.size() != 4) {
        return mlir::failure();
    }
    if (biasShape[N] != 1 || biasShape[H] != 1 || biasShape[W] != 1) {
        return mlir::failure();
    }
    if (biasShape[C] != convOutShape[C]) {
        return mlir::failure();
    }

    auto* newConv = rewriter.clone(*convOp);
    newConv->insertOperands(newConv->getNumOperands(), biasOp.input2());

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

void vpux::IE::ConvolutionOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
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
            return printTo(mlir::emitError(loc), "Wrong filter shape '{0}'", filterShape);
        }

        groups = conv.groups().getInt();
    } else {
        if (filterShape.size() != 5) {
            return printTo(mlir::emitError(loc), "Wrong filter shape '{0}'", filterShape);
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

    if (!mlir::isa_and_nonnull<ConstantInterface>(convOp.filter().getDefiningOp())) {
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

void vpux::IE::GroupConvolutionOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
                                                               mlir::MLIRContext* context) {
    patterns.insert<FuseConvAndBias>(context);
    patterns.insert<GroupsToAttr>(context);
}
