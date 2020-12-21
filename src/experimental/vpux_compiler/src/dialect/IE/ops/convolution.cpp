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

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/IR/PatternMatch.h>

#include <ngraph/coordinate.hpp>
#include <ngraph/op/max_pool.hpp>
#include <ngraph/util.hpp>
#include <ngraph/validation_util.hpp>

using namespace vpux;

mlir::LogicalResult vpux::IE::ConvolutionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ConvolutionOpAdaptor conv(operands, attrs);
    if (mlir::failed(conv.verify(loc))) {
        return ::mlir::failure();
    }

    std::function<SmallVector<int64_t, MAX_NUM_DIMS>(mlir::ArrayAttr && arrayAttr)> convertArrayAttrToSmallVector =
            [](mlir::ArrayAttr&& arrayAttr) {
                SmallVector<int64_t, MAX_NUM_DIMS> result;
                for (auto&& a : arrayAttr)
                    result.push_back(a.dyn_cast<mlir::IntegerAttr>().getInt());
                return result;
            };

    auto inShape = conv.input().getType().cast<mlir::RankedTensorType>().getShape();
    auto inType = conv.input().getType().cast<mlir::RankedTensorType>().getElementType();
    auto filterShape = conv.filter().getType().cast<mlir::RankedTensorType>().getShape();

    SmallVector<int64_t, MAX_NUM_DIMS> dataPaddingBelow = convertArrayAttrToSmallVector(conv.pads_end());
    SmallVector<int64_t, MAX_NUM_DIMS> dataPaddingAbove = convertArrayAttrToSmallVector(conv.pads_begin());
    SmallVector<int64_t, MAX_NUM_DIMS> windowStrides = convertArrayAttrToSmallVector(conv.strides());
    SmallVector<int64_t, MAX_NUM_DIMS> windowDilations = convertArrayAttrToSmallVector(conv.dilations());

    auto outputShape =
            ngraph::infer_convolution_forward(nullptr, ngraph::Shape(inShape.begin(), inShape.end()),
                                              ngraph::Strides(windowStrides.size(), 1),  // dummy data dilations
                                              ngraph::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
                                              ngraph::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
                                              ngraph::Shape(filterShape.begin(), filterShape.end()),
                                              ngraph::Strides(windowStrides.begin(), windowStrides.end()),
                                              ngraph::Strides(windowDilations.begin(), windowDilations.end()));

    auto __outputShape = outputShape.get_shape();

    SmallVector<int64_t, MAX_NUM_DIMS> mlirOutputShape(__outputShape.begin(), __outputShape.end());
    inferredReturnShapes.emplace_back(mlirOutputShape, inType);
    return mlir::success();
}

namespace {

class FuseConvAndBias final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    using mlir::OpRewritePattern<IE::ConvolutionOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseConvAndBias::matchAndRewrite(IE::ConvolutionOp convOp, mlir::PatternRewriter& rewriter) const {
    static const auto N = Dim(0);
    static const auto C = Dim(1);
    static const auto H = Dim(2);
    static const auto W = Dim(3);

    if (!convOp.output().hasOneUse()) {
        return mlir::failure();
    }

    auto& use = *convOp.output().getUses().begin();

    auto biasOp = mlir::dyn_cast_or_null<IE::AddOp>(use.getOwner());
    if (biasOp == nullptr) {
        return mlir::failure();
    }
    if (use.getOperandNumber() != 0) {
        return mlir::failure();
    }

    auto convOutShape = getShape(biasOp.input1());
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

    rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(biasOp, convOp.input(), convOp.filter(), biasOp.input2(),
                                                   convOp.strides(), convOp.pads_begin(), convOp.pads_end(),
                                                   convOp.dilations());

    return mlir::success();
}

}  // namespace

void vpux::IE::ConvolutionOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
                                                          mlir::MLIRContext* context) {
    patterns.insert<FuseConvAndBias>(context);
}
