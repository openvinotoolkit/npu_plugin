//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/PatternMatch.h>

#include <numeric>

using namespace vpux;

//
// getOutShape
//

namespace {

mlir::FailureOr<SmallVector<int64_t>> getOutShape(IE::ReshapeOpAdaptor reshape, mlir::Location loc) {
    if (reshape.shape() != nullptr && reshape.shape_value() != nullptr) {
        return errorAt(loc, "Ambiguous shape representation");
    }
    if (reshape.shape() == nullptr && reshape.shape_value() == nullptr) {
        return errorAt(loc, "Missed shape representation");
    }

    if (reshape.shape_value() != nullptr) {
        return parseIntArrayAttr<int64_t>(reshape.shape_value());
    }

    auto shapeConst = reshape.shape().getDefiningOp<Const::DeclareOp>();
    if (shapeConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for shape");
    }

    const auto shapeContent = shapeConst.content();
    auto shapeVec = to_small_vector(shapeContent.getValues<int64_t>());

    const auto specialZero = reshape.special_zero();

    const auto zeroDims = std::count_if(shapeVec.begin(), shapeVec.end(), [](int64_t v) {
        return v == 0;
    });
    const auto negativeDims = std::count_if(shapeVec.begin(), shapeVec.end(), [](int64_t v) {
        return v == -1;
    });

    if (negativeDims > 1) {
        return errorAt(loc, "Shape can not contain more than 1 negative value");
    }

    if (!(zeroDims != 0 && specialZero) && negativeDims == 0) {
        return shapeVec;
    } else {
        const auto inShape = to_small_vector(reshape.input().getType().cast<mlir::ShapedType>().getShape());

        auto dividend = std::accumulate(inShape.begin(), inShape.end(), int64_t(1), std::multiplies<int64_t>());

        for (size_t i = 0; i < shapeVec.size(); ++i) {
            auto& v = shapeVec[i];

            if (v == 0 && specialZero) {
                if (i >= inShape.size()) {
                    return errorAt(loc, "Shape value at '{0}' is out of range '{1}'", i, inShape.size());
                }

                v = inShape[i];
            }

            if (v > 0) {
                if (dividend % v != 0) {
                    return errorAt(loc, "Shape value at '{0}' ('{1}') is invalid", i, v);
                }

                dividend /= v;
            }
        }

        if (negativeDims > 0) {
            const auto negIt = std::find(shapeVec.begin(), shapeVec.end(), -1);
            VPUX_THROW_UNLESS(negIt != shapeVec.end(), "Shape vector broken");

            *negIt = dividend;
        }

        return shapeVec;
    }
}

}  // namespace

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::ReshapeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ReshapeOpAdaptor reshape(operands, attrs);
    if (mlir::failed(reshape.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = getOutShape(reshape, loc);
    if (mlir::failed(outShape)) {
        return mlir::failure();
    }

    const auto inType = reshape.input().getType().cast<mlir::RankedTensorType>();

    const auto outDesc = IE::getTensorAttr(ctx, DimsOrder::fromNumDims(outShape->size()), IE::getMemorySpace(inType),
                                           IE::isSparse(inType));

    inferredReturnShapes.emplace_back(outShape.getValue(), inType.getElementType(), outDesc);
    return mlir::success();
}

//
// inferElemTypeInfo
//

void vpux::IE::ReshapeOp::inferElemTypeInfo(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    const auto inputElemType = info.getInput(0);

    // Do not propagate element type down in per channel case.
    // E#31030
    if (inputElemType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>() == nullptr) {
        for (size_t outputInd = 0; outputInd < info.getNumOutputs(); ++outputInd) {
            info.setOutput(outputInd, inputElemType);
        }
    }
}

void vpux::IE::ReshapeOp::inferElemTypeInfoUp(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    const auto outputElemType = info.getOutput(0);

    if (outputElemType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>() != nullptr) {
        // E#31029: implement propagate type up for per channel, currently it leads to failures in later passes.
        return;
    }

    for (size_t inputInd = 0; inputInd < info.getNumInputs(); ++inputInd) {
        info.setInput(inputInd, outputElemType);
    }
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ReshapeOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (input().getType() == output().getType()) {
        return input();
    }

    VPUX_THROW_UNLESS(!operands.empty(), "Wrong number of operands : {0}", operands.size());

    if (const auto attr = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return attr.reshape(getShape(output()));
    }

    return nullptr;
}

//
// FuseReshapes
//

namespace {

class FuseReshapes final : public mlir::OpRewritePattern<IE::ReshapeOp> {
public:
    using mlir::OpRewritePattern<IE::ReshapeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseReshapes::matchAndRewrite(IE::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const {
    auto prevOp = origOp.input().getDefiningOp();
    if (prevOp == nullptr) {
        return mlir::failure();
    }
    if (!mlir::isa<IE::SqueezeOp, IE::UnsqueezeOp, IE::ReshapeOp, IE::AffineReshapeOp>(prevOp)) {
        return mlir::failure();
    }

    const auto outputShape = origOp.getType().getShape();
    const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);

    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, prevOp->getOperand(0), nullptr, false, outputShapeAttr);
    return mlir::success();
}

}  // namespace

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::ReshapeOp> {
public:
    using mlir::OpRewritePattern<IE::ReshapeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const {
    if (origOp.shape_value().hasValue()) {
        return mlir::failure();
    }

    const auto outShape = getOutShape(origOp, origOp->getLoc());
    if (mlir::failed(outShape)) {
        return mlir::failure();
    }

    const auto outShapeAttr = getIntArrayAttr(getContext(), outShape.getValue());

    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, origOp.input(), nullptr, false, outShapeAttr);
    return mlir::success();
}

}  // namespace

//
// ConvertToAffineReshape
//

namespace {

struct MinDimension {
    std::size_t& shapeIdx;
    ArrayRef<int64_t> shape;
    int64_t largeDimQuotient;

    MinDimension(std::size_t& shapeIdx, ArrayRef<int64_t> shape, const int64_t largeDimQuotient)
            : shapeIdx(shapeIdx), shape(shape), largeDimQuotient(largeDimQuotient){};
};

void handleConsecutiveOnes(ArrayRef<int64_t> inShape, ArrayRef<int64_t> outShape, std::size_t& startIn,
                           std::size_t& startOut, SmallVector<SmallVector<int64_t>>& reassociationVec) {
    std::size_t endIn = startIn;
    while (endIn < inShape.size() && inShape[endIn] == 1)
        endIn++;

    std::size_t endOut = startOut;
    while (endOut < outShape.size() && outShape[endOut] == 1)
        endOut++;

    for (; startIn < endIn && startOut < endOut; ++startIn, ++startOut) {
        reassociationVec[startIn].push_back(static_cast<int64_t>(startOut));
    }

    while (startIn < endIn) {
        reassociationVec[startIn].push_back(static_cast<int64_t>(startOut - 1));
        startIn++;
    }

    while (startOut < endOut) {
        reassociationVec[startIn - 1].push_back(static_cast<int64_t>(startOut));
        startOut++;
    }
}

// Note: When having dims equal to 1 in one of the shapes that do not have a corresponding 1 in the other shape, there
// might be multiple dim associations possible. The current algorithm takes only one into consideration.
// E.g.: 1 x 2 x 2 x 1 x 2 x 3 -> 1 x 4 x 6 has 2 possible mappings:
//      {0} -> {0}, {1, 2, 3} -> {1}, {4, 5} -> {2} (this one is computed by the fcn below)
//      {0} -> {0}, {1, 2} -> {1}, {3, 4, 5} -> {2}
mlir::FailureOr<SmallVector<SmallVector<int64_t>>> getReassociationMap(ArrayRef<int64_t> inShape,
                                                                       ArrayRef<int64_t> outShape) {
    const auto inSize = inShape.size();
    const auto outSize = outShape.size();

    const auto nextDimIsOne = [](ArrayRef<int64_t> shape, const std::size_t index) -> bool {
        return index + 1 < shape.size() && shape[index + 1] == 1;
    };

    SmallVector<SmallVector<int64_t>> reassociationVec(inSize);
    std::size_t inIdx = 0, outIdx = 0;
    for (; inIdx < inSize && outIdx < outSize; ++inIdx, ++outIdx) {
        if (inShape[inIdx] == 1 && outShape[outIdx] == 1) {
            // Pair dims equal to 1 that have corresponding dims in the other shape
            handleConsecutiveOnes(inShape, outShape, inIdx, outIdx, reassociationVec);

            if (inIdx >= inSize || outIdx >= outSize)
                break;
        }

        // If both dims are equal, pick the one that has a dim of 1 after it. If there is no corresponding dim equal to
        // 1 in the other shape, the mapping dim_large = 1 x dim_small will be added. Without that extra condition,
        // there could be cases where that extra 1 remains floating, leading the algorithm to decide that there is no
        // valid mapping between shapes.
        const bool isInputSmallerDim = inShape[inIdx] < outShape[outIdx] ||
                                       (inShape[inIdx] == outShape[outIdx] && nextDimIsOne(inShape, inIdx));
        auto minimum = isInputSmallerDim ? MinDimension(inIdx, inShape, outShape[outIdx])
                                         : MinDimension(outIdx, outShape, inShape[inIdx]);

        do {
            if (minimum.largeDimQuotient % minimum.shape[minimum.shapeIdx] != 0)
                return mlir::failure();

            reassociationVec[inIdx].push_back(static_cast<int64_t>(outIdx));

            minimum.largeDimQuotient /= minimum.shape[minimum.shapeIdx];

            if (minimum.largeDimQuotient == 1) {
                // Exit loop if the next dim isn't 1 or if there are 1s on next dim of both shapes
                if (!nextDimIsOne(minimum.shape, minimum.shapeIdx) ||
                    (nextDimIsOne(inShape, inIdx) && nextDimIsOne(outShape, outIdx))) {
                    break;
                }
            }

            ++minimum.shapeIdx;
        } while (minimum.shapeIdx < minimum.shape.size());
    }

    // One of the shapes has trailing 1s that cannot be the result of decomposing the last dim of the other shape
    if (inIdx < inSize || outIdx < outSize)
        return mlir::failure();

    return reassociationVec;
}

// AffineReshapes represent a subset of Reshape ops that can have their output layout inferred from input layout.
// The condition that Reshape must satisfy to be an AffineReshape is to have a clear mapping between the dims of the
// input and output shapes. That happens when each dim of the input shape can either be decomposed into several adjacent
// dims of the output shape OR the multiplication of it with adjacent input dims results in an output dim.
// E.g. n x c x h x w -> n x c x (h * w) (AffineReshape)
//      c x h x w -> (c / k) x k x h x w (AffineReshape)
//      n x c x h x w -> n x (c * w) x h (not AffineReshape - c, w not adjacent)
//      c x h x w -> (c / k) x (h * k) x w (not AffineReshape - not an exact decomposition into dims of the
//      other shape)
class ConvertToAffineReshape final : public mlir::OpRewritePattern<IE::ReshapeOp> {
public:
    using mlir::OpRewritePattern<IE::ReshapeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertToAffineReshape::matchAndRewrite(IE::ReshapeOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    const auto outputShape = origOp.getType().getShape();
    const auto outShapeAttr = getIntArrayAttr(getContext(), outputShape);

    const auto inShape = origOp.input().getType().cast<mlir::ShapedType>().getShape();
    const auto reassociationMap = getReassociationMap(inShape, outputShape);
    if (mlir::failed(reassociationMap)) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(
            origOp, origOp.input(), getIntArrayOfArray(getContext(), reassociationMap.getValue()), outShapeAttr);
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::ReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.insert<FuseReshapes>(ctx);
    patterns.insert<ConvertConstToAttr>(ctx);
    patterns.insert<ConvertToAffineReshape>(ctx);
}
