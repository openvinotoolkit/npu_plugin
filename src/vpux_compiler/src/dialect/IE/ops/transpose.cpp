//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

namespace {

mlir::LogicalResult getOrder(IE::TransposeOpAdaptor transpose, SmallVector<uint64_t>& order, mlir::Location loc) {
    const auto getDefaultOrder = [](mlir::ShapedType inType) {
        SmallVector<uint64_t> orderIndices{};
        for (const auto& idx : irange(inType.getRank()) | reversed) {
            orderIndices.push_back(idx);
        }

        return orderIndices;
    };

    if (transpose.order() != nullptr && transpose.order_value().hasValue()) {
        return errorAt(loc, "Ambiguous order representation");
    }
    if (transpose.order() == nullptr && !transpose.order_value().hasValue()) {
        return errorAt(loc, "Missed order representation");
    }

    const auto inDataType = transpose.input().getType().cast<mlir::ShapedType>();

    if (transpose.order() != nullptr) {
        auto orderOp = transpose.order().getDefiningOp<Const::DeclareOp>();
        if (orderOp == nullptr) {
            return errorAt(loc, "Only constant input is supported");
        }

        const auto orderContent = orderOp.content();
        const auto orderVals = orderContent.getValues<uint64_t>();

        order = orderVals.empty() ? getDefaultOrder(inDataType) : to_small_vector(orderVals);

        return mlir::success();
    }

    const auto perm = DimsOrder::fromAffineMap(transpose.order_value().getValue());
    order = to_small_vector(irange(perm.numDims()) | transformed([&](uint64_t idx) {
                                return checked_cast<uint64_t>(perm.dimAt(idx).ind());
                            }));

    return mlir::success();
}

mlir::Type inferElemType(mlir::AffineMap map, mlir::Type inputElemType) {
    const auto perAxisQType = inputElemType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>();
    if (perAxisQType == nullptr) {
        return inputElemType;
    }

    const auto origAxis = perAxisQType.getQuantizedDimension();
    const auto newAxis = DimsOrder::fromAffineMap(map).toPermutation()[origAxis];

    return mlir::quant::UniformQuantizedPerAxisType::get(
            perAxisQType.getFlags(), perAxisQType.getStorageType(), perAxisQType.getExpressedType(),
            perAxisQType.getScales(), perAxisQType.getZeroPoints(), newAxis.ind(), perAxisQType.getStorageTypeMin(),
            perAxisQType.getStorageTypeMax());
}

}  // namespace

mlir::LogicalResult vpux::IE::TransposeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::TransposeOpAdaptor transpose(operands, attrs);
    if (mlir::failed(transpose.verify(loc))) {
        return mlir::failure();
    }

    const auto inDataType = transpose.input().getType().cast<mlir::RankedTensorType>();
    const auto inDataShape = inDataType.getShape();

    SmallVector<uint64_t> order{};
    if (::getOrder(transpose, order, loc).failed()) {
        return mlir::failure();
    }

    if (inDataShape.size() != order.size()) {
        return errorAt(loc, "Order vector size doesn't match input rank");
    }

    const auto outRank = static_cast<uint64_t>(inDataShape.size());
    SmallVector<int64_t> outShapeVec(outRank);

    for (size_t i = 0; i < order.size(); ++i) {
        if (order[i] >= outRank) {
            return mlir::failure();
        }

        outShapeVec[i] = inDataShape[order[i]];
    }

    SmallVector<uint32_t> uorder;
    std::transform(order.begin(), order.end(), std::back_inserter(uorder), [](uint64_t dim) -> uint32_t {
        return static_cast<uint32_t>(dim);
    });

    const auto permutationMap = mlir::AffineMap::getPermutationMap(makeArrayRef(uorder), ctx);
    const auto outputElemType = inferElemType(permutationMap, inDataType.getElementType());

    const auto outDesc = IE::getTensorAttr(IE::getOrder(inDataType), nullptr);

    inferredReturnShapes.emplace_back(makeArrayRef(outShapeVec), outputElemType, outDesc);
    return mlir::success();
}

//
// inferElemTypeInfo
//

void vpux::IE::TransposeOp::inferElemTypeInfo(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    auto outputElemType = inferElemType((*this).order_value().getValue(), info.getInput(0));

    for (size_t outputInd = 0; outputInd < info.getNumOutputs(); ++outputInd) {
        info.setOutput(outputInd, outputElemType);
    }
}

void vpux::IE::TransposeOp::inferElemTypeInfoUp(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    const auto outputElemType = info.getOutput(0);

    if (outputElemType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>() != nullptr) {
        // E#31029: implement propagate type up for per channel, currently it leads to failures in later passes.
        return;
    }

    for (size_t inputInd = 0; inputInd < info.getNumInputs(); ++inputInd) {
        info.setInput(inputInd, outputElemType);
    }
}

namespace {

//
// ConvertConstToAttr
//

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::TransposeOp> {
public:
    using mlir::OpRewritePattern<IE::TransposeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposeOp transposeOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::TransposeOp transposeOp,
                                                        mlir::PatternRewriter& rewriter) const {
    if (transposeOp.order_value().hasValue()) {
        return mlir::failure();
    }

    SmallVector<uint64_t> order{};
    if (getOrder(transposeOp, order, transposeOp->getLoc()).failed()) {
        return mlir::failure();
    }

    const auto perm = to_small_vector(order | transformed([](uint64_t val) {
                                          return checked_cast<unsigned>(val);
                                      }));

    const auto orderAttr =
            mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(perm, transposeOp->getContext()));

    rewriter.replaceOpWithNewOp<IE::TransposeOp>(transposeOp, transposeOp.getType(), transposeOp.input(), nullptr,
                                                 orderAttr);

    return mlir::success();
}

//
// FuseTransposes
//

class FuseTransposes final : public mlir::OpRewritePattern<IE::TransposeOp> {
public:
    using mlir::OpRewritePattern<IE::TransposeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposeOp transposeOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseTransposes::matchAndRewrite(IE::TransposeOp transposeOp,
                                                    mlir::PatternRewriter& rewriter) const {
    if (!transposeOp.input().hasOneUse()) {
        return mlir::failure();
    }

    auto prevTransposeOp = mlir::dyn_cast_or_null<IE::TransposeOp>(transposeOp.input().getDefiningOp());
    if (prevTransposeOp == nullptr) {
        return mlir::failure();
    }

    SmallVector<uint64_t> prevOrder{};
    VPUX_THROW_UNLESS(getOrder(prevTransposeOp, prevOrder, prevTransposeOp->getLoc()).succeeded(),
                      "Failed to get order for Transpose operation '{0}'", prevTransposeOp->getName());

    SmallVector<uint64_t> order{};
    VPUX_THROW_UNLESS(getOrder(transposeOp, order, transposeOp->getLoc()).succeeded(),
                      "Failed to get order for Transpose operation '{0}'", transposeOp->getName());

    const auto prevPerm = to_small_vector(prevOrder | transformed([](uint64_t val) {
                                              return checked_cast<unsigned>(val);
                                          }));

    const auto perm = to_small_vector(order | transformed([](uint64_t val) {
                                          return checked_cast<unsigned>(val);
                                      }));

    auto prevPermMap = mlir::AffineMap::getPermutationMap(prevPerm, transposeOp->getContext());
    auto permMap = mlir::AffineMap::getPermutationMap(perm, transposeOp->getContext());

    const auto permAttr = mlir::AffineMapAttr::get(permMap.compose(prevPermMap));
    rewriter.replaceOpWithNewOp<IE::TransposeOp>(transposeOp, transposeOp.getType(), prevTransposeOp.input(), nullptr,
                                                 permAttr);

    return mlir::success();
}

//
// CollapseMutualTransposes
//

class CollapseMutualTransposes final : public mlir::OpRewritePattern<IE::TransposeOp> {
public:
    using mlir::OpRewritePattern<IE::TransposeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposeOp transposeOp, mlir::PatternRewriter& rewriter) const final;
};

bool isTrivialTransformation(const ShapeRef inShape, const ShapeRef outShape) {
    // Remove all trivial dimensions.
    const auto isTrivial = [](const int64_t dimVal) -> bool {
        return dimVal > 1;
    };
    Shape inShapeDiminished;
    std::copy_if(inShape.begin(), inShape.end(), std::back_inserter(inShapeDiminished), isTrivial);

    Shape outShapeDiminished;
    std::copy_if(outShape.begin(), outShape.end(), std::back_inserter(outShapeDiminished), isTrivial);

    // Check that resulting shapes preserve the order of dimensions.
    return inShapeDiminished == outShapeDiminished;
}

bool canBeCollapsed(IE::TransposeOp op) {
    // First, check whether the input of that transpose operation is a reshape operation.
    const auto lastTransposeIn = op.input();
    auto maybeReshapeOp = lastTransposeIn.getDefiningOp();
    if (!mlir::isa_and_nonnull<IE::ReshapeOp, IE::AffineReshapeOp, IE::SqueezeOp, IE::UnsqueezeOp>(maybeReshapeOp)) {
        return false;
    }

    // Now, find out whether this reshape operation has another transpose as an input.
    const auto reshapeIn = maybeReshapeOp->getOperand(0);
    auto firstTranspose = reshapeIn.getDefiningOp<IE::TransposeOp>();
    if (firstTranspose == nullptr) {
        return false;
    }

    // Only trivial reshapes can be collapsed.
    // Trivial means that all dimensions larger than 1 preserve order.
    // Examples:
    // 1x1x28x70 -> Reshape -> 1x28x70 -- trivial
    // 1x28x70x1 -> Reshape -> 1x28x70 -- trivial
    // 1x28x1x70 -> Reshape -> 1x28x70 -- trivial
    // 1x28x1x70 -> Reshape -> 1x70x28 -- non-trivial, since the order is not preserved.
    if (!isTrivialTransformation(getShape(maybeReshapeOp->getOperand(0)), getShape(maybeReshapeOp->getResult(0)))) {
        return false;
    }

    // Check that the second transpose is the inverse of the first one.
    if (!isTrivialTransformation(getShape(firstTranspose.input()), getShape(op.output()))) {
        return false;
    }

    return true;
}

// Replaces chain of Transpose -> Reshape -> Transpose with single Reshape.
// Checks whether `Reshape` operation between two `Transpose` operations gathers any dimensions.
// If so, it checks whether the second transposition cancels the effect of the first one.
// In such cases, the whole subgraph can be replaced with a single reshape.
mlir::LogicalResult CollapseMutualTransposes::matchAndRewrite(IE::TransposeOp transposeOp,
                                                              mlir::PatternRewriter& rewriter) const {
    if (!canBeCollapsed(transposeOp)) {
        return mlir::failure();
    }

    const auto lastTransposeIn = transposeOp.input();
    auto maybeReshapeOp = lastTransposeIn.getDefiningOp();
    const auto reshapeIn = maybeReshapeOp->getOperand(0);
    auto firstTranspose = reshapeIn.getDefiningOp<IE::TransposeOp>();
    const auto firstTransposeIn = firstTranspose.input();

    const auto shape = transposeOp.output().getType().cast<vpux::NDTypeInterface>().getShape();
    const auto newShape = to_small_vector(shape);
    const auto newShapeAttr = getIntArrayAttr(rewriter.getContext(), newShape);
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(transposeOp, firstTransposeIn, nullptr, false, newShapeAttr);

    return mlir::success();
}

}  // namespace

void vpux::IE::TransposeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
    patterns.add<FuseTransposes>(context);
    patterns.add<CollapseMutualTransposes>(context);
}

mlir::OpFoldResult vpux::IE::TransposeOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (const auto cst = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        if (order_value().hasValue()) {
            const auto orderAttr = DimsOrder::fromAffineMap(order_value().getValue());
            return cst.transpose(orderAttr);
        }
    }

    if (input().getType() == output().getType() && order_value().hasValue()) {
        const auto inputRank = static_cast<uint32_t>(input().getType().cast<mlir::ShapedType>().getRank());
        const auto idMap = mlir::AffineMap::getMultiDimIdentityMap(inputRank, getContext());
        const auto orderMap = order_value().getValue();
        if (idMap == orderMap) {
            return input();
        }
    }

    return nullptr;
}
