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

#include <vpux/compiler/utils/attributes.hpp>
#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

namespace {

mlir::LogicalResult getOrder(IE::TransposeOpAdaptor transpose, SmallVector<int64_t>& order, mlir::Location loc) {
    const auto getDefaultOrder = [](mlir::ShapedType inType) {
        SmallVector<int64_t> orderIndices{};
        for (const auto& idx : irange(inType.getRank()) | reversed) {
            orderIndices.push_back(idx);
        }

        return orderIndices;
    };

    if (transpose.order() != nullptr && transpose.order_value() != nullptr) {
        return errorAt(loc, "Ambiguous order representation");
    }
    if (transpose.order() == nullptr && transpose.order_value() == nullptr) {
        return errorAt(loc, "Missed order representation");
    }

    const auto inDataType = transpose.input().getType().cast<mlir::ShapedType>();

    if (transpose.order() != nullptr) {
        auto orderOp = transpose.order().getDefiningOp<IE::ConstantOp>();
        if (orderOp == nullptr) {
            return errorAt(loc, "Only constant input is supported");
        }

        const auto orderContent = orderOp.getContent().getValues<int32_t>();
        order = orderContent.empty() ? getDefaultOrder(inDataType) : orderContent;

        return mlir::success();
    }

    const auto perm = DimsOrder::fromAffineMap(transpose.order_value().getValue());
    order = to_small_vector(irange(perm.numDims()) | transformed([&](int64_t idx) {
                                return checked_cast<int64_t>(perm.dimAt(idx).ind());
                            }));

    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::IE::TransposeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::TransposeOpAdaptor transpose(operands, attrs);
    if (mlir::failed(transpose.verify(loc))) {
        return mlir::failure();
    }

    const auto inDataType = transpose.input().getType().cast<mlir::ShapedType>();
    const auto inDataShape = inDataType.getShape();

    SmallVector<int64_t> order{};
    if (getOrder(transpose, order, loc).failed()) {
        return mlir::failure();
    }

    if (inDataShape.size() != order.size()) {
        return errorAt(loc, "Order vector size doesn't match input rank");
    }

    const auto outRank = static_cast<int64_t>(inDataShape.size());
    SmallVector<int64_t> outShapeVec(outRank);

    for (size_t i = 0; i < order.size(); ++i) {
        if (order[i] >= outRank) {
            return mlir::failure();
        }

        outShapeVec[i] = inDataShape[order[i]];
    }

    inferredReturnShapes.emplace_back(makeArrayRef(outShapeVec), inDataType.getElementType());

    return mlir::success();
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

    SmallVector<int64_t> order{};
    if (getOrder(transposeOp, order, transposeOp->getLoc()).failed()) {
        return mlir::failure();
    }

    const auto perm = to_small_vector(order | transformed([](int64_t val) {
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

    SmallVector<int64_t> prevOrder{};
    VPUX_THROW_UNLESS(getOrder(prevTransposeOp, prevOrder, prevTransposeOp->getLoc()).succeeded(),
                      "Failed to get order for Transpose operation '{0}'", prevTransposeOp->getName());

    SmallVector<int64_t> order{};
    VPUX_THROW_UNLESS(getOrder(transposeOp, order, transposeOp->getLoc()).succeeded(),
                      "Failed to get order for Transpose operation '{0}'", transposeOp->getName());

    const auto prevPerm = to_small_vector(prevOrder | transformed([](int64_t val) {
                                              return checked_cast<unsigned>(val);
                                          }));

    const auto perm = to_small_vector(order | transformed([](int64_t val) {
                                          return checked_cast<unsigned>(val);
                                      }));

    auto prevPermMap = mlir::AffineMap::getPermutationMap(prevPerm, transposeOp->getContext());
    auto permMap = mlir::AffineMap::getPermutationMap(perm, transposeOp->getContext());

    const auto permAttr = mlir::AffineMapAttr::get(permMap.compose(prevPermMap));
    rewriter.replaceOpWithNewOp<IE::TransposeOp>(transposeOp, transposeOp.getType(), prevTransposeOp.input(), nullptr,
                                                 permAttr);

    return mlir::success();
}

}  // namespace

void vpux::IE::TransposeOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
                                                        mlir::MLIRContext* context) {
    patterns.insert<ConvertConstToAttr>(context);
    patterns.insert<FuseTransposes>(context);
}

mlir::OpFoldResult vpux::IE::TransposeOp::fold(ArrayRef<mlir::Attribute>) {
    if (input().getType() != output().getType()) {
        return nullptr;
    }

    return input();
}
