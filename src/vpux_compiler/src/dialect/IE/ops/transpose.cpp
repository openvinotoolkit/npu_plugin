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

//
// ConvertConstToAttr
//

namespace {

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

}  // namespace

void vpux::IE::TransposeOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
                                                        mlir::MLIRContext* context) {
    patterns.insert<ConvertConstToAttr>(context);
}
