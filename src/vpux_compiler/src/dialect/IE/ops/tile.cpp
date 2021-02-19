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

#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

namespace {

mlir::SmallVector<int64_t> calcTileOutputShape(mlir::Value input, ConstantInterface repeats) {
    const auto inType = input.getType().cast<mlir::ShapedType>();

    const auto repeats_content = repeats.getContent().getValues<int64_t>();

    auto outShape = to_small_vector(inType.getShape());

    // If number of elements in *"repeats"* is more than shape of *"data"*, then *"data"* will be promoted to
    // "*repeats*" by prepending new axes, e.g. let's shape of *"data"* is equal to (2, 3) and *"repeats"* is equal to
    // [2, 2, 2], then shape of *"data"* will be promoted to (1, 2, 3) and result shape will be (2, 4, 6).
    //
    // If number of elements in *"repeats"* is less than shape of *"data"*, then *"repeats"* will be promoted to
    // "*data*" by prepending 1's to it, e.g. let's shape of *"data"* is equal to (4, 2, 3) and *"repeats"* is equal to
    // [2, 2], then *"repeats"* will be promoted to [1, 2, 2] and result shape will be (4, 4, 6)

    while (repeats_content.size() > outShape.size()) {
        outShape.insert(outShape.begin(), 1);
    }

    auto out_shape_iter = std::prev(outShape.end());
    auto repeats_iter = std::prev(repeats_content.end());
    for (; out_shape_iter != std::prev(outShape.begin()) && repeats_iter != std::prev(repeats_content.begin());
         --out_shape_iter, --repeats_iter) {
        *out_shape_iter *= *repeats_iter;
    }
    return outShape;
}

}  // namespace

mlir::LogicalResult vpux::IE::TileOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::TileOpAdaptor tile(operands, attrs);
    if (mlir::failed(tile.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = tile.input().getType().cast<mlir::ShapedType>();
    auto repeatsConst = tile.repeats().getDefiningOp<ConstantInterface>();
    if (repeatsConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for repeats");
    }

    auto outShape = calcTileOutputShape(tile.input(), repeatsConst);

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}

namespace {

class AddUnsqueeze final : public mlir::OpRewritePattern<IE::TileOp> {
public:
    using mlir::OpRewritePattern<IE::TileOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::TileOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult AddUnsqueeze::matchAndRewrite(IE::TileOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto numRepeats = origOp.repeats().getType().cast<mlir::ShapedType>().getNumElements();
    const auto inputRank = origOp.input().getType().cast<mlir::ShapedType>().getRank();

    if (numRepeats <= inputRank) {
        // don't need to increase rank of input tensor
        return mlir::failure();
    }

    const auto nDimsToAdd = numRepeats - inputRank;

    if (nDimsToAdd <= 0) {
        // don't need to increase rank of input tensor
        return mlir::failure();
    }
    SmallVector<int64_t> unsqueezeParam;
    for (int i = 0; i < nDimsToAdd; ++i) {
        unsqueezeParam.push_back(i);
    }
    const auto unsqueezeParamType = mlir::RankedTensorType::get({checked_cast<int64_t>(unsqueezeParam.size())},
                                                                getSInt64Type(origOp->getContext()));
    const auto unsqueezeParamAttr = mlir::DenseElementsAttr::get(unsqueezeParamType, makeArrayRef(unsqueezeParam));
    auto unsqueezeParamConst =
            rewriter.create<IE::ConstantOp>(origOp->getLoc(), unsqueezeParamType, unsqueezeParamAttr);

    auto unsqueezeOp =
            rewriter.create<IE::UnsqueezeOp>(origOp->getLoc(), origOp.input(), unsqueezeParamConst.getResult());

    rewriter.replaceOpWithNewOp<IE::TileOp>(origOp, origOp.getType(), unsqueezeOp->getResult(0), origOp.repeats());

    return mlir::success();
}

}  // namespace

void vpux::IE::TileOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* ctx) {
    patterns.insert<AddUnsqueeze>(ctx);
}

mlir::OpFoldResult vpux::IE::TileOp::fold(ArrayRef<mlir::Attribute>) {
    if (input().getType() != output().getType()) {
        return nullptr;
    }
    // Tile with current param do nothing and should be optimized
    return input();
}

mlir::LogicalResult vpux::IE::PerAxisTileOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::PerAxisTileOpAdaptor perAxisTile(operands, attrs);
    if (mlir::failed(perAxisTile.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = perAxisTile.input().getType().cast<mlir::ShapedType>();

    const auto axis = checked_cast<unsigned int>(perAxisTile.axis().getInt());
    const auto tiles = checked_cast<unsigned int>(perAxisTile.tiles().getInt());

    auto outShape = to_small_vector(inType.getShape());

    if (axis > outShape.size()) {
        return errorAt(loc, "Axis is out of range. Avaliable range [0, {0}), but got axis = {1}", outShape.size(),
                       axis);
    }

    outShape[axis] *= tiles;

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}
