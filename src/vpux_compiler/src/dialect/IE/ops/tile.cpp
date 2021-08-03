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

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

namespace {

mlir::SmallVector<int64_t> calcTileOutputShape(mlir::Value input, Const::DeclareOp repeats) {
    const auto inType = input.getType().cast<mlir::ShapedType>();

    const auto repeatsContent = repeats.content();
    const auto repeatsVals = repeatsContent.getValues<int64_t>();

    auto outShape = to_small_vector(inType.getShape());

    // If number of elements in *"repeats"* is more than shape of *"data"*, then *"data"* will be promoted to
    // "*repeats*" by prepending new axes, e.g. let's shape of *"data"* is equal to (2, 3) and *"repeats"* is equal to
    // [2, 2, 2], then shape of *"data"* will be promoted to (1, 2, 3) and result shape will be (2, 4, 6).
    //
    // If number of elements in *"repeats"* is less than shape of *"data"*, then *"repeats"* will be promoted to
    // "*data*" by prepending 1's to it, e.g. let's shape of *"data"* is equal to (4, 2, 3) and *"repeats"* is equal to
    // [2, 2], then *"repeats"* will be promoted to [1, 2, 2] and result shape will be (4, 4, 6)

    while (repeatsVals.size() > outShape.size()) {
        outShape.insert(outShape.begin(), 1);
    }

    auto out_shape_iter = std::prev(outShape.end());
    auto repeats_iter = std::prev(repeatsVals.end());
    for (; out_shape_iter != std::prev(outShape.begin()) && repeats_iter != std::prev(repeatsVals.begin());
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
    auto repeatsConst = tile.repeats().getDefiningOp<Const::DeclareOp>();
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

    const auto unsqueezeParamsAttr = getIntArrayAttr(getContext(), unsqueezeParam);
    auto unsqueezeOp = rewriter.create<IE::UnsqueezeOp>(origOp->getLoc(), origOp.input(), nullptr, unsqueezeParamsAttr);

    rewriter.replaceOpWithNewOp<IE::TileOp>(origOp, origOp.getType(), unsqueezeOp->getResult(0), origOp.repeats());
    return mlir::success();
}

}  // namespace

void vpux::IE::TileOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
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
