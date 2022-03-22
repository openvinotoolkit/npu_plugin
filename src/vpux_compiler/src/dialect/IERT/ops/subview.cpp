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

#include "vpux/compiler/dialect/IERT/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// build
//

void vpux::IERT::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                  ShapeRef static_offsets, ShapeRef static_sizes) {
    build(builder, state, input, static_offsets.raw(), static_sizes.raw());
}

void vpux::IERT::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                  ArrayRef<int64_t> static_offsets, ArrayRef<int64_t> static_sizes) {
    build(builder, state, input, getIntArrayAttr(builder.getContext(), static_offsets),
          getIntArrayAttr(builder.getContext(), static_sizes));
}

void vpux::IERT::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                  mlir::ArrayAttr static_offsets, mlir::ArrayAttr static_sizes) {
    build(builder, state, input, static_offsets, static_sizes, nullptr);
}

void vpux::IERT::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                  ShapeRef static_offsets, ShapeRef static_sizes, ShapeRef static_strides) {
    build(builder, state, input, static_offsets.raw(), static_sizes.raw(), static_strides.raw());
}

void vpux::IERT::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                  ArrayRef<int64_t> static_offsets, ArrayRef<int64_t> static_sizes,
                                  ArrayRef<int64_t> static_strides) {
    build(builder, state, input, getIntArrayAttr(builder.getContext(), static_offsets),
          getIntArrayAttr(builder.getContext(), static_sizes), getIntArrayAttr(builder.getContext(), static_strides));
}

//
// ViewLikeOpInterface
//

mlir::Value vpux::IERT::SubViewOp::getViewSource() {
    return source();
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::IERT::SubViewOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                            mlir::Optional<mlir::Location> optLoc,
                                                            mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                            mlir::RegionRange /*regions*/,
                                                            mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IERT::SubViewOpAdaptor subViewOp(operands, attrs);
    if (mlir::failed(subViewOp.verify(loc))) {
        return mlir::failure();
    }

    const auto origType = subViewOp.source().getType().cast<vpux::NDTypeInterface>();

    const auto subViewShape = parseIntArrayAttr<int64_t>(subViewOp.static_sizes());
    const auto subViewOffsets = parseIntArrayAttr<int64_t>(subViewOp.static_offsets());
    const auto subViewStrides = subViewOp.static_strides() != nullptr
                                        ? parseIntArrayAttr<int64_t>(subViewOp.static_strides())
                                        : SmallVector<int64_t>(origType.getRank(), 1);

    if (subViewShape.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Tile shape '{0}' doesn't match MemRef rank '{1}'", subViewShape, origType.getRank());
    }
    if (subViewOffsets.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Tile offsets '{0}' doesn't match MemRef rank '{1}'", subViewOffsets, origType.getRank());
    }
    if (subViewStrides.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Tile strides '{0}' doesn't match MemRef rank '{1}'", subViewStrides, origType.getRank());
    }

    const auto subViewType =
            origType.extractViewTile(ShapeRef(subViewOffsets), ShapeRef(subViewShape), ShapeRef(subViewStrides));
    inferredTypes.push_back(subViewType);

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IERT::SubViewOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (source().getType() == result().getType()) {
        return source();
    }

    if (const auto origContent = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        const auto offset = Shape(parseIntArrayAttr<int64_t>(static_offsets()));
        const auto shape = Shape(parseIntArrayAttr<int64_t>(static_sizes()));
        return origContent.subview(offset, shape);
    }

    return nullptr;
}

//
// ComposeSubView
//

namespace {

class ComposeSubView final : public mlir::OpRewritePattern<IERT::SubViewOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(IERT::SubViewOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ComposeSubView::matchAndRewrite(IERT::SubViewOp origOp, mlir::PatternRewriter& rewriter) const {
    auto producerSubViewOp = origOp.source().getDefiningOp<IERT::SubViewOp>();
    if (producerSubViewOp == nullptr) {
        return mlir::failure();
    }

    if (origOp.static_strides().hasValue() || producerSubViewOp.static_strides().hasValue()) {
        return mlir::failure();
    }

    auto finalOffsets = parseIntArrayAttr<int64_t>(producerSubViewOp.static_offsets());
    const auto secondOffsets = parseIntArrayAttr<int64_t>(origOp.static_offsets());
    for (auto i : irange(finalOffsets.size())) {
        finalOffsets[i] += secondOffsets[i];
    }

    const auto finalOffsetsAttr = getIntArrayAttr(getContext(), finalOffsets);
    const auto finalShapeAttr = origOp.static_sizes();
    rewriter.replaceOpWithNewOp<IERT::SubViewOp>(origOp, producerSubViewOp.source(), finalOffsetsAttr, finalShapeAttr);

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IERT::SubViewOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<ComposeSubView>(ctx);
}
