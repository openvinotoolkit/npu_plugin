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

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::IE::CopyOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::CopyOpAdaptor copyOp(operands, attrs);
    if (mlir::failed(copyOp.verify(loc))) {
        return mlir::failure();
    }

    const auto ndInType = copyOp.input().getType().dyn_cast<vpux::NDTypeInterface>();
    if (ndInType == nullptr) {
        return errorAt(loc, "IE::CopyOp operand must have vpux::NDTypeInterface type");
    }

    const auto outMemSpace = copyOp.out_mem_space();
    const auto outType = ndInType.changeMemSpace(outMemSpace).cast<mlir::RankedTensorType>();

    inferredReturnShapes.emplace_back(outType.getShape(), outType.getElementType(), outType.getEncoding());
    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::CopyOp::fold(ArrayRef<mlir::Attribute>) {
    if (input().getType() == output().getType()) {
        return input();
    }

    return nullptr;
}

//
// FuseCopies
//

namespace {

class FuseCopies final : public mlir::OpRewritePattern<IE::CopyOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(IE::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseCopies::matchAndRewrite(IE::CopyOp origOp, mlir::PatternRewriter& rewriter) const {
    auto producerCopyOp = origOp.input().getDefiningOp<IE::CopyOp>();
    if (producerCopyOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::CopyOp>(origOp, producerCopyOp.input(), origOp.out_mem_spaceAttr());
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::CopyOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<FuseCopies>(ctx);
}
