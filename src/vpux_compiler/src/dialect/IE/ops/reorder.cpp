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

using namespace vpux;

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::ReorderOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ReorderOpAdaptor reorder(operands, attrs);
    if (mlir::failed(reorder.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = reorder.input().getType().cast<mlir::ShapedType>();
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType(),
                                      IE::getTensorAttr(reorder.dstOrder()));

    return mlir::success();
}

//
// Canonicalization
//

namespace {

#include <vpux/compiler/dialect/IE/rewriters/generated/reorder.hpp.inc>

}  // namespace

void vpux::IE::ReorderOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext*) {
    populateWithGenerated(patterns);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ReorderOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (input().getType() == output().getType()) {
        return input();
    }

    if (const auto cst = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return cst.reorder(DimsOrder::fromValue(output()));
    }

    return nullptr;
}
