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

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::ConvertOp::inferReturnTypes(
        mlir::MLIRContext* ctx,
        Optional<mlir::Location> loc,
        mlir::ValueRange operands,
        mlir::DictionaryAttr attributes,
        mlir::RegionRange regions,
        SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    return mlir::detail::inferReturnTensorTypes(
            ConvertOp::inferReturnTypeComponents,
            ctx,
            loc,
            operands,
            attributes,
            regions,
            inferredReturnTypes);
}

mlir::LogicalResult vpux::IE::ConvertOp::inferReturnTypeComponents(
        mlir::MLIRContext*,
        Optional<mlir::Location>,
        mlir::ValueRange operands,
        mlir::DictionaryAttr attrs,
        mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    VPUX_THROW_UNLESS(operands.size() == 1,
                      "Got wrong number of operands : {0}",
                      operands.size());

    const auto dstTypeAttr =
            attrs.get("dstType").dyn_cast_or_null<mlir::TypeAttr>();
    VPUX_THROW_UNLESS(dstTypeAttr != nullptr, "Missing dstType attribute");

    const auto inType = operands[0].getType().cast<mlir::RankedTensorType>();

    inferredReturnShapes.emplace_back(inType.getShape(),
                                      dstTypeAttr.getValue());

    return mlir::success();
}

namespace IE_Convert {
namespace {

#include <vpux/compiler/dialect/IE/rewriters/generated/convert.hpp.inc>

}  // namespace
}  // namespace IE_Convert

void vpux::IE::ConvertOp::getCanonicalizationPatterns(
        mlir::OwningRewritePatternList& patterns,
        mlir::MLIRContext* context) {
    IE_Convert::populateWithGenerated(context, patterns);
}

mlir::OpFoldResult vpux::IE::ConvertOp::fold(ArrayRef<mlir::Attribute>) {
    const auto inType = input().getType().cast<mlir::RankedTensorType>();
    const auto outType = output().getType().cast<mlir::RankedTensorType>();

    if (inType.getElementType() == outType.getElementType()) {
        return input();
    }

    return nullptr;
}
