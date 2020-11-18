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

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::verifySoftMaxLayer(mlir::Operation* op) {
    auto layer = mlir::cast<SoftMaxLayerInterface>(op);

    const auto inRank = layer.getInputShape().size();
    const auto axisInd = layer.getAxisDim().ind();

    if (axisInd < 0 || checked_cast<size_t>(axisInd) >= inRank) {
        return printTo(
                op->emitError(),
                "'{0}' axis index '{1}' is out of input tensor rank '{2}'",
                op->getName().getStringRef(),
                axisInd,
                inRank);
    }

    return mlir::success();
}

ShapeRef vpux::IE::SoftMaxOp::getInputShape() {
    return ShapeRef(
            input().getType().cast<mlir::RankedTensorType>().getShape());
}

Dim vpux::IE::SoftMaxOp::getAxisDim() {
    return Dim(axisInd());
}

mlir::LogicalResult vpux::IE::SoftMaxOp::inferReturnTypes(
        mlir::MLIRContext* ctx,
        Optional<mlir::Location> loc,
        mlir::ValueRange operands,
        mlir::DictionaryAttr attributes,
        mlir::RegionRange regions,
        SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    return mlir::detail::inferReturnTensorTypes(
            SoftMaxOp::inferReturnTypeComponents,
            ctx,
            loc,
            operands,
            attributes,
            regions,
            inferredReturnTypes);
}

mlir::LogicalResult vpux::IE::SoftMaxOp::inferReturnTypeComponents(
        mlir::MLIRContext*,
        Optional<mlir::Location>,
        mlir::ValueRange operands,
        mlir::DictionaryAttr,
        mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    VPUX_THROW_UNLESS(operands.size() == 1,
                      "Got wrong number of operands : {0}",
                      operands.size());

    const auto inType = operands[0].getType().cast<mlir::RankedTensorType>();

    inferredReturnShapes.emplace_back(inType.getShape(),
                                      inType.getElementType());

    return mlir::success();
}

namespace IE_SoftMax {
namespace {

#include <vpux/compiler/dialect/IE/rewriters/generated/softmax.hpp.inc>

}  // namespace
}  // namespace IE_SoftMax

void vpux::IE::SoftMaxOp::getCanonicalizationPatterns(
        mlir::OwningRewritePatternList& patterns,
        mlir::MLIRContext* context) {
    IE_SoftMax::populateWithGenerated(context, patterns);
}
