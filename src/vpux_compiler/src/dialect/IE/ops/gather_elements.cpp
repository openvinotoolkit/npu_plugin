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
#include "vpux/compiler/dialect/IE/utils/to_ngraph.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::GatherElementsOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::GatherElementsOpAdaptor gatherElements(operands, attrs);
    if (mlir::failed(gatherElements.verify(loc))) {
        return mlir::failure();
    }

    const auto inIndicesType = gatherElements.indices().getType().cast<mlir::ShapedType>();
    const auto inInputType = gatherElements.input().getType().cast<mlir::ShapedType>();

    inferredReturnShapes.emplace_back(inIndicesType.getShape(), inInputType.getElementType());
    return mlir::success();
}

std::unique_ptr<ngraph::Node> vpux::IE::GatherElementsOp::toNgraph(ngraph::OutputVector &outputs)
{
    return std::make_unique<opset_latest::GatherElements>(outputs.at(0), outputs.at(1), axis());
}
