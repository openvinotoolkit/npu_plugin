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

#include "vpux/utils/core/checked_cast.hpp"

#include <legacy/ngraph_ops/lrn_ie.hpp>

using namespace vpux;

mlir::LogicalResult vpux::IE::LRN_IEOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::LRN_IEOpAdaptor lrn_ie(operands, attrs);
    if (mlir::failed(lrn_ie.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = lrn_ie.input().getType().cast<mlir::ShapedType>();
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

std::shared_ptr<ngraph::Node> vpux::IE::LRN_IEOp::toNgraph(ngraph::OutputVector &outputs)
{
    const auto reg = exportLRN_IERegion(region());

    return std::make_shared<ngraph::op::LRN_IE>(outputs.at(0), alpha().convertToDouble(),
        beta().convertToDouble(), bias().convertToDouble(), size(), reg);
}
