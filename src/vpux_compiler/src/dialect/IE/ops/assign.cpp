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

using namespace vpux;

mlir::LogicalResult vpux::IE::AssignOp::inferReturnTypeComponents(::mlir::MLIRContext* ctx, ::mlir::Optional<::mlir::Location> optLoc, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange /*regions*/, ::mlir::SmallVectorImpl<::mlir::ShapedTypeComponents>& inferredReturnShapes)
{
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::AssignOpAdaptor assign(operands, attrs);
    if (mlir::failed(assign.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = assign.input().getType().cast<mlir::ShapedType>();
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());
    // inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    // TODO: add implementation here

    return mlir::success();
}
