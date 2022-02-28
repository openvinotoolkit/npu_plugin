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
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::RollOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::RollOpAdaptor roll(operands, attrs);
    if (mlir::failed(roll.verify(loc))) {
        return mlir::failure();
    }

    //TODO
    
    const auto inType = roll.data().getType().cast<mlir::ShapedType>();
    //const auto inShape = inType.getShape();
    
    // auto shiftType = IE::constInputToData(loc, roll.shift()).getValue();
    //  auto axesType = IE::constInputToData(loc, roll.axes()).getValue();

    //SmallVector<int64_t> outShape;
    
    inferredReturnShapes.emplace_back(inType.getElementType());

    return mlir::success();
}
