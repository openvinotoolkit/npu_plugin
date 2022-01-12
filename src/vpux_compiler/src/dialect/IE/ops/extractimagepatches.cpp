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
#include "vpux/compiler/core/attributes/shape.hpp

using namespace vpux;

//TODO

mlir::LogicalResult vpux::IE::ExtractImagePatchesOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ExtractImagePatchesOpAdaptor extractImagePatches(operands, attrs);
    if (mlir::failed(extractImagePatches.verify(loc))) {
        return mlir::failure();
    }
        //paddingType ??
//      const auto sizes = parseIntArrayAttr<int32_t>(extractImagePatches.sizes());      //uint32_t ???
//      const auto strides = parseIntArrayAttr<int32_t>(extractImagePatches.strides());
//      const auto rates = parseIntArrayAttr<int32_t>(extractImagePatches.rates());
//      const auto paddingType = extractImagePatches.paddingType().getValue();

//    //data the 4-D tensor of type T with shape [batch, depth, in_rows, in_cols].

//    const auto inType = extractImagePatches.input().getType().cast<mlir::ShapedType>();
//    const auto inShape = inType.getShape();

//    if (inShape.size() != 4) {
//        return errorAt(loc, "Dimension of the tensor with shapes - input should be 4. Got {0} D tensor",
//                       inShape.size());
//    }

//    // sizes, strides, rates  > 0 ( = non-negative integer number) !?? - verify

//    SmallVector<int64_t> output_shape;
//    output_shape.push_back(inType.getShape()[0]); // ?? - batch
//    output_shape.push_back(inType.getShape()[1] * ... ); // ?? - size[0] * size[1] * depth
//    output_shape.push_back(..); // ?? -  out_rows
//    output_shape.push_back(..); // ?? - out_cols

//    inferredReturnShapes.emplace_back(output_shape, inType.getElementType());

    return mlir::success();
}
