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
//#include "vpux/compiler/dialect/VPU/attributes.hpp"

//#include "vpux/compiler/core/attributes/shape.hpp"
//#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

//#include "vpux/utils/core/checked_cast.hpp"
//#include "vpux/utils/core/error.hpp"

//#include <mlir/IR/BlockAndValueMapping.h>
//#include <mlir/IR/PatternMatch.h>

//#include <ngraph/coordinate.hpp>
//#include <ngraph/validation_util.hpp>

using namespace vpux;

//TODO

//mlir::LogicalResult vpux::IE::ExtractImagePatchesOp::inferReturnTypeComponents(
//        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
//        mlir::DictionaryAttr attrs, mlir::RegionRange,
//        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
//    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

//    IE::ExtractImagePatchesOpAdaptor extractImagePatches(operands, attrs);
//    if (mlir::failed(extractImagePatches.verify(loc))) {
//        return mlir::failure();
//    }

//    // sizes, strides, rates, paddingType

//    const auto <attribute_names> = extractImagePatches.<attribute_names>.get.. ;

//    //data the 4-D tensor of type T with shape [batch, depth, in_rows, in_cols].

//    const auto inShapeFeatureMap = inTypeFeatureMap.getShape();
//    const auto inType = extractImagePatches.input().getType().cast<mlir::ShapedType>();

//    if (inShapeFeatureMap.size() != 4) {
//        return errorAt(loc, "Dimension of the tensor with shapes - input should be 4. Got {0} D tensor",
//                       inShape.size());
//    }

//    // sizes, strides, rates  > 0 ( = non-negative integer number)

//    SmallVector<int64_t> output_shape;
//    output_shape.push_back(inShapeFeatureMap[1]);

//    inferredReturnShapes.emplace_back(output_shape, inType.getElementType());
//    return mlir::success();
//}
