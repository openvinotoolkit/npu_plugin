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
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::MultiplyOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::MultiplyOpAdaptor multiply(operands, attrs);
    if (mlir::failed(multiply.verify(loc))) {
        return ::mlir::failure();
    }

    auto in1Type = multiply.input1().getType().cast<mlir::RankedTensorType>();
    auto in2Type = multiply.input2().getType().cast<mlir::RankedTensorType>();

    auto outShapeOrResult = IE::broadcastEltwiseShape(in1Type.getShape(), in2Type.getShape(),
                                                      multiply.auto_broadcast().getValue(), loc);
    mlir::LogicalResult result = outShapeOrResult;
    if (result.value == mlir::LogicalResult::Success) {
        inferredReturnShapes.emplace_back(outShapeOrResult.getValue(), in1Type.getElementType());
    }
    return mlir::success();
}
