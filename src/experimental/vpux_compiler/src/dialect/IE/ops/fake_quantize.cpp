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

mlir::LogicalResult vpux::IE::FakeQuantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::FakeQuantizeOpAdaptor quantize(operands, attrs);
    if (mlir::failed(quantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = quantize.input().getType().cast<mlir::ShapedType>();
    const auto inputLowType = quantize.input_low().getType().cast<mlir::ShapedType>();
    const auto inputHighType = quantize.input_high().getType().cast<mlir::ShapedType>();
    const auto outputLowType = quantize.output_low().getType().cast<mlir::ShapedType>();
    const auto outputHighType = quantize.output_high().getType().cast<mlir::ShapedType>();
    const auto autob = quantize.auto_broadcast().getValue();

    const auto outShapeOrResult =
            IE::broadcastEltwiseShape({inputType.getShape(), inputLowType.getShape(), inputHighType.getShape(),
                                       outputLowType.getShape(), outputHighType.getShape()},
                                      autob, loc);

    if (mlir::succeeded(outShapeOrResult)) {
        inferredReturnShapes.emplace_back(outShapeOrResult.getValue(), inputType.getElementType());
    }

    return outShapeOrResult;
}
