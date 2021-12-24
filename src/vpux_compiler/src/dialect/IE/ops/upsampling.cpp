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

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::UpsamplingOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::UpsamplingOpAdaptor upsampling(operands, attrs);
    if (mlir::failed(upsampling.verify(loc))) {
        return mlir::failure();
    }

    auto pad_l_vector = parseIntArrayAttr<int32_t>(upsampling.pad_l());
    auto pad_r_vector = parseIntArrayAttr<int32_t>(upsampling.pad_r());
    auto upsampling_factor_vector = parseIntArrayAttr<int32_t>(upsampling.upsampling_factor());

    const auto inType = upsampling.input().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();

    SmallVector<int64_t> outputShape{
            inShape[0],
            inShape[1] + (inShape[1] - 1) * (upsampling_factor_vector[2] - 1) + pad_l_vector[2] + pad_r_vector[2],
            inShape[2] + (inShape[2] - 1) * (upsampling_factor_vector[1] - 1) + pad_l_vector[1] + pad_r_vector[1],
            inShape[3] + (inShape[3] - 1) * (upsampling_factor_vector[0] - 1) + pad_l_vector[0] + pad_r_vector[0],
    };

    inferredReturnShapes.emplace_back(outputShape, inType.getElementType());

    return mlir::success();
}
