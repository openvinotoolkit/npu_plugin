//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::UpsamplingOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              mlir::Optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::UpsamplingOpAdaptor upsampling(operands, attrs);
    if (mlir::failed(upsampling.verify(loc))) {
        return mlir::failure();
    }

    auto padLVector = parseIntArrayAttr<int32_t>(upsampling.pad_l());
    auto padRVector = parseIntArrayAttr<int32_t>(upsampling.pad_r());
    auto upsamplingFactorVector = parseIntArrayAttr<int32_t>(upsampling.upsampling_factor());

    const auto inType = upsampling.input().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape().raw();

    VPUX_THROW_UNLESS(inShape.size() == 4, "Upsampling supports only 4D input tensor");
    VPUX_THROW_UNLESS(padLVector.size() == 3, "Upsampling supports pads only for 3 axes");
    VPUX_THROW_UNLESS(padRVector.size() == 3, "Upsampling supports pads only for 3 axes");
    VPUX_THROW_UNLESS(upsamplingFactorVector.size() == 3, "Upsampling supports factors only for 3 axes");

    SmallVector<int64_t> outputShape{
            inShape[0],
            inShape[1] + (inShape[1] - 1) * (upsamplingFactorVector[2] - 1) + padLVector[2] + padRVector[2],
            inShape[2] + (inShape[2] - 1) * (upsamplingFactorVector[1] - 1) + padLVector[1] + padRVector[1],
            inShape[3] + (inShape[3] - 1) * (upsamplingFactorVector[0] - 1) + padLVector[0] + padRVector[0],
    };

    const auto outType = inType.changeShape(Shape(outputShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::UpsamplingOp::serialize(EMU::BlobWriter& writer) {
    SmallVector<int32_t> pad_x = {checked_cast<int32_t>(pad_l()[0].cast<mlir::IntegerAttr>().getInt()),
                                  checked_cast<int32_t>(pad_r()[0].cast<mlir::IntegerAttr>().getInt())};
    auto pad_x_vector = writer.createVector(pad_x);

    SmallVector<int32_t> pad_y = {checked_cast<int32_t>(pad_l()[1].cast<mlir::IntegerAttr>().getInt()),
                                  checked_cast<int32_t>(pad_r()[1].cast<mlir::IntegerAttr>().getInt())};
    auto pad_y_vector = writer.createVector(pad_y);

    SmallVector<int32_t> pad_z = {checked_cast<int32_t>(pad_l()[2].cast<mlir::IntegerAttr>().getInt()),
                                  checked_cast<int32_t>(pad_r()[2].cast<mlir::IntegerAttr>().getInt())};
    auto pad_z_vector = writer.createVector(pad_z);

    MVCNN::UpsamplingParamsBuilder builder(writer);
    builder.add_upsampling_factor_x(checked_cast<int32_t>(upsampling_factor()[0].cast<mlir::IntegerAttr>().getInt()));
    builder.add_upsampling_factor_y(checked_cast<int32_t>(upsampling_factor()[1].cast<mlir::IntegerAttr>().getInt()));
    builder.add_upsampling_factor_z(checked_cast<int32_t>(upsampling_factor()[2].cast<mlir::IntegerAttr>().getInt()));
    builder.add_pad_x(pad_x_vector);
    builder.add_pad_y(pad_y_vector);
    builder.add_pad_z(pad_z_vector);

    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_UpsamplingParams});
}
