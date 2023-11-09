//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::FakeQuantizeOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::FakeQuantizeOpAdaptor quantize(operands, attrs);
    if (mlir::failed(quantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = quantize.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputLowType = quantize.input_low().getType().cast<vpux::NDTypeInterface>();
    const auto inputHighType = quantize.input_high().getType().cast<vpux::NDTypeInterface>();
    const auto outputLowType = quantize.output_low().getType().cast<vpux::NDTypeInterface>();
    const auto outputHighType = quantize.output_high().getType().cast<vpux::NDTypeInterface>();
    const auto autob = quantize.auto_broadcast();

    const auto outShapeOrResult = IE::broadcastEltwiseShape(
            {inputType.getShape().raw(), inputLowType.getShape().raw(), inputHighType.getShape().raw(),
             outputLowType.getShape().raw(), outputHighType.getShape().raw()},
            autob, loc);

    if (mlir::succeeded(outShapeOrResult)) {
        const auto outType = inputType.changeShape(Shape(outShapeOrResult.value()));
        inferredReturnTypes.push_back(outType);
    }

    return outShapeOrResult;
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::FakeQuantizeOp::serialize(EMU::BlobWriter& writer) {
    const auto getRawFP16 = [](const float16& val) {
        return val.to_bits();
    };

    const auto getVecFP16 = [&](Const::ContentAttr attr) {
        const auto attrContent = attr.fold();
        return writer.createVector(attrContent.getValues<float16>() | transformed(getRawFP16));
    };

    auto inLowConst = input_low().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = input_high().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = output_low().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = output_high().getDefiningOp<Const::DeclareOp>();

    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        VPUX_THROW("Got non constant parameters for FakeQuantize");
    }

    const auto input_low = getVecFP16(inLowConst.getContentAttr());
    const auto input_high = getVecFP16(inHighConst.getContentAttr());
    const auto output_low = getVecFP16(outLowConst.getContentAttr());
    const auto output_high = getVecFP16(outHighConst.getContentAttr());

    MVCNN::FakeQuantizeParamsBuilder builder(writer);
    builder.add_levels(checked_cast<uint32_t>(levels()));
    builder.add_input_low(input_low);
    builder.add_input_high(input_high);
    builder.add_output_low(output_low);
    builder.add_output_high(output_high);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_FakeQuantizeParams});
}
