//
// Copyright 2020 Intel Corporation.
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

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::FakeQuantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
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

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::IE::FakeQuantizeOp::serialize(EMU::BlobWriter& writer) {
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

    const auto input_low = getVecFP16(inLowConst.contentAttr());
    const auto input_high = getVecFP16(inHighConst.contentAttr());
    const auto output_low = getVecFP16(outLowConst.contentAttr());
    const auto output_high = getVecFP16(outHighConst.contentAttr());

    MVCNN::FakeQuantizeParamsBuilder builder(writer);
    builder.add_levels(checked_cast<uint32_t>(levels()));
    builder.add_input_low(input_low);
    builder.add_input_high(input_high);
    builder.add_output_low(output_low);
    builder.add_output_high(output_high);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_FakeQuantizeParams});
}
