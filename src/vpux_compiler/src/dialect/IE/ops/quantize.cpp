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

#include <vpux/compiler/utils/quantization.hpp>
#include "vpux/compiler/dialect/IE/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::QuantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::QuantizeOpAdaptor quantize(operands, attrs);
    if (mlir::failed(quantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = quantize.input().getType().cast<mlir::ShapedType>();
    const auto dstElemType = quantize.dstElemType().getValue();

    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType);
    return mlir::success();
}

//
// fold
//

namespace {

mlir::quant::QuantizedType extractQuantizedType(mlir::Value operand) {
    const auto elemType = operand.getType().cast<mlir::ShapedType>().getElementType();
    const auto quantType = elemType.dyn_cast<mlir::quant::QuantizedType>();
    VPUX_THROW_UNLESS(quantType != nullptr, "Type must be quantized, but provided {0}", elemType);
    return quantType;
}

}  // namespace

mlir::OpFoldResult vpux::IE::QuantizeOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (const auto cst = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        const auto quantType = extractQuantizedType(output());
        const auto quantStorageType = normalizeQuantStorageType(quantType);
        return cst.convertElemType(quantStorageType).quantCast(quantType);
    }

    if (auto dequantize = input().getDefiningOp<IE::DequantizeOp>()) {
        if (dequantize.input().getType() == output().getType()) {
            return dequantize.input();
        }
    }

    return nullptr;
}


mlir::LogicalResult vpux::IE::DequantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::DequantizeOpAdaptor dequantize(operands, attrs);
    if (mlir::failed(dequantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = dequantize.input().getType().cast<mlir::ShapedType>();
    const auto dstElemType = dequantize.dstElemType().getValue();

    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType);
    return mlir::success();
}

namespace {

std::pair<EMU::BlobWriter::Vector<uint16_t>, EMU::BlobWriter::Vector<uint16_t>> serializeScalesAndZeroPoints(
        mlir::Value input, mlir::Value output, EMU::BlobWriter& writer) {
    const auto inType = input.getType().cast<mlir::RankedTensorType>().getElementType();
    const auto outType = output.getType().cast<mlir::RankedTensorType>().getElementType();

    const auto qType = inType.isa<mlir::quant::QuantizedType>() ? inType.cast<mlir::quant::QuantizedType>()
                                                                : outType.cast<mlir::quant::QuantizedType>();

    const auto getRawFP16 = [](auto val) {
        const auto valFP16 = float16(val);
        return valFP16.to_bits();
    };

    const auto getVecFP16 = [&](auto range) {
        return writer.createVector(range | transformed(getRawFP16));
    };

    SmallVector<double> scales;
    SmallVector<int64_t> zeroPoints;
    if (qType.isa<mlir::quant::UniformQuantizedType>()) {
        auto quantParams = qType.cast<mlir::quant::UniformQuantizedType>();
        scales = {quantParams.getScale()};
        zeroPoints = {quantParams.getZeroPoint()};
    } else if (qType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        auto quantParams = qType.cast<mlir::quant::UniformQuantizedPerAxisType>();
        scales = {quantParams.getScales().begin(), quantParams.getScales().end()};
        zeroPoints = {quantParams.getZeroPoints().begin(), quantParams.getZeroPoints().end()};
    } else {
        VPUX_THROW("Unsupported quantized type {0}", qType);
    }

    return {getVecFP16(scales), getVecFP16(zeroPoints)};
}

}  // namespace

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::IE::QuantizeOp::serialize(EMU::BlobWriter& writer) {
    auto scalesAndZeroPoints = serializeScalesAndZeroPoints(input(), output(), writer);

    MVCNN::QuantizeParamsBuilder builder(writer);
    builder.add_scale(scalesAndZeroPoints.first);
    builder.add_zero(scalesAndZeroPoints.second);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_QuantizeParams});
}

EMU::BlobWriter::SpecificTask vpux::IE::DequantizeOp::serialize(EMU::BlobWriter& writer) {
    auto scalesAndZeroPoints = serializeScalesAndZeroPoints(input(), output(), writer);

    MVCNN::QuantizeParamsBuilder builder(writer);
    builder.add_scale(scalesAndZeroPoints.first);
    builder.add_zero(scalesAndZeroPoints.second);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_QuantizeParams});
}
