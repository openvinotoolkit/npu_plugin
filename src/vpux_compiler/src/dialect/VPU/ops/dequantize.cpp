//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::DequantizeOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              mlir::Optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::DequantizeOpAdaptor dequantize(operands, attrs);
    if (mlir::failed(dequantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = dequantize.input().getType().cast<vpux::NDTypeInterface>();
    const auto dstElemType = dequantize.dstElemType();

    const auto outType = inType.changeElemType(dstElemType);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

void vpux::VPU::DequantizeOp::inferLayoutInfo(mlir::Operation* origOp, IE::LayerLayoutInfo& info) {
    const auto inType = origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();

    const auto qType = inType.cast<mlir::quant::QuantizedType>();

    if (qType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto numDims = info.getInput(0).numDims();
        if (numDims == 3) {
            info.fill(DimsOrder::HWC);
        } else if (numDims == 4) {
            info.fill(DimsOrder::NHWC);
        } else {
            VPUX_THROW("Unsupported rank '{0}'", numDims);
        }
    } else {
        VPU::inferLayoutInfoSameInOutSpecificDimsOrder(
                info, {DimsOrder::CHW, DimsOrder::HWC, DimsOrder::NCHW, DimsOrder::NHWC});
    }
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::DequantizeOp::serialize(EMU::BlobWriter& writer) {
    auto scalesAndZeroPoints = vpux::serializeScalesAndZeroPointsEmu(input(), output(), writer);

    MVCNN::QuantizeParamsBuilder builder(writer);
    builder.add_scale(scalesAndZeroPoints.first);
    builder.add_zero(scalesAndZeroPoints.second);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_QuantizeParams});
}
