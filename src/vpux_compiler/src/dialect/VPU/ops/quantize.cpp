//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::QuantizeOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                            mlir::Optional<mlir::Location> optLoc,
                                                            mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                            mlir::RegionRange /*regions*/,
                                                            mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::QuantizeOpAdaptor quantize(operands, attrs);
    if (mlir::failed(quantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = quantize.input().getType().cast<vpux::NDTypeInterface>();
    const auto dstElemType = quantize.dstElemType();

    const auto outType = inType.changeElemType(dstElemType);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

void vpux::VPU::QuantizeOp::inferLayoutInfo(mlir::Operation* origOp, IE::LayerLayoutInfo& info) {
    const auto outType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();

    const auto qType = outType.cast<mlir::quant::QuantizedType>();

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

EMU::BlobWriter::SpecificTask vpux::VPU::QuantizeOp::serialize(EMU::BlobWriter& writer) {
    auto scalesAndZeroPoints = vpux::serializeScalesAndZeroPointsEmu(input(), output(), writer);

    MVCNN::QuantizeParamsBuilder builder(writer);
    builder.add_scale(scalesAndZeroPoints.first);
    builder.add_zero(scalesAndZeroPoints.second);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_QuantizeParams});
}
