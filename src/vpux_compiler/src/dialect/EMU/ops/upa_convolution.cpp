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

#include "vpux/compiler/dialect/EMU/ops.hpp"

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::EMU::verifyOp(ConvolutionUPAOp op) {
    // There are two uPA tasks which perform convolution: ConvUPA and SWConvUPA.
    // SWConvUPA supports parallel execution on multiple uPA units.
    // However, it does not have group support, so group convolutions go to ConvUPA.
    // ConvUPA expects NCHW order, SWConvUPA expects YXOI.
    const auto expectedFilterLayout = (op.groups() > 1) ? DimsOrder::OIYX : DimsOrder::YXOI;
    const auto filterLayout = DimsOrder::fromValue(op.filter());

    if (filterLayout != expectedFilterLayout) {
        return errorAt(op, "filter layout must be {0}, got {1}", expectedFilterLayout, filterLayout);
    }

    return mlir::success();
}

EMU::BlobWriter::SpecificTask vpux::EMU::ConvolutionUPAOp::serialize(EMU::BlobWriter& writer) {
    static const auto dY = Dim(2);
    static const auto dX = Dim(3);

    const auto strides = EMU::BlobWriter::createOrder3(this->strides());
    const auto dilations = EMU::BlobWriter::createOrder3(this->dilations());
    const auto padsBegin = EMU::BlobWriter::createOrder3(this->padsBegin());
    const auto padsEnd = EMU::BlobWriter::createOrder3(this->padsEnd());

    const auto filterShape = getShape(filter());
    const auto kernel =
            MVCNN::order3(checked_cast<uint8_t>(filterShape[dX]), checked_cast<uint8_t>(filterShape[dY]), 0);

    if (groups() > 1) {
        MVCNN::ConvolutionParamsBuilder builder(writer);
        builder.add_kernel(&kernel);
        builder.add_strides(&strides);
        builder.add_dilations(&dilations);
        builder.add_pads_begin(&padsBegin);
        builder.add_pads_end(&padsEnd);
        builder.add_group(checked_cast<int32_t>(groups()));
        const auto paramsOff = builder.Finish();
        return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ConvolutionParams});
    } else {
        MVCNN::SWConvolutionParamsBuilder builder(writer);
        builder.add_kernel(&kernel);
        builder.add_strides(&strides);
        builder.add_dilations(&dilations);
        builder.add_pads_begin(&padsBegin);
        builder.add_pads_end(&padsEnd);
        builder.add_group(checked_cast<int32_t>(groups()));
        const auto paramsOff = builder.Finish();
        return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_SWConvolutionParams});
    }
}
