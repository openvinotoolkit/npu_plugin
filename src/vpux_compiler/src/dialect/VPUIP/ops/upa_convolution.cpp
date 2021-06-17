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

#include <vpux/compiler/utils/extentions.hpp>
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/blob_reader.hpp"

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::verifyOp(ConvolutionUPAOp op) {
    if (verifySameInOutSpecificDimsOrder(op, {DimsOrder::NCHW}).failed()) {
        return mlir::failure();
    }

    const auto filterLayout = DimsOrder::fromValue(op.filter());
    if (filterLayout != DimsOrder::NCHW) {
        return errorAt(op, "filter layout must be NCHW, got {0}", filterLayout);
    }

    return mlir::success();
}

void vpux::VPUIP::ConvolutionUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                          mlir::Value filter, mlir::Value bias, mlir::Value output,
                                          mlir::ArrayAttr strides, mlir::ArrayAttr dilations, mlir::ArrayAttr padsBegin,
                                          mlir::ArrayAttr padsEnd, uint32_t groups) {
    build(builder, state, input, filter, bias, output, mlir::ValueRange{}, mlir::ValueRange{}, strides, dilations,
          padsBegin, padsEnd, groups, nullptr, false);
}

mlir::LogicalResult vpux::VPUIP::ConvolutionUPAOp::isSupportedLayout(mlir::Operation* op, vpux::DataOrderInfo& info) {
    VPUX_THROW_UNLESS(mlir::isa<IERT::GroupConvolutionOp>(op) || mlir::isa<IERT::ConvolutionOp>(op),
                      "Operation {0} is not Convolution like", op->getName());

    if (isSupportedLayoutSameInOutSpecificDimsOrder(op, info, {DimsOrder::NCHW}).failed()) {
        // filter layout
        info.setInput(1, DimsOrder::NCHW);
        return mlir::failure();
    }

    // check filter layout
    if (!info.hasInput(1) || info.getInput(1) != DimsOrder::NCHW) {
        fillDataInfo(info, 2, 1, DimsOrder::NCHW);
        return mlir::failure();
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ConvolutionUPAOp::serialize(VPUIP::BlobWriter& writer) {
    static const auto dY = Dim(2);
    static const auto dX = Dim(3);

    const auto strides = VPUIP::BlobWriter::createOrder3(this->strides());
    const auto dilations = VPUIP::BlobWriter::createOrder3(this->dilations());
    const auto padsBegin = VPUIP::BlobWriter::createOrder3(this->padsBegin());
    const auto padsEnd = VPUIP::BlobWriter::createOrder3(this->padsEnd());

    const auto filterShape = getShape(filter());
    const auto kernel =
            MVCNN::order3(checked_cast<uint8_t>(filterShape[dX]), checked_cast<uint8_t>(filterShape[dY]), 0);

    MVCNN::ConvolutionParamsBuilder builder(writer);
    builder.add_kernel(&kernel);
    builder.add_strides(&strides);
    builder.add_dilations(&dilations);
    builder.add_pads_begin(&padsBegin);
    builder.add_pads_end(&padsEnd);
    builder.add_group(checked_cast<int32_t>(groups()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ConvolutionParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseConvolution(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                           ArrayRef<mlir::Value> outputs,
                                                           const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 3, "UPAConvolution supports only 3 inputs, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAConvolution supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_ConvolutionParams();
    const auto strides = parseOrder3(params->strides());
    const auto dilations = parseOrder3(params->dilations());
    const auto padsBegin = parseOrder3(params->pads_begin());
    const auto padsEnd = parseOrder3(params->pads_end());
    return builder.create<VPUIP::ConvolutionUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], inputs[2],
                                                   outputs[0], strides, dilations, padsBegin, padsEnd, params->group());
}
