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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/blob_reader.hpp"

using namespace vpux;

void vpux::VPUIP::GRNUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                  mlir::Value output, mlir::FloatAttr bias) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, bias, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::GRNUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::GRNParamsBuilder builder(writer);
    builder.add_bias(static_cast<float>(bias().convertToDouble()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_GRNParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseGRN(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                   ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPAGRN supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAGRN supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_GRNParams();
    const auto bias = getFPAttr(_ctx, params->bias());

    return builder.create<VPUIP::GRNUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0], bias);
}
