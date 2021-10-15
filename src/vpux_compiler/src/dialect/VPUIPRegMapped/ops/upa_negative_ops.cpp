//
// Copyright 2021 Intel Corporation.
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

#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"

// Alex: #include "vpux/compiler/dialect/VPUIPRegMapped/blob_reader.hpp"

using namespace vpux;

void vpux::VPUIPRegMapped::NegativeUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                                                mlir::Value input, mlir::Value output) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, nullptr);
}

// VPUIPRegMapped::BlobWriter::SpecificTask vpux::VPUIPRegMapped::NegativeUPAOp::serialize(VPUIPRegMapped::BlobWriter&
// writer) {
void vpux::VPUIPRegMapped::NegativeUPAOp::serialize(std::vector<char>& buffer) {
    /*
    const auto negative = MVCNN::CreatePowerParams(writer, 0.0f, -1.0f, 1.0f);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_PowerParams);
    builder.add_nested_params(negative.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
    */

    (void)buffer;
}

/*
// Alex
mlir::Operation* vpux::VPUIPRegMapped::BlobReader::parseNegative(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                                 ArrayRef<mlir::Value> outputs,
                                                                 const MVCNN::UPALayerTask*) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPANegative supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPANegative supports only 1 output, got {0}", outputs.size());
    return builder.create<VPUIPRegMapped::NegativeUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
}
*/