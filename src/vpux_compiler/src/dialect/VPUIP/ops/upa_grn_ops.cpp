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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

using namespace vpux;

void vpux::VPUIP::GRNUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                  mlir::Value output, mlir::FloatAttr bias) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, bias, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::GRNUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto biasVal = bias().convertToFloat();

    MVCNN::GRNParamsBuilder builder(writer);
    builder.add_bias(checked_cast<float>(biasVal));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_GRNParams});
}
