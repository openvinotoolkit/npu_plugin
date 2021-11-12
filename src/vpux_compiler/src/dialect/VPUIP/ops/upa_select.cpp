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

#include "vpux/compiler/dialect/VPUIP/blob_reader.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

void vpux::VPUIP::SelectUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input1, mlir::Value input2,
                                     mlir::Value input3, mlir::Value output, VPUIP::EltwiseLayerTypeAttr type) {
    build(builder, state, input1, input2, input3, output, mlir::ValueRange{}, mlir::ValueRange{}, type, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SelectUPAOp::serialize(VPUIP::BlobWriter& writer) {
    VPUIP::BlobWriter::String type;
    switch (this->type()) {
    case VPUIP::EltwiseLayerType::SELECT:
        type = writer.createString("select");
        break;
    default:
        VPUX_THROW("Unsupported EltwiseLayerType {0}", this->type());
    }

    MVCNN::EltwiseParamsBuilder builder(writer);
    builder.add_operation(type);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_EltwiseParams});
}