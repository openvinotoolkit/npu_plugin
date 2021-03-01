//
// Copyright 2021 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

using namespace vpux;

void vpux::VPUIP::NegativeUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                       mlir::Value output) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::NegativeUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto negative = MVCNN::CreatePowerParams(writer, 0.0f, -1.0f, 1.0f);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_PowerParams);
    builder.add_nested_params(negative.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}
