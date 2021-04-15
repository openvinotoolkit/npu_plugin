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
