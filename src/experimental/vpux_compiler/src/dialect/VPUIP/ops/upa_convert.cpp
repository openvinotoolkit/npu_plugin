//
// Copyright 2020 Intel Corporation.
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

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

void vpux::VPUIP::ConvertUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                      mlir::Value output) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, nullptr, false, false,
          nullptr, nullptr, false);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ConvertUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto scale = scaleAttr() ? scaleAttr().getValueAsDouble() : 1.0;
    const auto bias = biasAttr() ? biasAttr().getValueAsDouble() : 0.0;
    const auto batchID = checked_cast<int32_t>(this->batchID().getValueOr(0));

    MVCNN::ConvertParamsBuilder builder(writer);
    builder.add_scale(checked_cast<float>(scale));
    builder.add_scale(checked_cast<float>(bias));
    builder.add_from_detection_output(fromDetectionOutput());
    builder.add_have_batch(haveBatch());
    builder.add_batch_id(batchID);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ConvertParams});
}
