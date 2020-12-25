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

void vpux::VPUIP::EluUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                  mlir::Value output, mlir::FloatAttr x) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, x, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::EluUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const float x = getX();

    const auto elu = MVCNN::CreateEluParams(writer, x);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_EluParams);
    builder.add_nested_params(elu.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(getOperation(), {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams},
                                     maxShaves(), isTrailingSWLayer());
}
