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

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

void vpux::VPUIP::EltwiseUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input1,
                                      mlir::Value input2, mlir::Value output, VPUIP::EltwiseLayerTypeAttr type) {
    build(builder, state, input1, input2, output, mlir::ValueRange{}, mlir::ValueRange{}, type, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::EltwiseUPAOp::serialize(VPUIP::BlobWriter& writer) {
    VPUIP::BlobWriter::String type;
    switch (this->type()) {
    case VPUIP::EltwiseLayerType::ADD:
        type = writer.createString("sum");
        break;
    case VPUIP::EltwiseLayerType::MULTIPLY:
        type = writer.createString("prod");
        break;
    case VPUIP::EltwiseLayerType::DIVIDE:
        type = writer.createString("div");
        break;
    case VPUIP::EltwiseLayerType::SQUARED_DIFF:
        type = writer.createString("sqdiff");
        break;
    case VPUIP::EltwiseLayerType::POWER:
        type = writer.createString("pow");
        break;
    case VPUIP::EltwiseLayerType::FLOOR_MOD:
        type = writer.createString("floormod");
        break;
    case VPUIP::EltwiseLayerType::MIN:
        type = writer.createString("min");
        break;
    case VPUIP::EltwiseLayerType::MAX:
        type = writer.createString("max");
        break;
    default:
        VPUX_THROW("Unsupported EltwiseLayerType {0}", this->type());
    }

    MVCNN::EltwiseParamsBuilder builder(writer);
    builder.add_operation(type);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_EltwiseParams});
}
