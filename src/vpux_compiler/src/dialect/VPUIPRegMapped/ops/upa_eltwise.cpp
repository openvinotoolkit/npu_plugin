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

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

void vpux::VPUIPRegMapped::EltwiseUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                                               mlir::Value input1, mlir::Value input2, mlir::Value output,
                                               VPUIPRegMapped::EltwiseLayerTypeAttr type) {
    build(builder, state, input1, input2, output, mlir::ValueRange{}, mlir::ValueRange{}, type, nullptr, nullptr);
}

// VPUIPRegMapped::BlobWriter::SpecificTask vpux::VPUIPRegMapped::EltwiseUPAOp::serialize(VPUIPRegMapped::BlobWriter&
// writer) {
void vpux::VPUIPRegMapped::EltwiseUPAOp::serialize(std::vector<char>& buffer) {
    /*
    VPUIPRegMapped::BlobWriter::String type;
    switch (this->type()) {
    case VPUIPRegMapped::EltwiseLayerType::ADD:
        type = writer.createString("sum");
        break;
    case VPUIPRegMapped::EltwiseLayerType::MULTIPLY:
        type = writer.createString("prod");
        break;
    case VPUIPRegMapped::EltwiseLayerType::DIVIDE:
        type = writer.createString("div");
        break;
    case VPUIPRegMapped::EltwiseLayerType::SQUARED_DIFF:
        type = writer.createString("sqdiff");
        break;
    case VPUIPRegMapped::EltwiseLayerType::POWER:
        type = writer.createString("pow");
        break;
    case VPUIPRegMapped::EltwiseLayerType::FLOOR_MOD:
        type = writer.createString("floormod");
        break;
    case VPUIPRegMapped::EltwiseLayerType::MIN:
        type = writer.createString("min");
        break;
    case VPUIPRegMapped::EltwiseLayerType::MAX:
        type = writer.createString("max");
        break;
    default:
        VPUX_THROW("Unsupported EltwiseLayerType {0}", this->type());
    }

    MVCNN::EltwiseParamsBuilder builder(writer);
    builder.add_operation(type);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_EltwiseParams});
    */

    (void)buffer;
}

/*
mlir::Operation* vpux::VPUIPRegMapped::BlobReader::parseEltwise(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                                ArrayRef<mlir::Value> outputs,
                                                                const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() >= 1 && inputs.size() <= 2, "UPAEltwise supports 1 or 2 inputs, got {0}",
                      inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAEltwise supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_EltwiseParams();
    const auto strType = params->operation()->str();
    EltwiseLayerType type;
    if (strType == "sum") {
        type = EltwiseLayerType::ADD;
    } else if (strType == "prod") {
        type = EltwiseLayerType::MULTIPLY;
    } else if (strType == "div") {
        type = EltwiseLayerType::DIVIDE;
    } else if (strType == "sqdiff") {
        type = EltwiseLayerType::SQUARED_DIFF;
    } else if (strType == "pow") {
        type = EltwiseLayerType::POWER;
    } else if (strType == "floormod") {
        type = EltwiseLayerType::FLOOR_MOD;
    } else if (strType == "min") {
        type = EltwiseLayerType::MIN;
    } else if (strType == "max") {
        type = EltwiseLayerType::MAX;
    } else {
        VPUX_THROW("Unsupported EltwiseLayerType {0}", strType);
    }

    return builder.create<VPUIPRegMapped::EltwiseUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], outputs[0],
                                                        EltwiseLayerTypeAttr::get(_ctx, type));
}
*/