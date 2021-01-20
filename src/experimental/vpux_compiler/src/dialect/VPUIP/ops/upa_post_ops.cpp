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

#include "vpux/compiler/core/attributes/stride_reqs.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

//
// verifyPostOp
//

mlir::LogicalResult vpux::VPUIP::verifyPostOp(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifyPostOp");

    auto layer = mlir::dyn_cast<LayerInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
    }

    auto inputs = layer.getInputs();
    auto outputs = layer.getOutputs();
    for (auto val : concat<mlir::Value>(inputs, outputs)) {
        const auto shape = getShape(val);
        const auto order = DimsOrder::fromValue(val);
        const auto elemSize = getElemTypeSize(val.getType());
        const auto strides = getStrides(val);
        const auto memShape = order->toMemoryOrder(shape);
        const auto memStrides = order->toMemoryOrder(strides);

        // TODO : can we fix that limitation?
        const auto strideReqs = StrideReqs::compact(shape.size()).remove(MemDim(1));

        if (!strideReqs.checkStrides(memStrides, elemSize, memShape)) {
            return errorAt(op, "Memory strides '{0}' do not match requirements '{1}'", memStrides, strideReqs);
        }
    }

    return mlir::success();
}

//
// ClampUPAOp
//

void vpux::VPUIP::ClampUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                    mlir::Value output, mlir::FloatAttr min, mlir::FloatAttr max) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, min, max, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ClampUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const float min = getMin();
    const float max = getMax();

    const auto clamp = MVCNN::CreateClampParams(writer, min, max);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_ClampParams);
    builder.add_nested_params(clamp.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// EluUPAOp
//

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

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// HSwishUPAOp
//

void vpux::VPUIP::HSwishUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     mlir::Value output) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::HSwishUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto hswish = MVCNN::CreateHSwishParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_HSwishParams);
    builder.add_nested_params(hswish.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// Tanh
//

void vpux::VPUIP::TanhUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                   mlir::Value output) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::TanhUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto tanh = MVCNN::CreateTanhParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_TanhParams);
    builder.add_nested_params(tanh.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// ReLUUPAOp
//

void vpux::VPUIP::ReLUUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                   mlir::Value output) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, false);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ReLUUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto relu = MVCNN::CreateReluParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_ReluParams);
    builder.add_nested_params(relu.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// SigmoidUPAOp
//

void vpux::VPUIP::SigmoidUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                      mlir::Value output) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, false);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SigmoidUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto sigmoid = MVCNN::CreateSigmoidParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_SigmoidParams);
    builder.add_nested_params(sigmoid.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// PRelu
//

void vpux::VPUIP::PReluUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                    mlir::Value negative_slope, mlir::Value output) {
    build(builder, state, input, negative_slope, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::PReluUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto prelu = MVCNN::CreatePReluParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_PReluParams);
    builder.add_nested_params(prelu.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// LeakyRelu
//

void vpux::VPUIP::LeakyReluUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                        mlir::Value output, mlir::FloatAttr negative_slope) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, negative_slope, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::LeakyReluUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const float negative_slope = getNegativeSlope();

    const auto leaky_relu = MVCNN::CreateLeakyReluParams(writer, negative_slope);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_LeakyReluParams);
    builder.add_nested_params(leaky_relu.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}
