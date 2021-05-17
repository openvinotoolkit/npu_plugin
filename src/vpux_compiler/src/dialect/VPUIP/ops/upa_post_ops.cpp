//
// Copyright 2020 Intel Corporation.
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

    for (auto& operand : layer.getOpOperands()) {
        const auto opVal = operand.get();
        // TODO : can we fix that limitation?
        const auto strideReqs =
                StrideReqs::compact(opVal.getType().cast<mlir::ShapedType>().getRank()).remove(MemDim(1));

        if (!strideReqs.checkStrides(opVal)) {
            return errorAt(op, "Value '{0}' strides do not match requirements '{1}'", opVal, strideReqs);
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
    const float min_val = min().convertToFloat();
    const float max_val = max().convertToFloat();

    const auto clamp = MVCNN::CreateClampParams(writer, min_val, max_val);

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
    const float x_val = x().convertToFloat();

    const auto elu = MVCNN::CreateEluParams(writer, x_val);

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
// Exp
//

void vpux::VPUIP::ExpUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                  mlir::Value output) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ExpUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto exp = MVCNN::CreateExpParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_ExpParams);
    builder.add_nested_params(exp.Union());
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
    const float negative_slope_val = negative_slope().convertToFloat();

    const auto leaky_relu = MVCNN::CreateLeakyReluParams(writer, negative_slope_val);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_LeakyReluParams);
    builder.add_nested_params(leaky_relu.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// Swish
//

void vpux::VPUIP::SwishUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                    mlir::Value output, mlir::FloatAttr beta) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, beta, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SwishUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto beta = beta_valueAttr().getValueAsDouble();

    const auto swish = MVCNN::CreateSwishParams(writer, checked_cast<float>(beta));

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_SwishParams);
    builder.add_nested_params(swish.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// ScaleShift
//

void vpux::VPUIP::ScaleShiftUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value weights, mlir::Value biases, mlir::Value output) {
    build(builder, state, input, weights, biases, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ScaleShiftUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto scaleShift = MVCNN::CreateScaleShiftParams(writer);

    MVCNN::PostOpsNestedParams opType{};
    if (weights() != nullptr && biases() != nullptr) {
        opType = MVCNN::PostOpsNestedParams_ScaleShiftParams;
    } else if (weights() != nullptr) {
        opType = MVCNN::PostOpsNestedParams_ScaleParams;
    } else if (biases() != nullptr) {
        opType = MVCNN::PostOpsNestedParams_BiasParams;
    } else {
        VPUX_THROW("ScaleShift must have weights or biases");
    }

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(opType);
    builder.add_nested_params(scaleShift.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}
