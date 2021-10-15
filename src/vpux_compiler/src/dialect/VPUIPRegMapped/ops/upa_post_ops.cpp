//
// Copyright Intel Corporation.
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

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
// Alex: #include "vpux/compiler/dialect/VPUIPRegMapped/blob_reader.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

//
// verifyPostOp
//

mlir::LogicalResult vpux::VPUIPRegMapped::verifyPostOp(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifyPostOp");

    auto layer = mlir::dyn_cast<IERT::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement RT Layer interface", op->getName());
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

void vpux::VPUIPRegMapped::ClampUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                             mlir::Value output, mlir::FloatAttr min, mlir::FloatAttr max) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, min, max, nullptr, nullptr);
}

// VPUIPRegMapped::BlobWriter::SpecificTask vpux::VPUIPRegMapped::ClampUPAOp::serialize(VPUIPRegMapped::BlobWriter&
// writer) {
void vpux::VPUIPRegMapped::ClampUPAOp::serialize(std::vector<char>& buffer) {
    /*
    const float min_val = static_cast<float>(min().convertToDouble());
    const float max_val = static_cast<float>(max().convertToDouble());

    const auto clamp = MVCNN::CreateClampParams(writer, min_val, max_val);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_ClampParams);
    builder.add_nested_params(clamp.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
    */

    (void)buffer;
}

//
// EluUPAOp
//

void vpux::VPUIPRegMapped::EluUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                           mlir::Value output, mlir::FloatAttr x) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, x, nullptr, nullptr);
}

// VPUIPRegMapped::BlobWriter::SpecificTask vpux::VPUIPRegMapped::EluUPAOp::serialize(VPUIPRegMapped::BlobWriter&
// writer) {
void vpux::VPUIPRegMapped::EluUPAOp::serialize(std::vector<char>& buffer) {
    /*
    const float x_val = static_cast<float>(x().convertToDouble());

    const auto elu = MVCNN::CreateEluParams(writer, x_val);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_EluParams);
    builder.add_nested_params(elu.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
    */

    (void)buffer;
}

//
// HSwishUPAOp
//

void vpux::VPUIPRegMapped::HSwishUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                              mlir::Value output) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, nullptr);
}

// VPUIPRegMapped::BlobWriter::SpecificTask vpux::VPUIPRegMapped::HSwishUPAOp::serialize(VPUIPRegMapped::BlobWriter&
// writer) {
void vpux::VPUIPRegMapped::HSwishUPAOp::serialize(std::vector<char>& buffer) {
    /*
    const auto hswish = MVCNN::CreateHSwishParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_HSwishParams);
    builder.add_nested_params(hswish.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
    */

    (void)buffer;
}

//
// FloorUPAOp
//

void vpux::VPUIPRegMapped::FloorUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                             mlir::Value output) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, nullptr);
}

// VPUIPRegMapped::BlobWriter::SpecificTask vpux::VPUIPRegMapped::FloorUPAOp::serialize(VPUIPRegMapped::BlobWriter&
// writer) {
void vpux::VPUIPRegMapped::FloorUPAOp::serialize(std::vector<char>& buffer) {
    /*
    const auto floor = MVCNN::CreateFloorParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_FloorParams);
    builder.add_nested_params(floor.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
    */

    (void)buffer;
}

//
// MishUPAOp
//

void vpux::VPUIPRegMapped::MishUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                            mlir::Value output) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, nullptr);
}

// VPUIPRegMapped::BlobWriter::SpecificTask vpux::VPUIPRegMapped::MishUPAOp::serialize(VPUIPRegMapped::BlobWriter&
// writer) {
void vpux::VPUIPRegMapped::MishUPAOp::serialize(std::vector<char>& buffer) {
    /*
    const auto mish = MVCNN::CreateMishParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_MishParams);
    builder.add_nested_params(mish.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
    */

    (void)buffer;
}

//
// ErfUPAOp
//

void vpux::VPUIPRegMapped::ErfUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                           mlir::Value output) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, nullptr);
}

// VPUIPRegMapped::BlobWriter::SpecificTask vpux::VPUIPRegMapped::ErfUPAOp::serialize(VPUIPRegMapped::BlobWriter&
// writer) {
void vpux::VPUIPRegMapped::ErfUPAOp::serialize(std::vector<char>& buffer) {
    /*
    const auto erf = MVCNN::CreateErfParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_ErfParams);
    builder.add_nested_params(erf.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
    */

    (void)buffer;
}

//
// Tanh
//

void vpux::VPUIPRegMapped::TanhUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                            mlir::Value output) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, nullptr);
}

// VPUIPRegMapped::BlobWriter::SpecificTask vpux::VPUIPRegMapped::TanhUPAOp::serialize(VPUIPRegMapped::BlobWriter&
// writer) {
void vpux::VPUIPRegMapped::TanhUPAOp::serialize(std::vector<char>& buffer) {
    /*
    const auto tanh = MVCNN::CreateTanhParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_TanhParams);
    builder.add_nested_params(tanh.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
    */

    (void)buffer;
}

//
// Exp
//

void vpux::VPUIPRegMapped::ExpUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                           mlir::Value output) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, nullptr);
}

// VPUIPRegMapped::BlobWriter::SpecificTask vpux::VPUIPRegMapped::ExpUPAOp::serialize(VPUIPRegMapped::BlobWriter&
// writer) {
void vpux::VPUIPRegMapped::ExpUPAOp::serialize(std::vector<char>& buffer) {
    /*
    const auto exp = MVCNN::CreateExpParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_ExpParams);
    builder.add_nested_params(exp.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
    */

    (void)buffer;
}

//
// ReLUUPAOp
//

void vpux::VPUIPRegMapped::ReLUUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                            mlir::Value output) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, false);
}

// VPUIPRegMapped::BlobWriter::SpecificTask vpux::VPUIPRegMapped::ReLUUPAOp::serialize(VPUIPRegMapped::BlobWriter&
// writer) {
void vpux::VPUIPRegMapped::ReLUUPAOp::serialize(std::vector<char>& buffer) {
    /*
    const auto relu = MVCNN::CreateReluParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_ReluParams);
    builder.add_nested_params(relu.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
    */

    (void)buffer;
}

//
// SigmoidUPAOp
//

void vpux::VPUIPRegMapped::SigmoidUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                               mlir::Value output) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, false);
}

// VPUIPRegMapped::BlobWriter::SpecificTask vpux::VPUIPRegMapped::SigmoidUPAOp::serialize(VPUIPRegMapped::BlobWriter&
// writer) {
void vpux::VPUIPRegMapped::SigmoidUPAOp::serialize(std::vector<char>& buffer) {
    /*
    const auto sigmoid = MVCNN::CreateSigmoidParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_SigmoidParams);
    builder.add_nested_params(sigmoid.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
    */

    (void)buffer;
}

//
// PRelu
//

void vpux::VPUIPRegMapped::PReluUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                             mlir::Value negative_slope, mlir::Value output) {
    build(builder, state, input, negative_slope, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, nullptr);
}

// VPUIPRegMapped::BlobWriter::SpecificTask vpux::VPUIPRegMapped::PReluUPAOp::serialize(VPUIPRegMapped::BlobWriter&
// writer) {
void vpux::VPUIPRegMapped::PReluUPAOp::serialize(std::vector<char>& buffer) {
    /*
    const auto prelu = MVCNN::CreatePReluParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_PReluParams);
    builder.add_nested_params(prelu.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
    */

    (void)buffer;
}

//
// LeakyRelu
//

void vpux::VPUIPRegMapped::LeakyReluUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                                                 mlir::Value input, mlir::Value output,
                                                 mlir::FloatAttr negative_slope) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, negative_slope, nullptr, nullptr);
}

// VPUIPRegMapped::BlobWriter::SpecificTask vpux::VPUIPRegMapped::LeakyReluUPAOp::serialize(VPUIPRegMapped::BlobWriter&
// writer) {
void vpux::VPUIPRegMapped::LeakyReluUPAOp::serialize(std::vector<char>& buffer) {
    /*
    const float negative_slope_val = static_cast<float>(negative_slope().convertToDouble());

    const auto leaky_relu = MVCNN::CreateLeakyReluParams(writer, negative_slope_val);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_LeakyReluParams);
    builder.add_nested_params(leaky_relu.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
    */

    (void)buffer;
}

//
// Swish
//

void vpux::VPUIPRegMapped::SwishUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                             mlir::Value output, mlir::FloatAttr beta) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, beta, nullptr, nullptr);
}

// VPUIPRegMapped::BlobWriter::SpecificTask vpux::VPUIPRegMapped::SwishUPAOp::serialize(VPUIPRegMapped::BlobWriter&
// writer) {
void vpux::VPUIPRegMapped::SwishUPAOp::serialize(std::vector<char>& buffer) {
    /*
    const auto beta = beta_valueAttr().getValueAsDouble();

    const auto swish = MVCNN::CreateSwishParams(writer, checked_cast<float>(beta));

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_SwishParams);
    builder.add_nested_params(swish.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
    */

    (void)buffer;
}

//
// ScaleShift
//

void vpux::VPUIPRegMapped::ScaleShiftUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                                                  mlir::Value input, mlir::Value weights, mlir::Value biases,
                                                  mlir::Value output) {
    build(builder, state, input, weights, biases, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, nullptr);
}

// VPUIPRegMapped::BlobWriter::SpecificTask vpux::VPUIPRegMapped::ScaleShiftUPAOp::serialize(VPUIPRegMapped::BlobWriter&
// writer) {
void vpux::VPUIPRegMapped::ScaleShiftUPAOp::serialize(std::vector<char>& buffer) {
    /*
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
    */

    (void)buffer;
}

/*
// Alex
mlir::Operation* vpux::VPUIPRegMapped::BlobReader::parsePostOps(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                                ArrayRef<mlir::Value> outputs,
                                                                const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() >= 1 && inputs.size() <= 3, "UPAPostOps supports 1, 2 or 3 inputs, got {0}",
                      inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAPostOps supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_PostOpsParams();

    mlir::Operation* op;
    switch (params->nested_params_type()) {
    case MVCNN::PostOpsNestedParams_ClampParams: {
        const auto clampParams = params->nested_params_as_ClampParams();
        op = builder.create<VPUIPRegMapped::ClampUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0],
                                                        getFPAttr(_ctx, clampParams->min()),
                                                        getFPAttr(_ctx, clampParams->max()));
        break;
    }
    case MVCNN::PostOpsNestedParams_EluParams: {
        const auto eluParams = params->nested_params_as_EluParams();
        op = builder.create<VPUIPRegMapped::EluUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0],
                                                      getFPAttr(_ctx, eluParams->x()));
        break;
    }
    case MVCNN::PostOpsNestedParams_HSwishParams:
        op = builder.create<VPUIPRegMapped::HSwishUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_FloorParams:
        op = builder.create<VPUIPRegMapped::FloorUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_MishParams:
        op = builder.create<VPUIPRegMapped::MishUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_ErfParams:
        op = builder.create<VPUIPRegMapped::ErfUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_TanhParams:
        op = builder.create<VPUIPRegMapped::TanhUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_ReluParams:
        op = builder.create<VPUIPRegMapped::ReLUUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_SigmoidParams:
        op = builder.create<VPUIPRegMapped::SigmoidUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_PReluParams:
        op = builder.create<VPUIPRegMapped::PReluUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_LeakyReluParams: {
        const auto leakyReluParams = params->nested_params_as_LeakyReluParams();
        op = builder.create<VPUIPRegMapped::LeakyReluUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0],
                                                            getFPAttr(_ctx, leakyReluParams->negative_slope()));
        break;
    }
    case MVCNN::PostOpsNestedParams_SwishParams: {
        const auto swishParams = params->nested_params_as_SwishParams();
        op = builder.create<VPUIPRegMapped::SwishUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0],
                                                        getFPAttr(_ctx, swishParams->beta()));
        break;
    }
    case MVCNN::PostOpsNestedParams_BiasParams:
        op = builder.create<VPUIPRegMapped::ScaleShiftUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], nullptr, inputs[1],
                                                             outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_ScaleParams:
        op = builder.create<VPUIPRegMapped::ScaleShiftUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], nullptr,
                                                             outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_ScaleShiftParams:
        op = builder.create<VPUIPRegMapped::ScaleShiftUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1],
                                                             inputs[2], outputs[0]);
        break;
    default:
        VPUX_THROW("Unsupported PostOps operation type {0}", params->nested_params_type());
    }
    return op;
}
*/
