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

#include "vpux/compiler/dialect/EMU/ops.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

namespace {

MVCNN::RoundMode converVPUXRoundModeToMVCNN(vpux::IE::RoundMode vpux_mode) {
    MVCNN::RoundMode mvcnn_mode;
    switch (vpux_mode) {
    case IE::RoundMode::HALF_TO_EVEN:
        mvcnn_mode = MVCNN::RoundMode::RoundMode_HALF_TO_EVEN;
        break;
    case IE::RoundMode::HALF_AWAY_FROM_ZERO:
        mvcnn_mode = MVCNN::RoundMode::RoundMode_HALF_AWAY_FROM_ZERO;
        break;
    default:
        VPUX_THROW("Unsupported RoundMode {0}", vpux_mode);
    }
    return mvcnn_mode;
}

}  // namespace
//
// verifyPostOp
//

mlir::LogicalResult vpux::EMU::verifyPostOp(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifyPostOp");

    for (auto& operand : op->getOpOperands()) {
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

EMU::BlobWriter::SpecificTask vpux::EMU::ClampUPAOp::serialize(EMU::BlobWriter& writer) {
    const float min_val = static_cast<float>(min().convertToDouble());
    const float max_val = static_cast<float>(max().convertToDouble());

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

EMU::BlobWriter::SpecificTask vpux::EMU::EluUPAOp::serialize(EMU::BlobWriter& writer) {
    const float x_val = static_cast<float>(x().convertToDouble());

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

EMU::BlobWriter::SpecificTask vpux::EMU::HSwishUPAOp::serialize(EMU::BlobWriter& writer) {
    const auto hswish = MVCNN::CreateHSwishParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_HSwishParams);
    builder.add_nested_params(hswish.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// FloorUPAOp
//

EMU::BlobWriter::SpecificTask vpux::EMU::FloorUPAOp::serialize(EMU::BlobWriter& writer) {
    const auto floor = MVCNN::CreateFloorParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_FloorParams);
    builder.add_nested_params(floor.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// RoundUPAOp
//

EMU::BlobWriter::SpecificTask vpux::EMU::RoundUPAOp::serialize(EMU::BlobWriter& writer) {
    const auto roundMode = converVPUXRoundModeToMVCNN(mode());
    const auto round = MVCNN::CreateRoundParams(writer, roundMode);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_RoundParams);
    builder.add_nested_params(round.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// MishUPAOp
//

EMU::BlobWriter::SpecificTask vpux::EMU::MishUPAOp::serialize(EMU::BlobWriter& writer) {
    const auto mish = MVCNN::CreateMishParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_MishParams);
    builder.add_nested_params(mish.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// ErfUPAOp
//

EMU::BlobWriter::SpecificTask vpux::EMU::ErfUPAOp::serialize(EMU::BlobWriter& writer) {
    const auto erf = MVCNN::CreateErfParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_ErfParams);
    builder.add_nested_params(erf.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// Tanh
//

EMU::BlobWriter::SpecificTask vpux::EMU::TanhUPAOp::serialize(EMU::BlobWriter& writer) {
    const auto tanh = MVCNN::CreateTanhParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_TanhParams);
    builder.add_nested_params(tanh.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// Sqrt
//

EMU::BlobWriter::SpecificTask vpux::EMU::SqrtUPAOp::serialize(EMU::BlobWriter& writer) {
    const auto sqrt = MVCNN::CreateSqrtParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_SqrtParams);
    builder.add_nested_params(sqrt.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// LogUPAOp
//

EMU::BlobWriter::SpecificTask vpux::EMU::LogUPAOp::serialize(EMU::BlobWriter& writer) {
    const auto log = MVCNN::CreateLogParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_LogParams);
    builder.add_nested_params(log.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// Exp
//

EMU::BlobWriter::SpecificTask vpux::EMU::ExpUPAOp::serialize(EMU::BlobWriter& writer) {
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

EMU::BlobWriter::SpecificTask vpux::EMU::ReLUUPAOp::serialize(EMU::BlobWriter& writer) {
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

EMU::BlobWriter::SpecificTask vpux::EMU::SigmoidUPAOp::serialize(EMU::BlobWriter& writer) {
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

EMU::BlobWriter::SpecificTask vpux::EMU::PReluUPAOp::serialize(EMU::BlobWriter& writer) {
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

EMU::BlobWriter::SpecificTask vpux::EMU::LeakyReluUPAOp::serialize(EMU::BlobWriter& writer) {
    const float negative_slope_val = static_cast<float>(negative_slope().convertToDouble());

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

EMU::BlobWriter::SpecificTask vpux::EMU::SwishUPAOp::serialize(EMU::BlobWriter& writer) {
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

EMU::BlobWriter::SpecificTask vpux::EMU::ScaleShiftUPAOp::serialize(EMU::BlobWriter& writer) {
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

//
// CeilingUPAOp
//

EMU::BlobWriter::SpecificTask vpux::EMU::CeilingUPAOp::serialize(EMU::BlobWriter& writer) {
    const auto ceiling = MVCNN::CreateCeilingParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_CeilingParams);
    builder.add_nested_params(ceiling.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}
