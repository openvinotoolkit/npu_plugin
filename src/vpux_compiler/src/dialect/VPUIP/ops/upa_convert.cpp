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

#include "vpux/compiler/dialect/VPUIP/blob_reader.hpp"
#include "vpux/compiler/utils/hash.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/hash.hpp"

#include <mlir/IR/BuiltinTypes.h>

#include <unordered_set>

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::verifyOp(ConvertUPAOp op) {
    const mlir::Type GF_U8 = getUInt8Type(op.getContext());
    const mlir::Type GF_FP16 = mlir::Float16Type::get(op.getContext());
    const mlir::Type GF_FP32 = mlir::Float32Type::get(op.getContext());
    const mlir::Type GF_INT32 = getSInt32Type(op.getContext());

    const std::unordered_set<std::pair<mlir::Type, mlir::Type>> supportedConversions{
            {GF_FP16, GF_FP32}, {GF_FP16, GF_INT32}, {GF_FP32, GF_FP16}, {GF_INT32, GF_FP16}, {GF_U8, GF_FP16},
            {GF_U8, GF_FP32},   {GF_FP16, GF_U8},    {GF_FP32, GF_U8},   {GF_INT32, GF_U8},
    };

    const auto inType = op.input().getType().cast<mlir::ShapedType>().getElementType();
    const auto outType = op.output().getType().cast<mlir::ShapedType>().getElementType();

    if (supportedConversions.find({inType, outType}) == supportedConversions.end()) {
        return errorAt(op, "Unsupported conversion type : '{0}' -> '{1}'", inType, outType);
    }

    const auto batchID = op.batchID().getValueOr(0);
    if (!op.haveBatch() && batchID != 0) {
        return errorAt(op, "Invalid batch parameters");
    }

    return mlir::success();
}

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

mlir::Operation* vpux::VPUIP::BlobReader::parseConvert(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                       ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask*) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPAConvert supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAConvert supports only 1 output, got {0}", outputs.size());
    return builder.create<VPUIP::ConvertUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
}