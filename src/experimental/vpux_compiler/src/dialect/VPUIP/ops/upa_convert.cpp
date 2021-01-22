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
