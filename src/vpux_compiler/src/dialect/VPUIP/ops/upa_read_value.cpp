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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/blob_reader.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/hash.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/hash.hpp"

#include <mlir/IR/BuiltinTypes.h>

#include <unordered_set>

using namespace vpux;

#if 0
mlir::LogicalResult vpux::VPUIP::verifyOp(ReadValueUPAOp op) {
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
#endif

// void vpux::VPUIP::ReadValueUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
//                                         mlir::Value output) {
//     build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, nullptr, false, false,
//           nullptr, nullptr, false);
// }

// TODO: end here
void vpux::VPUIP::ReadValueUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input, mlir::Value input2,
                                        mlir::Value output, mlir::StringAttr variable_id) {

// void vpux::VPUIP::ReadValueUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input, mlir::Value second_input,
//                                         mlir::Value output, mlir::StringAttr variable_id) {
// mlir::ValueRange{}, mlir::ValueRange{}
    std::cout << "vpux::VPUIP::ReadValueUPAOp::build start" << std::endl;
    // build(builder, state, input, second_input, output, mlir::ValueRange{}, mlir::ValueRange{}, variable_id, nullptr, nullptr);
    build(builder, state, input, input2, output, mlir::ValueRange{}, mlir::ValueRange{}, variable_id, nullptr, nullptr);

    // std::cout << "input2 = " << input2.getType().cast<mlir::ShapedType>().getElementType() << std::endl;

    std::cout << "variable_id = " << variable_id.getValue().data() << "," << variable_id.getValue().size() << std::endl;

    std::cout << "vpux::VPUIP::ReadValueUPAOp::build end" << std::endl;
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ReadValueUPAOp::serialize(VPUIP::BlobWriter& writer) {
    VPUIP::BlobWriter::String variable_id_str = writer.createString(variable_id().str());

    std::cout << "serialize: variable_id().str() = " << variable_id().str() << std::endl;

    MVCNN::ReadValueParamsBuilder builder(writer);
    builder.add_variable_id(variable_id_str);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ReadValueParams});
}

// mlir::Operation* vpux::VPUIP::BlobReader::parseConvert(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
//                                                        ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask*) {
//     VPUX_THROW_UNLESS(inputs.size() == 1, "UPAConvert supports only 1 input, got {0}", inputs.size());
//     VPUX_THROW_UNLESS(outputs.size() == 1, "UPAConvert supports only 1 output, got {0}", outputs.size());
//     return builder.create<VPUIP::ConvertUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
// }
