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

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/hash.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/hash.hpp"

#include <mlir/IR/BuiltinTypes.h>

#include <unordered_set>

using namespace vpux;

mlir::LogicalResult vpux::EMU::verifyOp(ConvertUPAOp op) {
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
