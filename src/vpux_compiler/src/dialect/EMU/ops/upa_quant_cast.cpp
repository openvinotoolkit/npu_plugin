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

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/IE/float16.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

namespace {

}  // namespace

mlir::LogicalResult vpux::EMU::verifyOp(QuantCastUPAOp op) {
    const auto inType = op.input().getType().cast<mlir::MemRefType>().getElementType();
    const auto outType = op.output().getType().cast<mlir::MemRefType>().getElementType();

    if (!((inType.isF16() && outType.isa<mlir::quant::QuantizedType>()) ||
          (inType.isa<mlir::quant::QuantizedType>() && outType.isF16()))) {
        return errorAt(op, "Unsupported quantize/dequantize conversion '{0}' -> '{1}'", inType, outType);
    }

    const auto qType = inType.isa<mlir::quant::QuantizedType>() ? inType.cast<mlir::quant::QuantizedType>()
                                                                : outType.cast<mlir::quant::QuantizedType>();

    if (!qType.getStorageType().isInteger(8)) {
        return errorAt(op, "Unsupported quantized storage type '{0}'", qType.getStorageType());
    }

    if (const auto perAxis = qType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        if (perAxis.getQuantizedDimension() != 1) {
            return errorAt(op, "Only per-channel quantization is supported");
        }

        // TODO: support per-channel zero point
        const auto zeroPoints = perAxis.getZeroPoints();
        if (zeroPoints.empty()) {
            return errorAt(op, "Missing zero points");
        }

        const auto firstVal = zeroPoints[0];
        for (auto val : zeroPoints.drop_front()) {
            if (val != firstVal) {
                return errorAt(op, "Only splat zero points are supported");
            }
        }
    } else if (!qType.isa<mlir::quant::UniformQuantizedType>()) {
        return errorAt(op, "Unsupported quantized type '{0}'", qType);
    }

    return mlir::success();
}
