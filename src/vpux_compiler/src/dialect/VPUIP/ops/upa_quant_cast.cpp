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

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/utils/extentions.hpp"

#include "vpux/utils/IE/float16.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

namespace {

std::pair<ArrayRef<double>, ArrayRef<int64_t>> getScalesAndZeroPoints(mlir::Value input, mlir::Value output) {
    const auto inType = input.getType().cast<mlir::MemRefType>().getElementType();
    const auto outType = output.getType().cast<mlir::MemRefType>().getElementType();

    const auto qType = inType.isa<mlir::quant::QuantizedType>() ? inType.cast<mlir::quant::QuantizedType>()
                                                                : outType.cast<mlir::quant::QuantizedType>();

    if (qType.isa<mlir::quant::UniformQuantizedType>()) {
        return {makeArrayRef(qType.cast<mlir::quant::UniformQuantizedType>().getScale()),
                makeArrayRef(qType.cast<mlir::quant::UniformQuantizedType>().getZeroPoint())};
    } else if (qType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        return {makeArrayRef(qType.cast<mlir::quant::UniformQuantizedPerAxisType>().getScales()),
                makeArrayRef(qType.cast<mlir::quant::UniformQuantizedPerAxisType>().getZeroPoints())};
    }

    VPUX_THROW("Unsupported quantized type {0}", qType);
}

}  // namespace

mlir::LogicalResult vpux::VPUIP::verifyOp(QuantCastUPAOp op) {
    const auto inType = op.input().getType().cast<mlir::MemRefType>().getElementType();
    const auto outType = op.output().getType().cast<mlir::MemRefType>().getElementType();

    if (!((inType.isF16() && outType.isa<mlir::quant::QuantizedType>()) ||
          (inType.isa<mlir::quant::QuantizedType>() && outType.isF16()))) {
        return errorAt(op, "Unsupported quantize/dequantize conversion '{0}' -> '{1}'", inType, outType);
    }

    const auto qType = inType.isa<mlir::quant::QuantizedType>() ? inType.cast<mlir::quant::QuantizedType>()
                                                                : outType.cast<mlir::quant::QuantizedType>();

    if (!qType.getStorageType().isSignlessInteger(8)) {
        return errorAt(op, "Unsupported quantized storage type '{0}'", qType.getStorageType());
    }

    if (const auto perAxis = qType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        if (perAxis.getQuantizedDimension() != 1) {
            return errorAt(op, "Only per-channel quantization is suppoted");
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

mlir::LogicalResult vpux::VPUIP::QuantCastUPAOp::isSupportedLayout(mlir::Operation* op, vpux::DataOrderInfo& info) {
    const auto quantizeOp = mlir::dyn_cast<IERT::QuantizeOp>(op);
    const auto dequantizeOp = mlir::dyn_cast<IERT::DequantizeOp>(op);

    VPUX_THROW_UNLESS(quantizeOp != nullptr && dequantizeOp != nullptr, "Operation {0} is not quantizer",
                      op->getName());

    LayerInterface layer = quantizeOp ? quantizeOp : dequantizeOp;
    const auto scales = getScalesAndZeroPoints(layer.getInputs()[0], layer.getOutputs()[0]).first;

    if (scales.size() > 1) {
        const auto numDims = layer.getInputs()[0].getType().cast<mlir::ShapedType>().getRank();
        const auto supportedLayout = numDims == 3 ? DimsOrder::HCW : DimsOrder::NHCW;
        if (!info.hasInput(0)) {
            fillDataInfo(info, 1, 1, supportedLayout);
            return mlir::failure();
        }

        const auto mainOrder = info.getInput(0);
        if (mainOrder != DimsOrder::HCW && mainOrder != DimsOrder::NHCW) {
            fillDataInfo(info, 1, 1, supportedLayout);
            return mlir::failure();
        }
    }

    return isSupportedLayoutSameDimsOrder(op, info);
}

void vpux::VPUIP::QuantCastUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                        mlir::Value output) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, false);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::QuantCastUPAOp::serialize(BlobWriter& writer) {
    const auto getRawFP16 = [](auto val) {
        const auto valFP16 = float16(val);
        return valFP16.to_bits();
    };

    const auto getVecFP16 = [&](auto range) {
        return writer.createVector(range | transformed(getRawFP16));
    };

    auto scalesAndZeroPoints = getScalesAndZeroPoints(input(), output());

    BlobWriter::Vector<uint16_t> scales = getVecFP16(scalesAndZeroPoints.first);
    BlobWriter::Vector<uint16_t> zeroPoints = getVecFP16(scalesAndZeroPoints.second);

    MVCNN::QuantizeParamsBuilder builder(writer);
    builder.add_scale(scales);
    builder.add_zero(zeroPoints);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_QuantizeParams});
}
