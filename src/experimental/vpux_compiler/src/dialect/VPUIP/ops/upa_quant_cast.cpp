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
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <ngraph/type/float16.hpp>

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::verifyOp(QuantCastUPAOp op) {
    const auto inType = op.input().getType().cast<mlir::MemRefType>().getElementType();
    const auto outType = op.output().getType().cast<mlir::MemRefType>().getElementType();

    if (!((inType.isF16() && outType.isa<mlir::quant::QuantizedType>()) ||
          (inType.isa<mlir::quant::QuantizedType>() && outType.isF16()))) {
        return printTo(op.emitError(), "Unsupported quantize/dequantize conversion '{0}' -> '{1}'", inType, outType);
    }

    const auto qType = inType.isa<mlir::quant::QuantizedType>() ? inType.cast<mlir::quant::QuantizedType>()
                                                                : outType.cast<mlir::quant::QuantizedType>();

    if (!qType.getStorageType().isSignlessInteger(8)) {
        return printTo(op.emitError(), "Unsupported quantized storage type '{0}'", qType.getStorageType());
    }

    if (qType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        // TODO: support per-channel zero point
        const auto zeroPoints = qType.cast<mlir::quant::UniformQuantizedPerAxisType>().getZeroPoints();
        if (zeroPoints.empty()) {
            return printTo(op.emitError(), "Missing zero points");
        }

        const auto firstVal = zeroPoints[0];
        for (auto val : zeroPoints.drop_front()) {
            if (val != firstVal) {
                return printTo(op.emitError(), "Only splat zero points are supported");
            }
        }
    } else if (!qType.isa<mlir::quant::UniformQuantizedType>()) {
        return printTo(op.emitError(), "Unsupported quantized type '{0}'", qType);
    }

    const auto inShape = getShape(op.input());
    const auto outShape = getShape(op.output());

    if (inShape.size() != 4) {
        return printTo(op.emitError(), "Got unsupported input shape '{0}', only 4D is supported", inShape);
    }
    if (outShape.size() != 4) {
        return printTo(op.emitError(), "Got unsupported output shape '{0}', only 4D is supported", outShape);
    }
    if (inShape != outShape) {
        return printTo(op.emitError(), "Input shape '{0}' doesn't match with output shape '{1}'", inShape, outShape);
    }

    const auto inOrder = DimsOrder::fromValue(op.input());
    const auto outOrder = DimsOrder::fromValue(op.output());

    if (!inOrder.hasValue()) {
        return printTo(op.emitError(), "Input Type '{0}' has unknown DimsOrder", op.input().getType());
    }
    if (!outOrder.hasValue()) {
        return printTo(op.emitError(), "Output Type '{0}' has unknown DimsOrder", op.output().getType());
    }

    if (inOrder.getValue() != DimsOrder::NCHW && inOrder.getValue() != DimsOrder::NHWC) {
        return printTo(op.emitError(), "Got unsupported input DimsOrder '{0}', only NCHW and NHWC are supported",
                       inOrder);
    }
    if (inOrder != outOrder) {
        return printTo(op.emitError(), "Input DimsOrder '{0}' doesn't match with output '{1}'", inOrder, outOrder);
    }

    return mlir::success();
}

void vpux::VPUIP::QuantCastUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                        mlir::Value output) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, false);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::QuantCastUPAOp::serialize(BlobWriter& writer) {
    const auto inType = input().getType().cast<mlir::MemRefType>().getElementType();
    const auto outType = output().getType().cast<mlir::MemRefType>().getElementType();

    const auto qType = inType.isa<mlir::quant::QuantizedType>() ? inType.cast<mlir::quant::QuantizedType>()
                                                                : outType.cast<mlir::quant::QuantizedType>();

    const auto getRawFP16 = [](auto val) {
        const auto valFP16 = ngraph::float16(val);
        return valFP16.to_bits();
    };

    const auto getVecFP16 = [&](auto range) {
        return writer.createVector(range | transformed(getRawFP16));
    };

    BlobWriter::Vector<uint16_t> scales;
    BlobWriter::Vector<uint16_t> zeroPoints;

    if (qType.isa<mlir::quant::UniformQuantizedType>()) {
        scales = getVecFP16(makeArrayRef(qType.cast<mlir::quant::UniformQuantizedType>().getScale()));
        zeroPoints = getVecFP16(makeArrayRef(qType.cast<mlir::quant::UniformQuantizedType>().getZeroPoint()));
    } else if (qType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        scales = getVecFP16(qType.cast<mlir::quant::UniformQuantizedPerAxisType>().getScales());
        zeroPoints = getVecFP16(qType.cast<mlir::quant::UniformQuantizedPerAxisType>().getZeroPoints());
    }

    MVCNN::QuantizeParamsBuilder builder(writer);
    builder.add_scale(scales);
    builder.add_zero(zeroPoints);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_QuantizeParams});
}
