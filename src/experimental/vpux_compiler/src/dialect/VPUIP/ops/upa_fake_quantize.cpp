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

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::verifyOp(FakeQuantizeUPAOp op) {
    static const Byte SHAVE_LIB_DATA_SIZE = 112_KB;

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

    const Byte elemSize = getElemTypeSize(op.input().getType().cast<mlir::MemRefType>());
    const auto inStrides = getStrides(op.input());

    const auto memShape = inOrder->toMemoryOrder(inShape);
    const auto memStrides = inOrder->toMemoryOrder(inStrides);

    const auto strideReqs = StrideReqs::compact(inShape.size());
    if (!strideReqs.checkStrides(memStrides, elemSize, memShape)) {
        return printTo(op.emitError(), "Only compact strides are supported");
    }

    const auto md0 = memShape[MemDim(0)];
    const auto md1 = memShape[MemDim(1)];

    if (checked_cast<int64_t>(md0 * md1 * sizeof(fp16_t)) >= SHAVE_LIB_DATA_SIZE.count() / 2) {
        return printTo(op.emitError(), "Memory buffers doesn't fit to kernel inner CMX buffer");
    }

    return mlir::success();
}

void vpux::VPUIP::FakeQuantizeUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                           mlir::Value output, uint32_t levels, mlir::ElementsAttr input_low,
                                           mlir::ElementsAttr input_high, mlir::ElementsAttr output_low,
                                           mlir::ElementsAttr output_high) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, levels, input_low, input_high,
          output_low, output_high, nullptr, false);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::FakeQuantizeUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto getRawFP16 = [](const ngraph::float16& val) {
        return val.to_bits();
    };

    const auto getVecFP16 = [&](mlir::ElementsAttr attr) {
        return writer.createVector(attr.cast<ConstContentAttr>().getValues<ngraph::float16>() |
                                   transformed(getRawFP16));
    };

    const auto input_low = getVecFP16(this->input_low());
    const auto input_high = getVecFP16(this->input_high());
    const auto output_low = getVecFP16(this->output_low());
    const auto output_high = getVecFP16(this->output_high());

    MVCNN::FakeQuantizeParamsBuilder builder(writer);
    builder.add_levels(levels());
    builder.add_input_low(input_low);
    builder.add_input_high(input_high);
    builder.add_output_low(output_low);
    builder.add_output_high(output_high);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_FakeQuantizeParams});
}
