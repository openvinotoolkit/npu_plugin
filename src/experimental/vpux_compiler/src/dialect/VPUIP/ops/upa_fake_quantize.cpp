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

#include "vpux/utils/IE/float16.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

namespace {

bool checkFakeQuantizeParamsShape(ShapeRef shape, int64_t numChannels) {
    if (shape.empty()) {
        return true;
    }

    if (shape.size() == 1) {
        if (shape[Dim(0)] != 1 && shape[Dim(0)] != numChannels) {
            return false;
        }

        return true;
    }

    if (shape.size() != 4) {
        return false;
    }

    if (shape[Dim(0)] != 1 || shape[Dim(2)] != 1 || shape[Dim(3)] != 1) {
        return false;
    }
    if (shape[Dim(1)] != 1 && shape[Dim(1)] != numChannels) {
        return false;
    }

    return true;
}

}  // namespace

mlir::LogicalResult vpux::VPUIP::verifyOp(FakeQuantizeUPAOp op) {
    static const auto C = Dim(1);

    const auto inShape = getShape(op.input());
    const auto inOrder = DimsOrder::fromValue(op.input());
    const auto inStrides = getStrides(op.input());
    const auto memShape = inOrder.toMemoryOrder(inShape);

    const auto strideReqs = StrideReqs::compact(inShape.size());
    if (!strideReqs.checkStrides(op.input())) {
        return errorAt(op, "Only compact strides are supported");
    }

    const auto md0 = memShape[MemDim(0)];
    const auto md1 = memShape[MemDim(1)];

    if (Byte(md0 * md1 * FP16_SIZE) >= Byte(SHAVE_LIB_DATA_SIZE / 2)) {
        return errorAt(op, "Memory buffers doesn't fit to inner CMX buffer");
    }

    const auto numChannels = inShape[C];

    const auto inLowShape = getShape(op.input_low().getType());
    const auto inHighShape = getShape(op.input_high().getType());
    const auto outLowShape = getShape(op.output_low().getType());
    const auto outHighShape = getShape(op.output_high().getType());

    if (!checkFakeQuantizeParamsShape(inLowShape, numChannels)) {
        return errorAt(op, "input_low shape is not per-tensor/per-channel : '{0}'", inLowShape);
    }
    if (!checkFakeQuantizeParamsShape(inHighShape, numChannels)) {
        return errorAt(op, "input_high shape is not per-tensor/per-channel : '{0}'", inHighShape);
    }
    if (!checkFakeQuantizeParamsShape(outLowShape, numChannels)) {
        return errorAt(op, "output_low shape is not per-tensor/per-channel : '{0}'", outLowShape);
    }
    if (!checkFakeQuantizeParamsShape(outHighShape, numChannels)) {
        return errorAt(op, "output_high shape is not per-tensor/per-channel : '{0}'", outHighShape);
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
    const auto getRawFP16 = [](const float16& val) {
        return val.to_bits();
    };

    const auto getVecFP16 = [&](mlir::ElementsAttr attr) {
        return writer.createVector(attr.cast<ConstContentAttr>().getValues<float16>() | transformed(getRawFP16));
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
