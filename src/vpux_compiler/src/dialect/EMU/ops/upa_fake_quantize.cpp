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
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

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

mlir::LogicalResult vpux::EMU::verifyOp(FakeQuantizeUPAOp op) {
    static const auto C = Dim(1);

    const auto inShape = getShape(op.input());
    const auto outShape = getShape(op.output());
    if (inShape != outShape) {
        return errorAt(op, "Input and output shapes must be equal. Got: {0} != {1}", inShape, outShape);
    }

    const auto inOrder = DimsOrder::fromValue(op.input());
    const auto inStrides = getStrides(op.input());
    const auto memShape = inOrder.toMemoryOrder(inShape);

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

EMU::BlobWriter::SpecificTask vpux::EMU::FakeQuantizeUPAOp::serialize(EMU::BlobWriter& writer) {
    const auto getRawFP16 = [](const float16& val) {
        return val.to_bits();
    };

    const auto getVecFP16 = [&](Const::ContentAttr attr) {
        const auto attrContent = attr.fold();
        return writer.createVector(attrContent.getValues<float16>() | transformed(getRawFP16));
    };

    const auto input_low = getVecFP16(this->input_lowAttr());
    const auto input_high = getVecFP16(this->input_highAttr());
    const auto output_low = getVecFP16(this->output_lowAttr());
    const auto output_high = getVecFP16(this->output_highAttr());

    MVCNN::FakeQuantizeParamsBuilder builder(writer);
    builder.add_levels(checked_cast<uint32_t>(levels()));
    builder.add_input_low(input_low);
    builder.add_input_high(input_high);
    builder.add_output_low(output_low);
    builder.add_output_high(output_high);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_FakeQuantizeParams});
}