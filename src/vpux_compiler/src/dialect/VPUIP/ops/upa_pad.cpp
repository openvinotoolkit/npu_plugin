//
// Copyright 2021 Intel Corporation.
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
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

namespace {

MVCNN::PadMode converVPUXPadModeToMVCNN(vpux::IE::PadMode vpux_mode) {
    MVCNN::PadMode mvcnn_mode;
    switch (vpux_mode) {
    case IE::PadMode::EDGE:
        mvcnn_mode = MVCNN::PadMode::PadMode_Edge;
        break;
    case IE::PadMode::REFLECT:
        mvcnn_mode = MVCNN::PadMode::PadMode_Reflect;
        break;
    case IE::PadMode::CONSTANT:
        mvcnn_mode = MVCNN::PadMode::PadMode_Constant;
        break;
    case IE::PadMode::SYMMETRIC:
        mvcnn_mode = MVCNN::PadMode::PadMode_Symmetric;
        break;
    default:
        VPUX_THROW("Unsupported PadMode {0}", vpux_mode);
    }
    return mvcnn_mode;
}

}  // namespace

mlir::LogicalResult vpux::VPUIP::verifyOp(PadUPAOp op) {
    const auto inShape = getShape(op.input());

    if (inShape.size() != op.pads_begin().size()) {
        return errorAt(op, "pads_begin attr size is not compatible with input tensor."
                           "The length of the list must be equal to the number of dimensions in the input tensor");
    }

    if (inShape.size() != op.pads_end().size()) {
        return errorAt(op, "pads_end attr size is not compatible with input tensor."
                           "The length of the list must be equal to the number of dimensions in the input tensor");
    }

    if (op.mode() == IE::PadMode::CONSTANT && !op.pad_value().hasValue()) {
        return errorAt(op, "pad_mode is CONSTANT but pad_value hasn't provided");
    }

    return mlir::success();
}

void vpux::VPUIP::PadUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                  mlir::Value output, mlir::ArrayAttr pad_begin, mlir::ArrayAttr pad_end,
                                  mlir::FloatAttr pad_value, vpux::IE::PadModeAttr mode) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, pad_begin, pad_end, pad_value, mode,
          nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::PadUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto padsBegin = writer.createVector(parseIntArrayAttr(pads_begin()) | transformed([](int64_t val) {
                                                   return static_cast<uint32_t>(val);
                                               }));
    const auto padsEnd = writer.createVector<>(parseIntArrayAttr(pads_end()) | transformed([](int64_t val) {
                                                   return static_cast<uint32_t>(val);
                                               }));

    MVCNN::PadParamsBuilder builder(writer);
    const auto padMode = converVPUXPadModeToMVCNN(mode());
    builder.add_pad_mode(padMode);
    if (padMode == MVCNN::PadMode::PadMode_Constant) {
        builder.add_padValue(pad_value()->convertToFloat());
    }
    builder.add_pads_begin(padsBegin);
    builder.add_pads_end(padsEnd);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PadParams});
}
