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

mlir::LogicalResult vpux::EMU::verifyOp(PadUPAOp op) {
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

EMU::BlobWriter::SpecificTask vpux::EMU::PadUPAOp::serialize(EMU::BlobWriter& writer) {
    const auto padsBegin = writer.createVector(parseIntArrayAttr<uint32_t>(pads_begin()));
    const auto padsEnd = writer.createVector(parseIntArrayAttr<uint32_t>(pads_end()));

    MVCNN::PadParamsBuilder builder(writer);
    const auto padMode = converVPUXPadModeToMVCNN(mode());
    builder.add_pad_mode(padMode);
    if (padMode == MVCNN::PadMode::PadMode_Constant) {
        builder.add_padValue(static_cast<float>(pad_value()->convertToDouble()));
    }
    builder.add_pads_begin(padsBegin);
    builder.add_pads_end(padsEnd);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PadParams});
}
