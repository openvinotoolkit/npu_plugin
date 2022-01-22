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

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::verifyOp(YuvToRgbUPAOp op) {
    if ((op.inFmt() != IE::ColorFmt::NV12) && (op.inFmt() != IE::ColorFmt::I420))
        return errorAt(op, "Invalid INPUT format '{0}'", op.inFmt());
    if ((op.outFmt() != IE::ColorFmt::RGB) && (op.outFmt() != IE::ColorFmt::BGR))
        return errorAt(op, "Invalid OUTPUT format '{0}'", op.outFmt());
    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::YuvToRgbUPAOp::serialize(VPUIP::BlobWriter& writer) {
    if (inFmt() == IE::ColorFmt::NV12) {
        MVCNN::ConvertColorNV12ToRGBParamsBuilder builder(writer);
        if (outFmt() == IE::ColorFmt::RGB) {
            builder.add_colorFormat(MVCNN::RgbFormat::RgbFormat_RGB);
        } else {
            builder.add_colorFormat(MVCNN::RgbFormat::RgbFormat_BGR);
        }
        const auto paramsOff = builder.Finish();
        return writer.createUPALayerTask(*this,
                                         {paramsOff.Union(), MVCNN::SoftwareLayerParams_ConvertColorNV12ToRGBParams});
    } else if (inFmt() == IE::ColorFmt::I420) {
        MVCNN::ConvertColorI420ToRGBParamsBuilder builder(writer);
        if (outFmt() == IE::ColorFmt::RGB) {
            builder.add_colorFormat(MVCNN::RgbFormat::RgbFormat_RGB);
        } else {
            builder.add_colorFormat(MVCNN::RgbFormat::RgbFormat_BGR);
        }
        const auto paramsOff = builder.Finish();
        return writer.createUPALayerTask(*this,
                                         {paramsOff.Union(), MVCNN::SoftwareLayerParams_ConvertColorI420ToRGBParams});
    }
    VPUX_THROW("Invalid color conversion '{0}' -> '{1}'", inFmt(), outFmt());
}

mlir::Operation* vpux::VPUIP::BlobReader::parseYuvToRgb(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                        ArrayRef<mlir::Value> outputs,
                                                        const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(!inputs.empty(), "inputs is empty");
    VPUX_THROW_UNLESS(inputs.size() <= 3, "UPAYuvToRgb supports only 1, 2 or 3 inputs, got {0}", outputs.size());
    const auto NV12params = task->softLayerParams_as_ConvertColorNV12ToRGBParams();
    const auto I420params = task->softLayerParams_as_ConvertColorI420ToRGBParams();
    if (NV12params) {
        auto secondInput = inputs.size() == 1 ? nullptr : inputs[1];
        switch (NV12params->colorFormat()) {
        case MVCNN::RgbFormat::RgbFormat_RGB:
            return builder.create<IE::YuvToRgbOp>(mlir::UnknownLoc::get(_ctx), inputs[0], secondInput, nullptr,
                                                  IE::ColorFmt::NV12, IE::ColorFmt::RGB);
        case MVCNN::RgbFormat::RgbFormat_BGR:
            return builder.create<IE::YuvToRgbOp>(mlir::UnknownLoc::get(_ctx), inputs[0], secondInput, nullptr,
                                                  IE::ColorFmt::NV12, IE::ColorFmt::BGR);
        default:
            VPUX_THROW("Unsupported color format {0}", NV12params->colorFormat());
        }
    }
    if (I420params) {
        auto secondInput = inputs.size() == 1 ? nullptr : inputs[1];
        auto thirdInput = inputs.size() == 1 ? nullptr : inputs[2];
        switch (NV12params->colorFormat()) {
        case MVCNN::RgbFormat::RgbFormat_RGB:
            return builder.create<IE::YuvToRgbOp>(mlir::UnknownLoc::get(_ctx), inputs[0], secondInput, thirdInput,
                                                  IE::ColorFmt::I420, IE::ColorFmt::RGB);
        case MVCNN::RgbFormat::RgbFormat_BGR:
            return builder.create<IE::YuvToRgbOp>(mlir::UnknownLoc::get(_ctx), inputs[0], secondInput, thirdInput,
                                                  IE::ColorFmt::I420, IE::ColorFmt::BGR);
        default:
            VPUX_THROW("Unsupported color format {0}", NV12params->colorFormat());
        }
    }
    VPUX_THROW("Invalid color conversion format");
}
