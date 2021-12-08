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

#include "vpux/compiler/dialect/VPUIP/blob_reader.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::verifyOp(YuvToRgbUPAOp op) {
    if ((op.inFmt() != IE::ColorFmt::NV12) && (op.inFmt() != IE::ColorFmt::I420))
        return errorAt(op, "YuvToRgb: invalid INPUT format");
    if ((op.outFmt() != IE::ColorFmt::RGB) && (op.outFmt() != IE::ColorFmt::BGR))
        return errorAt(op, "YuvToRgb: invalid OUTPUT format");
    return mlir::success();
}

void vpux::VPUIP::YuvToRgbUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input1,
                                       mlir::Value input2, mlir::Value input3, mlir::Value output,
                                       IE::ColorFmtAttr inFmt, IE::ColorFmtAttr outFmt) {
    build(builder, state, input1, input2, input3, output, inFmt, outFmt, nullptr);
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
    VPUX_THROW("Invalid color conversion !!!");
}
