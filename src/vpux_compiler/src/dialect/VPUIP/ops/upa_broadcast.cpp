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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/blob_reader.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

void vpux::VPUIP::BroadcastUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                        mlir::Value target_shape, mlir::Value axes_mapping, mlir::Value output,
                                        IE::BroadcastTypeAttr mode) {
    build(builder, state, input, target_shape, axes_mapping, output, mlir::ValueRange{}, mlir::ValueRange{}, mode,
          nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::BroadcastUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::BroadcastParamsBuilder builder(writer);

    MVCNN::BroadcastMode mvcnn_mode;

    if (this->mode() == IE::BroadcastType::NUMPY) {
        mvcnn_mode = MVCNN::BroadcastMode::BroadcastMode_NUMPY;
    } else if (this->mode() == IE::BroadcastType::BIDIRECTIONAL) {
        mvcnn_mode = MVCNN::BroadcastMode::BroadcastMode_BIDIRECTIONAL;
    } else if (this->mode() == IE::BroadcastType::EXPLICIT) {
        mvcnn_mode = MVCNN::BroadcastMode::BroadcastMode_EXPLICIT;
    } else {
        VPUX_THROW("Unsupported broadcast mode {0}", this->mode());
    }

    builder.add_mode(mvcnn_mode);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_BroadcastParams});
}
