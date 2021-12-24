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

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

MVCNN::SpaceToDepthMode convertVPUXSpaceToDepthModeToMVCNN(vpux::IE::SpaceToDepthMode vpux_mode) {
    MVCNN::SpaceToDepthMode mvcnn_mode;
    switch (vpux_mode) {
    case IE::SpaceToDepthMode::BLOCKS_FIRST:
        mvcnn_mode = MVCNN::SpaceToDepthMode::SpaceToDepthMode_BLOCKS_FIRST;
        break;
    case IE::SpaceToDepthMode::DEPTH_FIRST:
        mvcnn_mode = MVCNN::SpaceToDepthMode::SpaceToDepthMode_DEPTH_FIRST;
        break;
    default:
        VPUX_THROW("Unsupported SpaceToDepthMode {0}", vpux_mode);
    }
    return mvcnn_mode;
}

mlir::LogicalResult vpux::VPUIP::verifyOp(SpaceToDepthUPAOp op) {
    if (op.block_size() <= 0) {
        return errorAt(op, "block_size should be greater than zero");
    }

    return mlir::success();
}

void vpux::VPUIP::SpaceToDepthUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                           mlir::Value output, mlir::IntegerAttr block_size,
                                           vpux::IE::SpaceToDepthModeAttr mode) {
    build(builder, state, input, output, block_size, mode, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SpaceToDepthUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::SpaceToDepthParamsBuilder builder(writer);

    const auto blockSize = checked_cast<int32_t>(block_size());
    builder.add_blockSize(blockSize);

    const auto spdMode = convertVPUXSpaceToDepthModeToMVCNN(mode());
    builder.add_mode(spdMode);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_SpaceToDepthParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseSpaceToDepth(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                            ArrayRef<mlir::Value> outputs,
                                                            const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPASpaceToDepth supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPASpaceToDepth supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_SpaceToDepthParams();
    const auto block_size = getIntAttr(_ctx, params->blockSize());

    IE::SpaceToDepthMode spdMode;
    switch (params->mode()) {
    case MVCNN::SpaceToDepthMode::SpaceToDepthMode_BLOCKS_FIRST:
        spdMode = IE::SpaceToDepthMode::BLOCKS_FIRST;
        break;
    case MVCNN::SpaceToDepthMode::SpaceToDepthMode_DEPTH_FIRST:
        spdMode = IE::SpaceToDepthMode::DEPTH_FIRST;
        break;
    default:
        VPUX_THROW("Unsupported SpaceToDepthMode {0}", params->mode());
    }

    return builder.create<VPUIP::SpaceToDepthUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0], block_size,
                                                    IE::SpaceToDepthModeAttr::get(_ctx, spdMode));
}
