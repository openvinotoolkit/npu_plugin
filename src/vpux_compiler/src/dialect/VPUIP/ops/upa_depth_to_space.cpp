//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

MVCNN::DepthToSpaceMode DepthToSpaceMode2MVCNN(IE::DepthToSpaceMode mode) {
    MVCNN::DepthToSpaceMode out_code = MVCNN::DepthToSpaceMode_BLOCKS_FIRST;
    switch (mode) {
    case IE::DepthToSpaceMode::BLOCKS_FIRST:
        out_code = MVCNN::DepthToSpaceMode_BLOCKS_FIRST;
        break;
    case IE::DepthToSpaceMode::DEPTH_FIRST:
        out_code = MVCNN::DepthToSpaceMode_DEPTH_FIRST;
        break;
    default:
        VPUX_THROW("Unknown DepthToSpaceMode. Blocks_FIRST and DEPTH_FIRST methods are supported only");
    }
    return out_code;
}

mlir::LogicalResult vpux::VPUIP::verifyOp(DepthToSpaceUPAOp op) {
    if (op.block_size() <= 0) {
        return errorAt(op, "Block size should be greater than 0. Got {0}", op.block_size());
    }

    if (op.mode() != vpux::IE::DepthToSpaceMode::BLOCKS_FIRST && op.mode() != vpux::IE::DepthToSpaceMode::DEPTH_FIRST) {
        return errorAt(op, "Unknown DepthToSpaceMode. Blocks_FIRST and DEPTH_FIRST methods are supported only");
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::DepthToSpaceUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::DepthToSpaceParamsBuilder builder(writer);

    const auto blockSize = checked_cast<int32_t>(block_size());
    builder.add_blockSize(blockSize);

    builder.add_mode(DepthToSpaceMode2MVCNN(mode()));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_DepthToSpaceParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseDepthToSpace(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                            ArrayRef<mlir::Value> outputs,
                                                            const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPADepthToSpace supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPADepthToSpace supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_DepthToSpaceParams();
    const auto block_size = getIntAttr(_ctx, params->blockSize());
    IE::DepthToSpaceMode mode;
    switch (params->mode()) {
    case MVCNN::DepthToSpaceMode_BLOCKS_FIRST:
        mode = IE::DepthToSpaceMode::BLOCKS_FIRST;
        break;
    case MVCNN::DepthToSpaceMode_DEPTH_FIRST:
        mode = IE::DepthToSpaceMode::DEPTH_FIRST;
        break;
    default:
        VPUX_THROW("Unknown DepthToSpaceMode. BLOCKS_FIRST and DEPTH_FIRST methods are supported only");
    }
    return builder.create<VPUIP::DepthToSpaceUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0], block_size,
                                                    IE::DepthToSpaceModeAttr::get(_ctx, mode));
}
