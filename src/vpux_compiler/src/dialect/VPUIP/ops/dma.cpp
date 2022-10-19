//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPURT/types.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

namespace {

mlir::LogicalResult verifyTensorSize(mlir::Location loc, mlir::Value tensor) {
    const auto size = static_cast<Byte>(getCompactSize(tensor));

    if (size <= VPUIP::DMA_LIMIT) {
        return mlir::success();
    }

    return errorAt(loc, "The size of the DMA transaction {0} for a {1} tensor is greater than the limit {2}", size,
                   getShape(tensor), VPUIP::DMA_LIMIT);
}

}  // namespace

//
// NNDMAOp
//

void vpux::VPUIP::NNDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                 mlir::Value output_buff) {
    build(builder, state, input, output_buff, /*port=*/0, /*is_out_of_order=*/false, /*is_critical=*/false);
}

void vpux::VPUIP::NNDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                 mlir::Value output_buff, int64_t port) {
    build(builder, state, input, output_buff, /*port=*/port, /*is_out_of_order=*/false, /*is_critical=*/false);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::NNDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(input());
    const auto dstOff = writer.getTensorRef(output_buff());

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_port(checked_cast<uint8_t>(port()));
    builder.add_set_ord(static_cast<uint8_t>(!is_out_of_order()));  // ORD
    builder.add_set_crit(static_cast<uint8_t>(is_critical()));      // CRIT
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::verifyOp(NNDMAOp op) {
    auto loc = op.getLoc();

    if (auto distributedOut = op.output().getType().dyn_cast<VPUIP::DistributedBufferType>()) {
        auto mode = distributedOut.getDistribution().mode().getValue();
        // In this case DUPLICATED|SEGMENTED is an alias of DUPLICATED mode
        // SEGMENTED|MULTICASTED which can be a result of spilling of NCE output in SEGMENTED|MULTICASTED mode should be
        // treated as a DUPLICATED mode
        if (mode != VPU::DistributionMode::DUPLICATED &&
            mode != (VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED) &&
            mode != (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED)) {
            return errorAt(loc, "Only duplicated mode supported for output operand. Got: {0}",
                           VPU::stringifyDistributionMode(mode));
        }
    }

    return verifyTensorSize(loc, op.input());
}

//
// PermuteDMAOp
//

void vpux::VPUIP::PermuteDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                      mlir::Value output_buff, mlir::IntegerAttr dst_stride,
                                      mlir::AffineMapAttr mem_perm) {
    build(builder, state, input, output_buff, dst_stride, /*port=*/vpux::getIntAttr(builder, 0),
          /*is_out_of_order=*/nullptr,
          /*is_critical=*/nullptr, mem_perm);
}

void vpux::VPUIP::PermuteDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                      mlir::Value output_buff, mlir::IntegerAttr dst_stride,
                                      mlir::AffineMapAttr mem_perm, mlir::IntegerAttr port) {
    build(builder, state, input, output_buff, dst_stride, /*port=*/port,
          /*is_out_of_order=*/nullptr,
          /*is_critical=*/nullptr, mem_perm);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::PermuteDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(input());
    const auto dstOff = writer.getTensorRef(output_buff());
    const auto descriptor = writer.getPermuteNNDMADescriptorReference(getOperation());

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_port(checked_cast<uint8_t>(port()));
    builder.add_set_ord(static_cast<uint8_t>(!is_out_of_order()));  // ORD
    builder.add_set_crit(static_cast<uint8_t>(is_critical()));      // CRIT
    builder.add_descriptor(&descriptor);
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::verifyOp(PermuteDMAOp op) {
    auto loc = op.getLoc();
    return verifyTensorSize(loc, op.input());
}

//
// CompressedDMAOp
//

void vpux::VPUIP::CompressedDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value output_buff) {
    build(builder, state, input, output_buff, /*port=*/0, /*is_out_of_order=*/false, /*is_critical=*/false);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::CompressedDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(input());
    const auto dstOff = writer.getTensorRef(output_buff());

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_compression(true);
    builder.add_port(checked_cast<uint8_t>(port()));
    builder.add_set_ord(static_cast<uint8_t>(!is_out_of_order()));  // ORD
    builder.add_set_crit(static_cast<uint8_t>(is_critical()));      // CRIT
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

//
// DepthToSpaceDMAOp
//
void vpux::VPUIP::DepthToSpaceDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                           mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr block_size,
                                           vpux::IE::DepthToSpaceModeAttr mode, mlir::IntegerAttr output_channel,
                                           mlir::IntegerAttr output_width) {
    build(odsBuilder, odsState, input, output_buff, vpux::getIntAttr(odsBuilder, 0), block_size, mode, output_channel,
          output_width);
}

void vpux::VPUIP::DepthToSpaceDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                           mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr block_size,
                                           vpux::IE::DepthToSpaceModeAttr mode, mlir::IntegerAttr output_channel,
                                           mlir::IntegerAttr output_width, mlir::IntegerAttr port) {
    build(odsBuilder, odsState, input, output_buff, port, block_size, mode, output_channel, output_width);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::DepthToSpaceDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(input());
    const auto dstOff = writer.getTensorRef(output_buff());
    const auto descriptor = writer.getDepthToSpaceNNDMADescriptorReference(getOperation());

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_port(checked_cast<uint8_t>(port()));
    builder.add_set_ord(static_cast<uint8_t>(true));    // ORD
    builder.add_set_crit(static_cast<uint8_t>(false));  // CRIT
    builder.add_descriptor(&descriptor);
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::verifyOp(DepthToSpaceDMAOp op) {
    auto loc = op.getLoc();

    return verifyTensorSize(loc, op.input());
}
