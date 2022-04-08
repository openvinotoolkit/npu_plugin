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
                                      mlir::Value output_buff, mlir::AffineMapAttr mem_perm,
                                      VPUIP::DmaDescriptorAttr dma_descriptor) {
    build(builder, state, input, output_buff, /*port=*/vpux::getIntAttr(builder, 0),
          /*is_out_of_order=*/nullptr,
          /*is_critical=*/nullptr, mem_perm, dma_descriptor);
}

void vpux::VPUIP::PermuteDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                      mlir::Value output_buff, mlir::AffineMapAttr mem_perm,
                                      VPUIP::DmaDescriptorAttr dma_descriptor, mlir::IntegerAttr port) {
    build(builder, state, input, output_buff, /*port=*/port,
          /*is_out_of_order=*/nullptr,
          /*is_critical=*/nullptr, mem_perm, dma_descriptor);
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

void vpux::VPUIP::CompressedDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value output_buff, int64_t port) {
    build(builder, state, input, output_buff, /*port=*/port, /*is_out_of_order=*/false, /*is_critical=*/false);
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

mlir::LogicalResult vpux::VPUIP::verifyOp(CompressedDMAOp op) {
    auto loc = op.getLoc();

    if (auto distributedOut = op.output().getType().dyn_cast<VPUIP::DistributedBufferType>()) {
        auto mode = distributedOut.getDistribution().mode().getValue();
        if (mode != VPU::DistributionMode::DUPLICATED) {
            return errorAt(loc, "Only duplicated mode supported for output operand. Got: {0}",
                           VPU::stringifyDistributionMode(mode));
        }
    }

    return verifyTensorSize(loc, op.input());
}

//
// DepthToSpaceDMAOp
//

void vpux::VPUIP::DepthToSpaceDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                           mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr block_size,
                                           vpux::IE::DepthToSpaceModeAttr mode, VPUIP::DmaDescriptorAttr dma_descriptor,
                                           vpux::IE::ChannelPadding padded_channels) {
    build(odsBuilder, odsState, input, output_buff, /*port=*/vpux::getIntAttr(odsBuilder, 0), block_size, mode,
          dma_descriptor, padded_channels);
}

void vpux::VPUIP::DepthToSpaceDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                           mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr block_size,
                                           vpux::IE::DepthToSpaceModeAttr mode, VPUIP::DmaDescriptorAttr dma_descriptor,
                                           mlir::IntegerAttr port, vpux::IE::ChannelPadding padded_channels) {
    build(odsBuilder, odsState, input, output_buff, port, block_size, mode, dma_descriptor, padded_channels);
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

//
// SpaceToDepthDMAOp
//
void vpux::VPUIP::SpaceToDepthDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                           mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr block_size,
                                           vpux::IE::SpaceToDepthModeAttr mode,
                                           VPUIP::DmaDescriptorAttr dma_descriptor) {
    build(odsBuilder, odsState, input, output_buff, vpux::getIntAttr(odsBuilder, 0), block_size, mode, dma_descriptor);
}

void vpux::VPUIP::SpaceToDepthDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                           mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr block_size,
                                           vpux::IE::SpaceToDepthModeAttr mode, VPUIP::DmaDescriptorAttr dma_descriptor,
                                           mlir::IntegerAttr port) {
    build(odsBuilder, odsState, input, output_buff, port, block_size, mode, dma_descriptor);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SpaceToDepthDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(input());
    const auto dstOff = writer.getTensorRef(output_buff());
    const auto descriptor = writer.getSpaceToDepthNNDMADescriptorReference(getOperation());

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_port(checked_cast<uint8_t>(port()));
    builder.add_set_ord(static_cast<uint8_t>(true));    // ORD
    builder.add_set_crit(static_cast<uint8_t>(false));  // CRIT
    builder.add_descriptor(&descriptor);
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::verifyOp(SpaceToDepthDMAOp op) {
    auto loc = op.getLoc();

    return verifyTensorSize(loc, op.input());
}

//
// ExpandDMAOp
//

void vpux::VPUIP::ExpandDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     mlir::Value output_buff, mlir::ArrayAttr pads_begin, mlir::ArrayAttr pads_end,
                                     VPUIP::DmaDescriptorAttr dma_descriptor) {
    build(builder, state, input, output_buff, pads_begin, pads_end, dma_descriptor,
          /*port=*/vpux::getIntAttr(builder, 0),
          /*is_out_of_order=*/nullptr,
          /*is_critical=*/nullptr);
}

void vpux::VPUIP::ExpandDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     mlir::Value output_buff, mlir::ArrayAttr pads_begin, mlir::ArrayAttr pads_end,
                                     VPUIP::DmaDescriptorAttr dma_descriptor, mlir::IntegerAttr port) {
    build(builder, state, input, output_buff, pads_begin, pads_end, dma_descriptor, /*port=*/port,
          /*is_out_of_order=*/nullptr,
          /*is_critical=*/nullptr);
}

mlir::LogicalResult vpux::VPUIP::verifyOp(ExpandDMAOp op) {
    auto loc = op.getLoc();

    // In case ExpandDMA with input size large than VPUIP::DMA_LIMIT (16MB).
    // It should be tiled with several sub ExpandDMA that will be done at Unroll Pass.
    // Descriptor is generated at Unroll pass so using Descriptor as a flag to check the tensor size.
    if (op.dma_descriptor().hasValue()) {
        return verifyTensorSize(loc, op.input());
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ExpandDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(input());
    const auto dstOff = writer.getTensorRef(output_buff());
    const auto descriptor = writer.getExpandNNDMADescriptorReference(getOperation());

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_port(checked_cast<uint8_t>(port()));
    builder.add_set_ord(static_cast<uint8_t>(!is_out_of_order()));  // ORD
    builder.add_set_crit(static_cast<uint8_t>(is_critical()));      // CRIT
    builder.add_descriptor(&descriptor);
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

//
// PerAxisTileDMAOp
//

void vpux::VPUIP::PerAxisTileDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                          mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr axis,
                                          mlir::IntegerAttr tiles, VPUIP::DmaDescriptorAttr dma_descriptor) {
    build(odsBuilder, odsState, input, output_buff, vpux::getIntAttr(odsBuilder, 0), axis, tiles, dma_descriptor);
}

void vpux::VPUIP::PerAxisTileDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                          mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr axis,
                                          mlir::IntegerAttr tiles, VPUIP::DmaDescriptorAttr dma_descriptor,
                                          mlir::IntegerAttr port) {
    build(odsBuilder, odsState, input, output_buff, port, axis, tiles, dma_descriptor);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::PerAxisTileDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(input());
    const auto dstOff = writer.getTensorRef(output_buff());
    const auto descriptor = writer.getPerAxisTileNNDMADescriptorReference(getOperation());

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_port(checked_cast<uint8_t>(port()));
    builder.add_set_ord(static_cast<uint8_t>(true));    // ORD
    builder.add_set_crit(static_cast<uint8_t>(false));  // CRIT
    builder.add_descriptor(&descriptor);
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::verifyOp(PerAxisTileDMAOp op) {
    auto loc = op.getLoc();

    return verifyTensorSize(loc, op.input());
}

//
// UpsamplingDMAOp
//

mlir::LogicalResult vpux::VPUIP::verifyOp(UpsamplingDMAOp op) {
    auto loc = op.getLoc();
    return verifyTensorSize(loc, op.input());
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::UpsamplingDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(input());
    const auto dstOff = writer.getTensorRef(output_buff());
    const auto descriptor = writer.getUpsamplingNNDMADescriptorReference(getOperation());

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_port(checked_cast<uint8_t>(port()));
    builder.add_set_ord(static_cast<uint8_t>(true));    // ORD
    builder.add_set_crit(static_cast<uint8_t>(false));  // CRIT
    builder.add_descriptor(&descriptor);
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}
