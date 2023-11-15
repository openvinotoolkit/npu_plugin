//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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

mlir::LogicalResult verifyInOutElementType(mlir::Location loc, mlir::Value inTensor, mlir::Value outTensor) {
    const auto inType = inTensor.getType().cast<vpux::NDTypeInterface>();
    const auto outType = outTensor.getType().cast<vpux::NDTypeInterface>();

    if (inType.getElementType() != outType.getElementType()) {
        return errorAt(loc, "Input element type '{0}' doesn't match output element type '{1}'", inType.getElementType(),
                       outType.getElementType());
    }

    return mlir::success();
}

}  // namespace

//
// NNDMAOp
//

void vpux::VPUIP::NNDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                 mlir::Value output_buff) {
    build(builder, state, input, output_buff, /*port=*/0, /*channelType=*/nullptr, /*is_out_of_order=*/false,
          /*is_critical=*/false,
          /*spillId=*/nullptr, /*dma_hwp_id*/ nullptr);
}

void vpux::VPUIP::NNDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                 mlir::Value output_buff, int64_t port) {
    build(builder, state, input, output_buff, /*port=*/port, /*channelType=*/nullptr, /*is_out_of_order=*/false,
          /*is_critical=*/false,
          /*spillId=*/nullptr, /*dma_hwp_id*/ nullptr);
}

void vpux::VPUIP::NNDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                 mlir::Value output_buff, int64_t port, int32_t dma_hwp_id) {
    build(builder, state, input, output_buff, /*port=*/port, /*channelType=*/nullptr, /*is_out_of_order=*/false,
          /*is_critical=*/false,
          /*spillId=*/nullptr, vpux::getIntAttr(builder, dma_hwp_id));
}

void vpux::VPUIP::NNDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                 mlir::Value output_buff, mlir::IntegerAttr port, mlir::UnitAttr is_out_of_order,
                                 mlir::UnitAttr is_critical, mlir::IntegerAttr spillId) {
    build(builder, state, input, output_buff, /*port=*/port, /*channelType=*/nullptr,
          /*is_out_of_order=*/is_out_of_order,
          /*is_critical=*/is_critical, /*spillId=*/spillId, /*dma_hwp_id*/ nullptr);
}

void vpux::VPUIP::NNDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                 mlir::Value output_buff, int64_t port, bool is_out_of_order, bool is_critical,
                                 mlir::IntegerAttr spillId) {
    build(builder, state, input, output_buff, /*port=*/port, /*channelType=*/nullptr,
          /*is_out_of_order=*/is_out_of_order,
          /*is_critical=*/is_critical, /*spillId=*/spillId, /*dma_hwp_id*/ nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::NNDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(input());
    const auto dstOff = writer.getTensorRef(output_buff());

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_dma_hwp_id(this->dma_hwp_id().value_or(0));
    builder.add_port(checked_cast<uint8_t>(port()));
    builder.add_set_ord(static_cast<uint8_t>(!is_out_of_order()));  // ORD
    builder.add_set_crit(static_cast<uint8_t>(is_critical()));      // CRIT
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::NNDMAOp::verify() {
    auto loc = getLoc();
    return verifyTensorSize(loc, input());
}

//
// PermuteDMAOp
//

void vpux::VPUIP::PermuteDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                      mlir::Value output_buff, mlir::AffineMapAttr mem_perm,
                                      VPUIP::DMADescriptorAttr dma_descriptor) {
    build(builder, state, input, output_buff, /*port=*/vpux::getIntAttr(builder, 0), /*channelType=*/nullptr,
          /*is_out_of_order=*/nullptr,
          /*is_critical=*/nullptr, mem_perm, dma_descriptor, /* dma_hwp_id= */ nullptr);
}

void vpux::VPUIP::PermuteDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                      mlir::Value output_buff, mlir::AffineMapAttr mem_perm,
                                      VPUIP::DMADescriptorAttr dma_descriptor, mlir::IntegerAttr port) {
    build(builder, state, input, output_buff, /*port=*/port, /*channelType=*/nullptr,
          /*is_out_of_order=*/nullptr,
          /*is_critical=*/nullptr, mem_perm, dma_descriptor, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::PermuteDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(input());
    const auto dstOff = writer.getTensorRef(output_buff());
    const auto descriptor = writer.getPermuteNNDMADescriptorReference(getOperation());

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_port(checked_cast<uint8_t>(port()));
    builder.add_dma_hwp_id(this->dma_hwp_id().value_or(0));
    builder.add_set_ord(static_cast<uint8_t>(!is_out_of_order()));  // ORD
    builder.add_set_crit(static_cast<uint8_t>(is_critical()));      // CRIT
    builder.add_descriptor(&descriptor);
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::PermuteDMAOp::verify() {
    return verifyTensorSize(getLoc(), input());
}

//
// ConvertDMAOp
//

void vpux::VPUIP::ConvertDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                      mlir::Value output_buff) {
    build(builder, state, input, output_buff, /*port=*/0, /*channelType=*/nullptr, /*is_out_of_order=*/false,
          /*is_critical=*/false,
          /* dma_hwp_id=*/nullptr);
}

void vpux::VPUIP::ConvertDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                      mlir::Value output_buff, int64_t port) {
    build(builder, state, input, output_buff, /*port=*/port, /*channelType=*/nullptr, /*is_out_of_order=*/false,
          /*is_critical=*/false,
          /* dma_hwp_id=*/nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ConvertDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(input());
    const auto dstOff = writer.getTensorRef(output_buff());

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_dma_hwp_id(this->dma_hwp_id().value_or(0));
    builder.add_port(checked_cast<uint8_t>(port()));
    builder.add_set_ord(static_cast<uint8_t>(!is_out_of_order()));  // ORD
    builder.add_set_crit(static_cast<uint8_t>(is_critical()));      // CRIT
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::ConvertDMAOp::verify() {
    auto loc = getLoc();
    auto arch = VPU::getArch(getOperation());

    auto outputType = output_buff().getType().cast<vpux::NDTypeInterface>();
    const auto outputElementType = outputType.getElementType();
    auto inputType = input().getType().cast<vpux::NDTypeInterface>();
    const auto inputElementType = inputType.getElementType();

    if (!inputElementType.isF32() || (!outputElementType.isF16() && !outputElementType.isBF16())) {
        return errorAt(loc,
                       "Operation {0} is unsupported. Got arch {1} "
                       "and conversion from {2} to {3}",
                       getOperationName(), arch, inputElementType, outputElementType);
    }

    return verifyTensorSize(loc, input());
}

//
// DecompressDMAOp
//

void vpux::VPUIP::DecompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value output_buff, int64_t port, bool is_out_of_order,
                                         bool is_critical) {
    build(builder, state, input, /*act_compression_size_entry=*/nullptr, output_buff, /*port=*/port,
          /*channelType=*/nullptr,
          /*is_out_of_order=*/is_out_of_order, /*is_critical=*/is_critical, /* dma_hwp_id= */ nullptr);
}

void vpux::VPUIP::DecompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value output_buff, mlir::IntegerAttr port,
                                         mlir::UnitAttr is_out_of_order, mlir::UnitAttr is_critical) {
    build(builder, state, input, /*act_compression_size_entry=*/nullptr, output_buff, /*port=*/port,
          /*channelType=*/nullptr,
          /*is_out_of_order=*/is_out_of_order, /*is_critical=*/is_critical,
          /*dma_hwp_id=*/nullptr);
}

void vpux::VPUIP::DecompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value output_buff) {
    build(builder, state, input, output_buff, /*port=*/0, /*is_out_of_order=*/false,
          /*is_critical=*/false);
}

void vpux::VPUIP::DecompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value output_buff, int64_t port) {
    build(builder, state, input, output_buff, /*port=*/port, /*is_out_of_order=*/false,
          /*is_critical=*/false);
}

void vpux::VPUIP::DecompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value actCompressionSizeEntryBuff, mlir::Value output_buff, int64_t port,
                                         bool is_out_of_order, bool is_critical) {
    build(builder, state, input, actCompressionSizeEntryBuff, output_buff, /*port=*/port, /*channelType=*/nullptr,
          /*is_out_of_order=*/is_out_of_order, /*is_critical=*/is_critical,
          /* dma_hwp_id= */ nullptr);
}

void vpux::VPUIP::DecompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value actCompressionSizeEntryBuff, mlir::Value output_buff,
                                         mlir::IntegerAttr port, mlir::UnitAttr is_out_of_order,
                                         mlir::UnitAttr is_critical) {
    build(builder, state, input, actCompressionSizeEntryBuff, output_buff, /*port=*/port, /*channelType=*/nullptr,
          /*is_out_of_order=*/is_out_of_order, /*is_critical=*/is_critical,
          /* dma_hwp_id= */ nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::DecompressDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(input());
    const auto dstOff = writer.getTensorRef(output_buff());

    MVCNN::NNDMATaskBuilder builder(writer);

    if (act_compression_size_entry()) {
        const auto act_compression_size_entryOff = writer.getTensorRef(act_compression_size_entry());
        builder.add_act_compression_size_entry(act_compression_size_entryOff);
        // In case of Activation Spill Decompression: use DMA to decompress the activation.
        builder.add_compression(true);
    } else {
        // In case of BitCompactor weight compression: use BitCompactor to compress weights and the DMA will de-compress
        // the weights
        builder.add_compression(true);
    }

    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_dma_hwp_id(this->dma_hwp_id().value_or(0));
    builder.add_port(checked_cast<uint8_t>(port()));
    builder.add_set_ord(static_cast<uint8_t>(!is_out_of_order()));  // ORD
    builder.add_set_crit(static_cast<uint8_t>(is_critical()));      // CRIT
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::DecompressDMAOp::verify() {
    auto loc = getLoc();
    mlir::LogicalResult ret = mlir::success();
    if (ret.succeeded())
        ret = verifyInOutElementType(loc, input(), output());

    if (ret.succeeded())
        ret = verifyTensorSize(loc, input());

    return ret;
}

//
// CompressDMAOp
//

void vpux::VPUIP::CompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                       mlir::Value actCompressionSizeEntryBuff, mlir::Value output_buff, int64_t port,
                                       bool is_out_of_order, bool is_critical) {
    build(builder, state, input, actCompressionSizeEntryBuff, output_buff, /*port=*/port, /*channelType=*/nullptr,
          /*is_out_of_order=*/is_out_of_order, /*is_critical=*/is_critical,
          /* dma_hwp_id= */ nullptr);
}

void vpux::VPUIP::CompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                       mlir::Value actCompressionSizeEntryBuff, mlir::Value output_buff,
                                       mlir::IntegerAttr port, /*optional*/ mlir::UnitAttr is_out_of_order,
                                       /*optional*/ mlir::UnitAttr is_critical) {
    build(builder, state, input, actCompressionSizeEntryBuff, output_buff, /*port=*/port, /*channelType=*/nullptr,
          /*is_out_of_order=*/is_out_of_order, /*is_critical=*/is_critical,
          /* dma_hwp_id= */ nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::CompressDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(input());
    const auto dstOff = writer.getTensorRef(output_buff());
    const auto act_compression_size_entryOff = writer.getTensorRef(act_compression_size_entry());

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_act_compression_size_entry(act_compression_size_entryOff);
    // In case of Activation Spill Compression: use DMA to compress the activation.
    builder.add_compression(true);
    builder.add_port(checked_cast<uint8_t>(port()));
    builder.add_dma_hwp_id(this->dma_hwp_id().value_or(0));
    builder.add_set_ord(static_cast<uint8_t>(!is_out_of_order()));  // ORD
    builder.add_set_crit(static_cast<uint8_t>(is_critical()));      // CRIT
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::CompressDMAOp::verify() {
    auto loc = getLoc();
    mlir::LogicalResult ret = mlir::success();
    if (ret.succeeded())
        ret = verifyInOutElementType(loc, input(), output());

    if (ret.succeeded())
        ret = verifyTensorSize(loc, input());

    return ret;
}

//
// DepthToSpaceDMAOp
//

void vpux::VPUIP::DepthToSpaceDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                           mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr block_size,
                                           vpux::IE::DepthToSpaceModeAttr mode, VPUIP::DMADescriptorAttr dma_descriptor,
                                           vpux::IE::ChannelPaddingAttr padded_channels) {
    build(odsBuilder, odsState, input, output_buff, /* port= */ vpux::getIntAttr(odsBuilder, 0),
          /*channelType=*/nullptr, block_size, mode, dma_descriptor, padded_channels,
          /* dma_hwp_id= */ nullptr);
}

void vpux::VPUIP::DepthToSpaceDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                           mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr block_size,
                                           vpux::IE::DepthToSpaceModeAttr mode, VPUIP::DMADescriptorAttr dma_descriptor,
                                           mlir::IntegerAttr port, vpux::IE::ChannelPaddingAttr padded_channels) {
    build(odsBuilder, odsState, input, output_buff, port, /*channelType=*/nullptr, block_size, mode, dma_descriptor,
          padded_channels,
          /* dma_hwp_id= */ nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::DepthToSpaceDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(input());
    const auto dstOff = writer.getTensorRef(output_buff());
    const auto descriptor = writer.getDepthToSpaceNNDMADescriptorReference(getOperation());

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_port(checked_cast<uint8_t>(port()));
    builder.add_dma_hwp_id(this->dma_hwp_id().value_or(0));
    builder.add_set_ord(static_cast<uint8_t>(true));    // ORD
    builder.add_set_crit(static_cast<uint8_t>(false));  // CRIT
    builder.add_descriptor(&descriptor);
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::DepthToSpaceDMAOp::verify() {
    return verifyTensorSize(getLoc(), input());
}

//
// SpaceToDepthDMAOp
//
void vpux::VPUIP::SpaceToDepthDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                           mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr block_size,
                                           vpux::IE::SpaceToDepthModeAttr mode,
                                           VPUIP::DMADescriptorAttr dma_descriptor) {
    build(odsBuilder, odsState, input, output_buff, vpux::getIntAttr(odsBuilder, 0), /*channelType=*/nullptr,
          block_size, mode, dma_descriptor,
          /* dma_hwp_id= */ nullptr);
}

void vpux::VPUIP::SpaceToDepthDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                           mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr block_size,
                                           vpux::IE::SpaceToDepthModeAttr mode, VPUIP::DMADescriptorAttr dma_descriptor,
                                           mlir::IntegerAttr port) {
    build(odsBuilder, odsState, input, output_buff, port, /*channelType=*/nullptr, block_size, mode, dma_descriptor,
          /* dma_hwp_id= */ nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SpaceToDepthDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(input());
    const auto dstOff = writer.getTensorRef(output_buff());
    const auto descriptor = writer.getSpaceToDepthNNDMADescriptorReference(getOperation());

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_port(checked_cast<uint8_t>(port()));
    builder.add_dma_hwp_id(this->dma_hwp_id().value_or(0));
    builder.add_set_ord(static_cast<uint8_t>(true));    // ORD
    builder.add_set_crit(static_cast<uint8_t>(false));  // CRIT
    builder.add_descriptor(&descriptor);
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::SpaceToDepthDMAOp::verify() {
    return verifyTensorSize(getLoc(), input());
}

//
// ExpandDMAOp
//

void vpux::VPUIP::ExpandDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     mlir::Value output_buff, mlir::ArrayAttr pads_begin, mlir::ArrayAttr pads_end,
                                     VPUIP::DMADescriptorAttr dma_descriptor) {
    build(builder, state, input, output_buff, pads_begin, pads_end, dma_descriptor,
          /*port=*/vpux::getIntAttr(builder, 0), /*channelType=*/nullptr,
          /*is_out_of_order=*/nullptr,
          /*is_critical=*/nullptr,
          /* dma_hwp_id= */ nullptr);
}

void vpux::VPUIP::ExpandDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     mlir::Value output_buff, mlir::ArrayAttr pads_begin, mlir::ArrayAttr pads_end,
                                     VPUIP::DMADescriptorAttr dma_descriptor, mlir::IntegerAttr port) {
    build(builder, state, input, output_buff, pads_begin, pads_end, dma_descriptor, /*port=*/port,
          /*channelType=*/nullptr,
          /*is_out_of_order=*/nullptr,
          /*is_critical=*/nullptr,
          /* dma_hwp_id= */ nullptr);
}

mlir::LogicalResult vpux::VPUIP::ExpandDMAOp::verify() {
    // In case ExpandDMA with input size large than VPUIP::DMA_LIMIT (16MB).
    // It should be tiled with several sub ExpandDMA that will be done at Unroll Pass.
    // Descriptor is generated at Unroll pass so using Descriptor as a flag to check the tensor size.
    if (dma_descriptor().has_value()) {
        return verifyTensorSize(getLoc(), input());
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
    builder.add_dma_hwp_id(this->dma_hwp_id().value_or(0));
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
                                          mlir::IntegerAttr tiles, VPUIP::DMADescriptorAttr dma_descriptor) {
    build(odsBuilder, odsState, input, output_buff, vpux::getIntAttr(odsBuilder, 0), /*channelType=*/nullptr, axis,
          tiles, dma_descriptor,
          /* dma_hwp_id= */ nullptr);
}

void vpux::VPUIP::PerAxisTileDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                          mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr axis,
                                          mlir::IntegerAttr tiles, VPUIP::DMADescriptorAttr dma_descriptor,
                                          mlir::IntegerAttr port) {
    build(odsBuilder, odsState, input, output_buff, port, /*channelType=*/nullptr, axis, tiles, dma_descriptor,
          /* dma_hwp_id= */ nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::PerAxisTileDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(input());
    const auto dstOff = writer.getTensorRef(output_buff());
    const auto descriptor = writer.getPerAxisTileNNDMADescriptorReference(getOperation());

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_port(checked_cast<uint8_t>(port()));
    builder.add_dma_hwp_id(this->dma_hwp_id().value_or(0));
    builder.add_set_ord(static_cast<uint8_t>(true));    // ORD
    builder.add_set_crit(static_cast<uint8_t>(false));  // CRIT
    builder.add_descriptor(&descriptor);
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::PerAxisTileDMAOp::verify() {
    return verifyTensorSize(getLoc(), input());
}

//
// UpsamplingDMAOp
//

void vpux::VPUIP::UpsamplingDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value input,
                                         mlir::Value output_buff, mlir::ArrayAttr upsampling_factor,
                                         VPUIP::DMADescriptorAttr dma_descriptor, mlir::ArrayAttr expand) {
    build(odsBuilder, odsState, input, output_buff, upsampling_factor, dma_descriptor, expand,
          /* port */ vpux::getIntAttr(odsBuilder, 0), /*channelType=*/nullptr, /* dma_hwp_id */ nullptr);
}

void vpux::VPUIP::UpsamplingDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value input,
                                         mlir::Value output_buff, mlir::ArrayAttr upsampling_factor,
                                         VPUIP::DMADescriptorAttr dma_descriptor, mlir::ArrayAttr expand,
                                         int64_t port) {
    build(odsBuilder, odsState, input, output_buff, upsampling_factor, dma_descriptor, expand,
          /* port */ port, /*channelType=*/nullptr, /* dma_hwp_id */ nullptr);
}

mlir::LogicalResult vpux::VPUIP::UpsamplingDMAOp::verify() {
    return verifyTensorSize(getLoc(), input());
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::UpsamplingDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(input());
    const auto dstOff = writer.getTensorRef(output_buff());
    const auto descriptor = writer.getUpsamplingNNDMADescriptorReference(getOperation());

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_port(checked_cast<uint8_t>(port()));
    builder.add_dma_hwp_id(this->dma_hwp_id().value_or(0));
    builder.add_set_ord(static_cast<uint8_t>(true));    // ORD
    builder.add_set_crit(static_cast<uint8_t>(false));  // CRIT
    builder.add_descriptor(&descriptor);
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}
