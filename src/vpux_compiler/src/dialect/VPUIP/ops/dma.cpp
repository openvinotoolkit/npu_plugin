//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"

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
    build(builder, state, input, output_buff, /*port=*/nullptr, /*is_out_of_order=*/false,
          /*is_critical=*/false,
          /*spillId=*/nullptr, /*compress_candidate*/ nullptr, /*dma_hwp_id*/ nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::NNDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                 mlir::Value output_buff, int64_t port) {
    build(builder, state, input, output_buff, /*port=*/vpux::getIntAttr(builder, port),
          /*is_out_of_order=*/false,
          /*is_critical=*/false,
          /*spillId=*/nullptr, /*compress_candidate*/ nullptr, /*dma_hwp_id*/ nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::NNDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                 mlir::Value output_buff, int64_t port, int32_t dma_hwp_id) {
    build(builder, state, input, output_buff, /*port=*/vpux::getIntAttr(builder, port),
          /*is_out_of_order=*/false,
          /*is_critical=*/false,
          /*spillId=*/nullptr, /*compress_candidate*/ nullptr, vpux::getIntAttr(builder, dma_hwp_id),
          /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::NNDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                 mlir::Value output_buff, mlir::IntegerAttr port, mlir::UnitAttr is_out_of_order,
                                 mlir::UnitAttr is_critical, mlir::IntegerAttr spillId) {
    build(builder, state, input, output_buff, /*port=*/port,
          /*is_out_of_order=*/is_out_of_order,
          /*is_critical=*/is_critical, /*spillId=*/spillId, /*compress_candidate*/ nullptr, /*dma_hwp_id*/ nullptr,
          /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::NNDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                 mlir::Value output_buff, int64_t port, bool is_out_of_order, bool is_critical,
                                 mlir::IntegerAttr spillId, mlir::UnitAttr compress_candidate) {
    build(builder, state, input, output_buff, /*port=*/vpux::getIntAttr(builder, port),
          /*is_out_of_order=*/is_out_of_order,
          /*is_critical=*/is_critical, /*spillId=*/spillId, /*compress_candidate*/ compress_candidate,
          /*dma_hwp_id*/ nullptr,
          /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::NNDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                 mlir::Value output_buff, int64_t port, bool is_out_of_order, bool is_critical,
                                 mlir::IntegerAttr spillId) {
    build(builder, state, input, output_buff, /*port=*/vpux::getIntAttr(builder, port),
          /*is_out_of_order=*/is_out_of_order,
          /*is_critical=*/is_critical, /*spillId=*/spillId, /*compress_candidate*/ nullptr, /*dma_hwp_id*/ nullptr,
          /* profilingMetadata */ nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::NNDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(getInput());
    const auto dstOff = writer.getTensorRef(getOutputBuff());

    const auto port = getPort();
    VPUX_THROW_UNLESS(port.has_value(), "DMA port has not been set");
    const auto portValue = port.value();

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_dma_hwp_id(checked_cast<int32_t>(this->getDmaHwpId().value_or(0)));
    builder.add_port(checked_cast<uint8_t>(portValue));
    builder.add_set_ord(static_cast<uint8_t>(!getIsOutOfOrder()));  // ORD
    builder.add_set_crit(static_cast<uint8_t>(getIsCritical()));    // CRIT
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::NNDMAOp::verify() {
    auto loc = getLoc();
    return verifyTensorSize(loc, getInput());
}

size_t vpux::VPUIP::NNDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();

    // TODO: Expose API to get arch from cost model
    const auto arch = VPU::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel));
}

//
// PermuteDMAOp
//

void vpux::VPUIP::PermuteDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                      mlir::Value output_buff, mlir::AffineMapAttr mem_perm,
                                      VPUIP::DMADescriptorAttr dma_descriptor) {
    build(builder, state, input, output_buff, /*port=*/nullptr,
          /*is_out_of_order=*/nullptr,
          /*is_critical=*/nullptr, mem_perm, dma_descriptor, /* dma_hwp_id= */ nullptr,
          /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::PermuteDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                      mlir::Value output_buff, mlir::AffineMapAttr mem_perm,
                                      VPUIP::DMADescriptorAttr dma_descriptor, mlir::IntegerAttr port) {
    build(builder, state, input, output_buff, /*port=*/port,
          /*is_out_of_order=*/nullptr,
          /*is_critical=*/nullptr, mem_perm, dma_descriptor, nullptr, /* profilingMetadata */ nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::PermuteDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(getInput());
    const auto dstOff = writer.getTensorRef(getOutputBuff());
    const auto descriptor = writer.getPermuteNNDMADescriptorReference(getOperation());

    const auto port = getPort();
    VPUX_THROW_UNLESS(port.has_value(), "DMA port has not been set");
    const auto portValue = port.value();

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_port(checked_cast<uint8_t>(portValue));
    builder.add_dma_hwp_id(checked_cast<int32_t>(this->getDmaHwpId().value_or(0)));
    builder.add_set_ord(static_cast<uint8_t>(!getIsOutOfOrder()));  // ORD
    builder.add_set_crit(static_cast<uint8_t>(getIsCritical()));    // CRIT
    builder.add_descriptor(&descriptor);
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::PermuteDMAOp::verify() {
    return verifyTensorSize(getLoc(), getInput());
}

size_t vpux::VPUIP::PermuteDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();

    // TODO: E#97004 Expose API to get arch from cost model
    // TODO: E#89933 resolved, complex DMAs PermuteDMAOp will need to be handled by more detailed method than
    // getDMACost()
    const auto arch = VPU::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel));
}

//
// ConvertDMAOp
//

void vpux::VPUIP::ConvertDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                      mlir::Value output_buff) {
    build(builder, state, input, output_buff, /*port=*/nullptr, /*is_out_of_order=*/false,
          /*is_critical=*/false,
          /* dma_hwp_id=*/nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::ConvertDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                      mlir::Value output_buff, int64_t port) {
    build(builder, state, input, output_buff, /*port=*/vpux::getIntAttr(builder, port),
          /*is_out_of_order=*/false,
          /*is_critical=*/false,
          /* dma_hwp_id=*/nullptr, /* profilingMetadata */ nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ConvertDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(getInput());
    const auto dstOff = writer.getTensorRef(getOutputBuff());

    const auto port = getPort();
    VPUX_THROW_UNLESS(port.has_value(), "DMA port has not been set");
    const auto portValue = port.value();

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_dma_hwp_id(checked_cast<int32_t>(this->getDmaHwpId().value_or(0)));
    builder.add_port(checked_cast<uint8_t>(portValue));
    builder.add_set_ord(static_cast<uint8_t>(!getIsOutOfOrder()));  // ORD
    builder.add_set_crit(static_cast<uint8_t>(getIsCritical()));    // CRIT
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::ConvertDMAOp::verify() {
    auto loc = getLoc();
    auto arch = VPU::getArch(getOperation());

    // Skip checks if architecture is unknown since all of them depend on the architecture used
    if (arch == VPU::ArchKind::UNKNOWN) {
        return mlir::success();
    }

    auto outputType = getOutputBuff().getType().cast<vpux::NDTypeInterface>();
    const auto outputElementType = outputType.getElementType();
    auto inputType = getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inputElementType = inputType.getElementType();

    if ((arch == VPU::ArchKind::VPUX37XX || arch == VPU::ArchKind::VPUX30XX) || !inputElementType.isF32() ||
        (!outputElementType.isF16() && !outputElementType.isBF16())) {
        return errorAt(loc,
                       "Operation {0} is only supported for F32 to F16/BF16 conversion and not supported for VPUX37XX "
                       "or VPUX30XX arch. "
                       "Got arch {1} "
                       "and conversion from {2} to {3}",
                       getOperationName(), arch, inputElementType, outputElementType);
    }

    return verifyTensorSize(loc, getInput());
}

size_t vpux::VPUIP::ConvertDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();

    // TODO: E#97004 Expose API to get arch from cost model
    // TODO: E#89933 resolved, complex DMAs ConvertDMAOp will need to be handled by more detailed method than
    // getDMACost()
    const auto arch = VPU::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel));
}

//
// DecompressDMAOp
//

void vpux::VPUIP::DecompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value output_buff, int64_t port, bool is_out_of_order,
                                         bool is_critical) {
    build(builder, state, input, /*act_compression_size_entry=*/nullptr, output_buff,
          /*port=*/vpux::getIntAttr(builder, port),

          /*is_out_of_order=*/is_out_of_order, /*is_critical=*/is_critical, /* dma_hwp_id= */ nullptr,
          /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::DecompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value output_buff, mlir::IntegerAttr port,
                                         mlir::UnitAttr is_out_of_order, mlir::UnitAttr is_critical) {
    build(builder, state, input, /*act_compression_size_entry=*/nullptr, output_buff, /*port=*/port,

          /*is_out_of_order=*/is_out_of_order, /*is_critical=*/is_critical,
          /*dma_hwp_id=*/nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::DecompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value output_buff) {
    build(builder, state, input, /*act_compression_size_entry=*/nullptr, output_buff, /*port=*/nullptr,

          /*is_out_of_order=*/false, /*is_critical=*/false,
          /*dma_hwp_id=*/nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::DecompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value output_buff, int64_t port) {
    build(builder, state, input, output_buff, /*port=*/port, /*is_out_of_order=*/false,
          /*is_critical=*/false);
}

void vpux::VPUIP::DecompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value actCompressionSizeEntryBuff, mlir::Value output_buff,
                                         int64_t port) {
    build(builder, state, input, actCompressionSizeEntryBuff, output_buff, /*port=*/vpux::getIntAttr(builder, port),

          /*is_out_of_order=*/false, /*is_critical=*/false,
          /* dma_hwp_id= */ nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::DecompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value actCompressionSizeEntryBuff, mlir::Value output_buff, int64_t port,
                                         bool is_out_of_order, bool is_critical) {
    build(builder, state, input, actCompressionSizeEntryBuff, output_buff, /*port=*/vpux::getIntAttr(builder, port),

          /*is_out_of_order=*/is_out_of_order, /*is_critical=*/is_critical,
          /* dma_hwp_id= */ nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::DecompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value actCompressionSizeEntryBuff, mlir::Value output_buff,
                                         mlir::IntegerAttr port, mlir::UnitAttr is_out_of_order,
                                         mlir::UnitAttr is_critical) {
    build(builder, state, input, actCompressionSizeEntryBuff, output_buff, /*port=*/port,
          /*is_out_of_order=*/is_out_of_order, /*is_critical=*/is_critical,
          /* dma_hwp_id= */ nullptr, /* profilingMetadata */ nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::DecompressDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(getInput());
    const auto dstOff = writer.getTensorRef(getOutputBuff());

    MVCNN::NNDMATaskBuilder builder(writer);

    if (getActCompressionSizeEntry()) {
        const auto act_compression_size_entryOff = writer.getTensorRef(getActCompressionSizeEntry());
        builder.add_act_compression_size_entry(act_compression_size_entryOff);
        // In case of Activation Spill Decompression: use DMA to decompress the activation.
        builder.add_compression(true);
    } else {
        // In case of BitCompactor weight compression: use BitCompactor to compress weights and the DMA will de-compress
        // the weights
        builder.add_compression(true);
    }

    const auto port = getPort();
    VPUX_THROW_UNLESS(port.has_value(), "DMA port has not been set");
    const auto portValue = port.value();

    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_dma_hwp_id(checked_cast<int32_t>(this->getDmaHwpId().value_or(0)));
    builder.add_port(checked_cast<uint8_t>(portValue));
    builder.add_set_ord(static_cast<uint8_t>(!getIsOutOfOrder()));  // ORD
    builder.add_set_crit(static_cast<uint8_t>(getIsCritical()));    // CRIT
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::DecompressDMAOp::verify() {
    auto loc = getLoc();
    mlir::LogicalResult ret = mlir::success();
    if (ret.succeeded())
        ret = verifyInOutElementType(loc, getInput(), getOutput());

    if (ret.succeeded())
        ret = verifyTensorSize(loc, getInput());

    return ret;
}

size_t vpux::VPUIP::DecompressDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();

    // TODO: E#97004 Expose API to get arch from cost model
    // TODO: E#89933 resolved, complex DMAs DecompressDMAOp will need to be handled by more detailed method than
    // getDMACost()
    const auto arch = VPU::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel));
}

//
// CompressDMAOp
//

void vpux::VPUIP::CompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                       mlir::Value actCompressionSizeEntryBuff, mlir::Value output_buff, int64_t port) {
    build(builder, state, input, actCompressionSizeEntryBuff, output_buff, /*port=*/vpux::getIntAttr(builder, port),

          /*is_out_of_order=*/false, /*is_critical=*/false,
          /* dma_hwp_id= */ nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::CompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                       mlir::Value actCompressionSizeEntryBuff, mlir::Value output_buff, int64_t port,
                                       bool is_out_of_order, bool is_critical) {
    build(builder, state, input, actCompressionSizeEntryBuff, output_buff, /*port=*/vpux::getIntAttr(builder, port),

          /*is_out_of_order=*/is_out_of_order, /*is_critical=*/is_critical,
          /* dma_hwp_id= */ nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::CompressDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                       mlir::Value actCompressionSizeEntryBuff, mlir::Value output_buff,
                                       mlir::IntegerAttr port, /*optional*/ mlir::UnitAttr is_out_of_order,
                                       /*optional*/ mlir::UnitAttr is_critical) {
    build(builder, state, input, actCompressionSizeEntryBuff, output_buff, /*port=*/port,
          /*is_out_of_order=*/is_out_of_order, /*is_critical=*/is_critical,
          /* dma_hwp_id= */ nullptr, /* profilingMetadata */ nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::CompressDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(getInput());
    const auto dstOff = writer.getTensorRef(getOutputBuff());
    const auto act_compression_size_entryOff = writer.getTensorRef(getActCompressionSizeEntry());

    const auto port = getPort();
    VPUX_THROW_UNLESS(port.has_value(), "DMA port has not been set");
    const auto portValue = port.value();

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_act_compression_size_entry(act_compression_size_entryOff);
    // In case of Activation Spill Compression: use DMA to compress the activation.
    builder.add_compression(true);
    builder.add_port(checked_cast<uint8_t>(portValue));
    builder.add_dma_hwp_id(checked_cast<int32_t>(this->getDmaHwpId().value_or(0)));
    builder.add_set_ord(static_cast<uint8_t>(!getIsOutOfOrder()));  // ORD
    builder.add_set_crit(static_cast<uint8_t>(getIsCritical()));    // CRIT
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::CompressDMAOp::verify() {
    auto loc = getLoc();
    mlir::LogicalResult ret = mlir::success();
    if (ret.succeeded())
        ret = verifyInOutElementType(loc, getInput(), getOutput());

    if (ret.succeeded())
        ret = verifyTensorSize(loc, getInput());

    return ret;
}

size_t vpux::VPUIP::CompressDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();

    // TODO: E#97004 Expose API to get arch from cost model
    // TODO: E#89933 resolved, complex DMAs CompressDMAOp will need to be handled by more detailed method than
    // getDMACost()
    const auto arch = VPU::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel));
}

//
// DepthToSpaceDMAOp
//

void vpux::VPUIP::DepthToSpaceDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                           mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr block_size,
                                           vpux::IE::DepthToSpaceModeAttr mode, VPUIP::DMADescriptorAttr dma_descriptor,
                                           vpux::IE::ChannelPaddingAttr padded_channels) {
    build(odsBuilder, odsState, input, output_buff, /* port= */ nullptr, block_size, mode, dma_descriptor,
          padded_channels,
          /*is_out_of_order=*/nullptr, /*is_critical=*/nullptr, /* dma_hwp_id= */ nullptr,
          /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::DepthToSpaceDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                           mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr block_size,
                                           vpux::IE::DepthToSpaceModeAttr mode, VPUIP::DMADescriptorAttr dma_descriptor,
                                           mlir::IntegerAttr port, vpux::IE::ChannelPaddingAttr padded_channels) {
    build(odsBuilder, odsState, input, output_buff, port, block_size, mode, dma_descriptor, padded_channels,
          /*is_out_of_order=*/nullptr, /*is_critical=*/nullptr, /* dma_hwp_id= */ nullptr,
          /* profilingMetadata */ nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::DepthToSpaceDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(getInput());
    const auto dstOff = writer.getTensorRef(getOutputBuff());
    const auto descriptor = writer.getDepthToSpaceNNDMADescriptorReference(getOperation());

    const auto port = getPort();
    VPUX_THROW_UNLESS(port.has_value(), "DMA port has not been set");
    const auto portValue = port.value();

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_port(checked_cast<uint8_t>(portValue));
    builder.add_dma_hwp_id(checked_cast<int32_t>(this->getDmaHwpId().value_or(0)));
    builder.add_set_ord(static_cast<uint8_t>(!getIsOutOfOrder()));  // ORD
    builder.add_set_crit(static_cast<uint8_t>(getIsCritical()));    // CRIT
    builder.add_descriptor(&descriptor);
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::DepthToSpaceDMAOp::verify() {
    return verifyTensorSize(getLoc(), getInput());
}

size_t vpux::VPUIP::DepthToSpaceDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();

    // TODO: E#97004 Expose API to get arch from cost model
    // TODO: E#89933 resolved, complex DMAs DepthToSpaceDMAOp will need to be handled by more detailed method than
    // getDMACost()
    const auto arch = VPU::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel));
}

//
// SpaceToDepthDMAOp
//
void vpux::VPUIP::SpaceToDepthDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                           mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr block_size,
                                           vpux::IE::SpaceToDepthModeAttr mode,
                                           VPUIP::DMADescriptorAttr dma_descriptor) {
    build(odsBuilder, odsState, input, output_buff, nullptr, block_size, mode, dma_descriptor,
          /*is_out_of_order=*/nullptr, /*is_critical=*/nullptr, /* dma_hwp_id= */ nullptr,
          /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::SpaceToDepthDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                           mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr block_size,
                                           vpux::IE::SpaceToDepthModeAttr mode, VPUIP::DMADescriptorAttr dma_descriptor,
                                           mlir::IntegerAttr port) {
    build(odsBuilder, odsState, input, output_buff, port, block_size, mode, dma_descriptor,
          /*is_out_of_order=*/nullptr, /*is_critical=*/nullptr, /* dma_hwp_id= */ nullptr,
          /* profilingMetadata */ nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SpaceToDepthDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(getInput());
    const auto dstOff = writer.getTensorRef(getOutputBuff());
    const auto descriptor = writer.getSpaceToDepthNNDMADescriptorReference(getOperation());

    const auto port = getPort();
    VPUX_THROW_UNLESS(port.has_value(), "DMA port has not been set");
    const auto portValue = port.value();

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_port(checked_cast<uint8_t>(portValue));
    builder.add_dma_hwp_id(checked_cast<int32_t>(this->getDmaHwpId().value_or(0)));
    builder.add_set_ord(static_cast<uint8_t>(!getIsOutOfOrder()));  // ORD
    builder.add_set_crit(static_cast<uint8_t>(getIsCritical()));    // CRIT
    builder.add_descriptor(&descriptor);
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::SpaceToDepthDMAOp::verify() {
    return verifyTensorSize(getLoc(), getInput());
}

size_t vpux::VPUIP::SpaceToDepthDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();

    // TODO: E#97004 Expose API to get arch from cost model
    // TODO: E#89933 resolved, complex DMAs SpaceToDepthDMAOp will need to be handled by more detailed method than
    // getDMACost()
    const auto arch = VPU::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel));
}

//
// ExpandDMAOp
//

void vpux::VPUIP::ExpandDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     mlir::Value output_buff, mlir::ArrayAttr pads_begin, mlir::ArrayAttr pads_end,
                                     VPUIP::DMADescriptorAttr dma_descriptor) {
    build(builder, state, input, output_buff, pads_begin, pads_end, dma_descriptor,
          /*port=*/nullptr,
          /*is_out_of_order=*/nullptr,
          /*is_critical=*/nullptr,
          /* dma_hwp_id= */ nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::ExpandDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     mlir::Value output_buff, mlir::ArrayAttr pads_begin, mlir::ArrayAttr pads_end,
                                     VPUIP::DMADescriptorAttr dma_descriptor, mlir::IntegerAttr port) {
    build(builder, state, input, output_buff, pads_begin, pads_end, dma_descriptor, /*port=*/port,

          /*is_out_of_order=*/nullptr,
          /*is_critical=*/nullptr,
          /* dma_hwp_id= */ nullptr, /* profilingMetadata */ nullptr);
}

mlir::LogicalResult vpux::VPUIP::ExpandDMAOp::verify() {
    // In case ExpandDMA with input size large than VPUIP::DMA_LIMIT (16MB).
    // It should be tiled with several sub ExpandDMA that will be done at Unroll Pass.
    // Descriptor is generated at Unroll pass so using Descriptor as a flag to check the tensor size.
    if (getDmaDescriptor().has_value()) {
        return verifyTensorSize(getLoc(), getInput());
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ExpandDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(getInput());
    const auto dstOff = writer.getTensorRef(getOutputBuff());
    const auto descriptor = writer.getExpandNNDMADescriptorReference(getOperation());

    const auto port = getPort();
    VPUX_THROW_UNLESS(port.has_value(), "DMA port has not been set");
    const auto portValue = port.value();

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_port(checked_cast<uint8_t>(portValue));
    builder.add_dma_hwp_id(checked_cast<int32_t>(this->getDmaHwpId().value_or(0)));
    builder.add_set_ord(static_cast<uint8_t>(!getIsOutOfOrder()));  // ORD
    builder.add_set_crit(static_cast<uint8_t>(getIsCritical()));    // CRIT
    builder.add_descriptor(&descriptor);
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

size_t vpux::VPUIP::ExpandDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();

    // TODO: E#97004 Expose API to get arch from cost model
    // TODO: E#89933 resolved, complex DMAs ExpandDMAOp will need to be handled by more detailed method than
    // getDMACost()
    const auto arch = VPU::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel));
}

//
// PerAxisTileDMAOp
//

void vpux::VPUIP::PerAxisTileDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                          mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr axis,
                                          mlir::IntegerAttr tiles, VPUIP::DMADescriptorAttr dma_descriptor) {
    build(odsBuilder, odsState, input, output_buff, nullptr, axis, tiles, dma_descriptor,
          /*is_out_of_order=*/nullptr, /*is_critical=*/nullptr, /* dma_hwp_id= */ nullptr,
          /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::PerAxisTileDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                          mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr axis,
                                          mlir::IntegerAttr tiles, VPUIP::DMADescriptorAttr dma_descriptor,
                                          mlir::IntegerAttr port) {
    build(odsBuilder, odsState, input, output_buff, port, axis, tiles, dma_descriptor,
          /*is_out_of_order=*/nullptr, /*is_critical=*/nullptr, /* dma_hwp_id= */ nullptr,
          /* profilingMetadata */ nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::PerAxisTileDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(getInput());
    const auto dstOff = writer.getTensorRef(getOutputBuff());
    const auto descriptor = writer.getPerAxisTileNNDMADescriptorReference(getOperation());

    const auto port = getPort();
    VPUX_THROW_UNLESS(port.has_value(), "DMA port has not been set");
    const auto portValue = port.value();

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_port(checked_cast<uint8_t>(portValue));
    builder.add_dma_hwp_id(checked_cast<int32_t>(this->getDmaHwpId().value_or(0)));
    builder.add_set_ord(static_cast<uint8_t>(!getIsOutOfOrder()));  // ORD
    builder.add_set_crit(static_cast<uint8_t>(getIsCritical()));    // CRIT
    builder.add_descriptor(&descriptor);
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::PerAxisTileDMAOp::verify() {
    return verifyTensorSize(getLoc(), getInput());
}

size_t vpux::VPUIP::PerAxisTileDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();

    // TODO: E#97004 Expose API to get arch from cost model
    // TODO: E#89933 resolved, complex DMAs PerAxisTileDMAOp will need to be handled by more detailed method than
    // getDMACost()
    const auto arch = VPU::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel));
}

//
// UpsamplingDMAOp
//

void vpux::VPUIP::UpsamplingDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value input,
                                         mlir::Value output_buff, mlir::ArrayAttr upsampling_factor,
                                         VPUIP::DMADescriptorAttr dma_descriptor, mlir::ArrayAttr expand) {
    build(odsBuilder, odsState, input, output_buff, upsampling_factor, dma_descriptor, expand,
          /* port */ nullptr, /*is_out_of_order=*/nullptr,
          /*is_critical=*/nullptr, /* dma_hwp_id */ nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::UpsamplingDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value input,
                                         mlir::Value output_buff, mlir::ArrayAttr upsampling_factor,
                                         VPUIP::DMADescriptorAttr dma_descriptor, mlir::ArrayAttr expand,
                                         int64_t port) {
    build(odsBuilder, odsState, input, output_buff, upsampling_factor, dma_descriptor, expand,
          /* port */ vpux::getIntAttr(odsBuilder, port), /*is_out_of_order=*/false,
          /*is_critical=*/false,
          /* dma_hwp_id */ nullptr, /* profilingMetadata */ nullptr);
}

void vpux::VPUIP::UpsamplingDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value input,
                                         mlir::Value output_buff, mlir::ArrayAttr upsampling_factor,
                                         VPUIP::DMADescriptorAttr dma_descriptor, mlir::ArrayAttr expand, int64_t port,
                                         bool is_out_of_order, bool is_critical, mlir::IntegerAttr dma_hwp_id,
                                         VPUIP::DmaProfilingMetadataAttr profilingMetadata) {
    build(odsBuilder, odsState, input, output_buff, upsampling_factor, dma_descriptor, expand,
          /* port */ vpux::getIntAttr(odsBuilder, port),
          /*is_out_of_order=*/is_out_of_order,
          /*is_critical=*/is_critical,
          /* dma_hwp_id */ dma_hwp_id, /* profilingMetadata */ profilingMetadata);
}

mlir::LogicalResult vpux::VPUIP::UpsamplingDMAOp::verify() {
    // In case UpsamplingDMA with input size large than VPUIP::DMA_LIMIT (16MB).
    // It should be tiled with several sub UpsamplingDMA that will be done at Unroll Pass.
    // Descriptor is generated at Unroll pass so using Descriptor as a flag to check the tensor size.
    if (getDmaDescriptor().has_value()) {
        return verifyTensorSize(getLoc(), getInput());
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::UpsamplingDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensorRef(getInput());
    const auto dstOff = writer.getTensorRef(getOutputBuff());
    const auto descriptor = writer.getUpsamplingNNDMADescriptorReference(getOperation());

    const auto port = getPort();
    VPUX_THROW_UNLESS(port.has_value(), "DMA port has not been set");
    const auto portValue = port.value();

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_port(checked_cast<uint8_t>(portValue));
    builder.add_dma_hwp_id(checked_cast<int32_t>(this->getDmaHwpId().value_or(0)));
    builder.add_set_ord(static_cast<uint8_t>(!getIsOutOfOrder()));  // ORD
    builder.add_set_crit(static_cast<uint8_t>(getIsCritical()));    // CRIT
    builder.add_descriptor(&descriptor);
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

size_t vpux::VPUIP::UpsamplingDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();

    // TODO: E#97004 Expose API to get arch from cost model
    // TODO: E#89933 resolved, complex DMAs UpsamplingDMAOp will need to be handled by more detailed method than
    // getDMACost()
    const auto arch = VPU::getArch(module);
    return checked_cast<size_t>(getDMACost(getInput(), getOutput(), arch, costModel));
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SyncDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto port = getPort();
    VPUX_THROW_UNLESS(port.has_value(), "DMA port has not been set");
    const auto portValue = port.value();

    MVCNN::NNDMATaskBuilder builder(writer);
    const auto srcOff = writer.getTensorRef(getInput());
    const auto dstOff = writer.getTensorRef(getOutputBuff());
    const auto descriptor = writer.getSyncNNDMADescriptorReference(getOperation());
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_port(checked_cast<uint8_t>(portValue));
    builder.add_dma_hwp_id(this->getDmaHwpId().value_or(0));
    builder.add_set_ord(static_cast<uint8_t>(!getIsOutOfOrder()));  // ORD
    builder.add_set_crit(static_cast<uint8_t>(getIsCritical()));    // CRIT
    builder.add_descriptor(&descriptor);
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}

mlir::LogicalResult vpux::VPUIP::SyncDMAOp::verify() {
    auto loc = getLoc();
    const auto inSize = static_cast<Byte>(getCompactSize(getInput()));
    if (inSize.count() != 0) {
        return errorAt(loc, "input size should be zero {0}", inSize);
    }
    const auto outSize = static_cast<Byte>(getCompactSize(getResult()));
    if (outSize.count() != 0) {
        return errorAt(loc, "output size should be zero {0}", outSize);
    }
    return mlir::success();
}

size_t vpux::VPUIP::SyncDMAOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>&) {
    return 0;
}
