//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/VPUIPRegMapped/host_parsing/host_parsed_inference.h"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"

using namespace vpux;

//
// NNDMAOp
//

// For further development, please refer to the ticket E#36225.

namespace {

llvm::SmallVector<std::pair<uint32_t, int32_t>> reduce_dims_for_dma(mlir::Value val) {
    auto ndType = val.getType().cast<vpux::NDTypeInterface>();
    const auto memShape = ndType.getMemShape();
    const auto memStrides = ndType.getMemStrides();

    auto inner_most_index = memShape.size() - 1;
    llvm::SmallVector<std::pair<uint32_t, int32_t>> finalDims;

    auto previous_size = checked_cast<uint32_t>(memShape[MemDim(inner_most_index)]);
    auto previous_stride_bits = checked_cast<uint32_t>(vpux::Bit(memStrides[MemDim(inner_most_index)]).count());

    if (previous_size * ndType.getElemTypeSize().count() < previous_stride_bits) {
        int32_t final_stride = previous_stride_bits / CHAR_BIT;
        uint32_t final_size = previous_size * ndType.getElemTypeSize().count() / CHAR_BIT;

        finalDims.push_back({final_size, final_stride});
    }

    // TODO: Could there be some way to iterate over all MemDim's of a particular shape/order?
    //       Please refer to the ticket E#36225.
    for (size_t dim = inner_most_index - 1; dim > 0; --dim) {
        auto memDim = MemDim(dim);

        auto current_size = checked_cast<uint32_t>(memShape[memDim]);
        auto current_stride_bits = checked_cast<uint32_t>(vpux::Bit(memStrides[memDim]).count());

        if (previous_size * previous_stride_bits < current_stride_bits) {
            int32_t final_stride = current_stride_bits / CHAR_BIT;
            uint32_t final_size = (previous_size * previous_stride_bits) / CHAR_BIT;

            finalDims.push_back({final_size, final_stride});
        }

        previous_size = current_size;
        previous_stride_bits = current_stride_bits;
    }

    if (finalDims.size() == 0) {
        uint32_t final_size = (previous_size * previous_stride_bits) / CHAR_BIT;
        int32_t final_stride = final_size;
        finalDims.push_back({final_size, final_stride});
    }

    return finalDims;
}
}  // namespace

void vpux::VPUIPRegMapped::NNDMAOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    host_parsing::DmaWrapper dmaTask;

    // safe init to zero the structure
    memset(reinterpret_cast<void*>(&dmaTask), 0, sizeof(dmaTask));

    auto inputType = input().getType().cast<mlir::MemRefType>();

    dmaTask.start_after = checked_cast<uint16_t>(start_after());

    host_parsing::DmaDescriptor& descriptor = dmaTask.transaction;
    descriptor.cfg_link.cfg_bits.type = 1;
    descriptor.cfg_link.cfg_bits.burst_length = 16;
    descriptor.cfg_link.cfg_bits.barrier_en = 1;

    // If this DMA Op is not used by any other Op this is the last Op in DMA chain and link_address shall be zero

    vpux::VPUIPRegMapped::NNDMAOp dmaUser = nullptr;

    for (auto user : getResult().getUsers()) {
        auto newDmaUser = mlir::dyn_cast<vpux::VPUIPRegMapped::NNDMAOp>(user);

        if (newDmaUser) {
            VPUX_THROW_UNLESS(!dmaUser, "VPUIPRegMapped::NNDMAOp '{0}' at loc '{1}' has more than one DMA user",
                              getOperation()->getName(), getLoc());

            dmaUser = newDmaUser;
        }
    }

    if (dmaUser) {
        auto dmaOpIndex = dmaUser.getResult().getType().cast<VPUIPRegMapped::IndexType>();
        descriptor.link_address = static_cast<uint64_t>(dmaOpIndex.getValue());
    } else {
        descriptor.link_address = 0;
    }

    uint64_t cons_mask = 0;
    for (auto waitBarrier : waitBarriers()) {
        if (auto op = mlir::dyn_cast_or_null<VPUIPRegMapped::ConfigureBarrierOp>(waitBarrier.getDefiningOp())) {
            cons_mask |= static_cast<uint64_t>(1) << op.id();
        }
    }
    uint64_t prod_mask = 0;
    for (auto updateBarrier : updateBarriers()) {
        if (auto op = mlir::dyn_cast_or_null<VPUIPRegMapped::ConfigureBarrierOp>(updateBarrier.getDefiningOp())) {
            prod_mask |= static_cast<uint64_t>(1) << op.id();
        }
    }
    descriptor.barriers.cons_mask = cons_mask;
    descriptor.barriers.prod_mask = prod_mask;

    descriptor.length =
            checked_cast<uint32_t>(inputType.getNumElements() * vpux::Byte(vpux::getElemTypeSize(inputType)).count());

    // TODO: can we have this reduction at a pass at memref level? Need to place
    // some conditions on the DMA, and in some scenarios, may have to do 1*DMA -> n*DMA
    //      transaction rewrites.
    // Please refer to the ticket E#36225.
    auto reduced_dims_input = reduce_dims_for_dma(input());
    auto reduced_dims_output = reduce_dims_for_dma(output_buff());

    if (reduced_dims_input.size() > 2 || reduced_dims_output.size() > 2) {
        VPUX_THROW("cannot reduce dims to 2 for DMA; Reduced InSize: {0}, OutSize: {1}", reduced_dims_input.size(),
                   reduced_dims_output.size());
    }

    descriptor.attr2d.src_width = reduced_dims_input[0].first;
    descriptor.attr2d.src_stride = reduced_dims_input[0].second;
    descriptor.attr2d.dst_width = reduced_dims_output[0].first;
    descriptor.attr2d.dst_stride = reduced_dims_output[0].second;

    if (reduced_dims_input.size() == 2 && reduced_dims_output.size() == 2) {
        VPUX_THROW_UNLESS(reduced_dims_input[1].first == reduced_dims_output[1].first,
                          "DMA's don't have equal plane stride {0} != {1}", reduced_dims_input[1].first,
                          reduced_dims_output[1].first);
        descriptor.src_plane_stride = reduced_dims_input[1].second;
        descriptor.dst_plane_stride = reduced_dims_output[1].second;

        uint32_t nPlanes = descriptor.length / reduced_dims_input[1].first;
        VPUX_THROW_UNLESS(nPlanes < 256, "nPlanes is only on 8 bits; nPlanes here: {0}", nPlanes);
        descriptor.num_planes = nPlanes;
    } else if (reduced_dims_input.size() == 2) {
        descriptor.src_plane_stride = reduced_dims_input[1].second;
        descriptor.dst_plane_stride = descriptor.attr2d.dst_stride;

        uint32_t nPlanes = descriptor.length / reduced_dims_input[1].first;
        VPUX_THROW_UNLESS(nPlanes < 256, "nPlanes is only on 8 bits; nPlanes here: {0}", nPlanes);
        descriptor.num_planes = nPlanes;
    } else if (reduced_dims_output.size() == 2) {
        descriptor.src_plane_stride = descriptor.attr2d.src_stride;
        descriptor.dst_plane_stride = reduced_dims_output[1].second;
        uint32_t nPlanes = descriptor.length / reduced_dims_output[1].first;
        VPUX_THROW_UNLESS(nPlanes < 256, "nPlanes is only on 8 bits; nPlanes here: {0}", nPlanes);

        descriptor.num_planes = nPlanes;
    } else {
        descriptor.src_plane_stride = descriptor.attr2d.src_stride;
        descriptor.dst_plane_stride = descriptor.attr2d.dst_stride;
        descriptor.num_planes = 0;
    }

    uint8_t* ptrCharTmp = reinterpret_cast<uint8_t*>(&dmaTask);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUIPRegMapped::NNDMAOp::getBinarySize() {
    return sizeof(host_parsing::DmaWrapper);
}

mlir::FailureOr<uint64_t> vpux::VPUIPRegMapped::NNDMAOp::getOffsetOfWithinOperation(mlir::Value val) {
    if (val == input()) {
        return offsetof(host_parsing::DmaDescriptor, src);
    } else if (val == output_buff()) {
        return offsetof(host_parsing::DmaDescriptor, dst);
    }

    return mlir::failure();
}
