//
// Copyright (C) 2022 Intel Corporation.
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

#include <host_parsed_inference.h>
#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

//
// NNDMAOp
//

// For further development, please refer to the ticket #36225.
// Note: Copy-pasted from VPUIP.cpp.

namespace {

llvm::SmallVector<std::pair<uint32_t, int32_t>> reduce_dims_for_dma(mlir::Value val) {
    mlir::MemRefType memref = val.getType().cast<mlir::MemRefType>();

    auto const logicalShape = vpux::getShape(val);
    auto const logicalStrides = vpux::getStrides(val);
    auto const order = vpux::DimsOrder::fromValue(val);
    auto const memShape = order.toMemoryOrder(logicalShape);
    auto const memStrides = order.toMemoryOrder(logicalStrides);

    auto inner_most_index = memShape.size() - 1;
    llvm::SmallVector<std::pair<uint32_t, int32_t>> finalDims;

    uint32_t previous_size = (uint32_t)memShape[MemDim(inner_most_index)];
    uint32_t previous_stride_bits = (uint32_t)(vpux::Bit(memStrides[MemDim(inner_most_index)]).count());

    if (previous_size * memref.getElementTypeBitWidth() < previous_stride_bits) {
        int32_t final_stride = previous_stride_bits / CHAR_BIT;
        uint32_t final_size = previous_size * memref.getElementTypeBitWidth() / CHAR_BIT;

        finalDims.push_back({final_size, final_stride});
    }

    // TODO: Could there be some way to iterate over all MemDim's of a particular shape/order?
    //       Please refer to the ticket #36225.
    for (size_t dim = inner_most_index - 1; dim > 0; --dim) {
        auto memDim = MemDim(dim);

        uint32_t current_size = (uint32_t)memShape[memDim];
        uint32_t current_stride_bits = (uint32_t)(vpux::Bit(memStrides[memDim]).count());

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

    dmaTask.start_after = (uint16_t)start_after();

    host_parsing::DmaDescriptor& descriptor = dmaTask.transaction;
    descriptor.cfg_link.cfg_bits.type = 1;
    descriptor.cfg_link.cfg_bits.burst_length = 16;
    descriptor.cfg_link.cfg_bits.barrier_en = 1;

    uint64_t cons_mask = 0;
    for (auto waitBarrier : waitBarriers()) {
        auto op = llvm::dyn_cast<VPUIPRegMapped::ConfigureBarrierOp>(waitBarrier.getDefiningOp());
        cons_mask |= ((uint64_t)1) << op.id();
    }
    uint64_t prod_mask = 0;
    for (auto updateBarrier : updateBarriers()) {
        auto op = llvm::dyn_cast<VPUIPRegMapped::ConfigureBarrierOp>(updateBarrier.getDefiningOp());
        prod_mask |= ((uint64_t)1) << op.id();
    }
    descriptor.barriers.cons_mask = cons_mask;
    descriptor.barriers.prod_mask = prod_mask;

    descriptor.length = (uint32_t)(inputType.getNumElements() * vpux::Byte(vpux::getElemTypeSize(inputType)).count());

    // TODO: can we have this reduction at a pass at memref level? Need to place
    // some conditions on the DMA, and in some scenarios, may have to do 1*DMA -> n*DMA
    //      transaction rewrites.
    // Please refer to the ticket #36225.
    auto reduced_dims_input = reduce_dims_for_dma(input());
    auto reduced_dims_output = reduce_dims_for_dma(output());

    if (reduced_dims_input.size() > 2 || reduced_dims_output.size() > 2) {
        VPUX_THROW("cannot reduce dims to 2 for DMA");
    }

    descriptor.attr2d.src_width = reduced_dims_input[0].first;
    descriptor.attr2d.src_stride = reduced_dims_input[0].second;
    descriptor.attr2d.dst_width = reduced_dims_output[0].first;
    descriptor.attr2d.dst_stride = reduced_dims_output[0].second;

    if (reduced_dims_input.size() == 2 && reduced_dims_output.size() == 2) {
        VPUX_THROW_UNLESS(reduced_dims_input[1].first == reduced_dims_output[1].first,
                          "DMA's don't have equal plane stride");
        descriptor.src_plane_stride = reduced_dims_input[1].second;
        descriptor.dst_plane_stride = reduced_dims_output[1].second;

        uint32_t nPlanes = descriptor.length / reduced_dims_input[1].first;
        VPUX_THROW_UNLESS(nPlanes < 256, "nPlanes is only on 8 bits");
        descriptor.num_planes = nPlanes;
    } else if (reduced_dims_input.size() == 2) {
        descriptor.src_plane_stride = reduced_dims_input[1].second;
        descriptor.dst_plane_stride = descriptor.attr2d.dst_stride;

        uint32_t nPlanes = descriptor.length / reduced_dims_input[1].first;
        VPUX_THROW_UNLESS(nPlanes < 256, "nPlanes is only on 8 bits");
        descriptor.num_planes = nPlanes;
    } else if (reduced_dims_output.size() == 2) {
        descriptor.src_plane_stride = descriptor.attr2d.src_stride;
        descriptor.dst_plane_stride = reduced_dims_output[1].second;
        uint32_t nPlanes = descriptor.length / reduced_dims_output[1].first;
        VPUX_THROW_UNLESS(nPlanes < 256, "nPlanes is only on 8 bits");

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
