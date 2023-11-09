//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/ELF/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include "vpux/compiler/dialect/VPUMI37XX/utils.hpp"

#include <vpu_nnrt_api_37xx.h>

using namespace vpux;

//
// NNDMAOp
//

// For further development, please refer to the ticket E#36225.

void VPUMI37XX::NNDMAOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, mlir::Type index,
                               mlir::Value input, mlir::ValueRange output_buffs, mlir::Value previousDMAIdx,
                               mlir::ValueRange waitBarriers, mlir::ValueRange updateBarriers, bool compression,
                               uint64_t start_after, uint64_t clean_after, bool is_out_of_order, bool is_critical,
                               int64_t port, VPUIP::DMADescriptorAttr dma_descriptor) {
    build(odsBuilder, odsState, index, nullptr, input, output_buffs, previousDMAIdx, waitBarriers, updateBarriers,
          compression, start_after, clean_after, is_out_of_order, is_critical, port, dma_descriptor);
}

namespace {

void decode_storage_order(ShapeRef dims, StridesRef strides, unsigned char* order) {
    const unsigned int S = dims.size();

    for (unsigned int i = 0; i < S; ++i)
        order[i] = i;

    std::sort(&order[0], &order[0] + S, [&](int lhs, int rhs) {
        return std::make_tuple(strides[Dim(lhs)], dims[Dim(lhs)], lhs) <
               std::make_tuple(strides[Dim(rhs)], dims[Dim(rhs)], rhs);
    });
}

class SimplifiedTensorLayout {
public:
    explicit SimplifiedTensorLayout(mlir::Value value) {
        VPUX_THROW_UNLESS(value, "Encountered nullptr value");

        auto ndType = value.getType().cast<vpux::NDTypeInterface>();
        const auto sizes = ndType.getShape();
        const auto strides = ndType.getStrides();
        auto dims = static_cast<unsigned int>(sizes.size());

        std::vector<unsigned char> order(dims, 0);
        decode_storage_order(sizes, strides, order.data());

        unsigned int line_stride_in_bits = 0;
        unsigned int plane_stride_in_bits = 0;
        unsigned int* rt_dims[SimplifiedTensorLayout::STRIDING_LEVELS] = {&line_length_, &plane_length_};
        unsigned int* rt_strides[SimplifiedTensorLayout::STRIDING_LEVELS] = {&line_stride_in_bits,
                                                                             &plane_stride_in_bits};

        auto bit_strides = [&](Dim i) -> unsigned int {
            return static_cast<unsigned int>(strides[i].count());
        };

        unsigned int previous_size = 1;
        unsigned int previous_stride = static_cast<unsigned int>(vpux::getElemTypeSize(ndType).count());
        unsigned int total_length_in_bits = previous_stride;

        for (unsigned int dim = 0, level = 0; dim < dims; ++dim) {
            const unsigned int crt_size = sizes[Dim(order[dim])];
            unsigned int crt_stride = bit_strides(Dim(order[dim]));
            total_length_in_bits *= crt_size;

            if (previous_size * previous_stride < crt_stride) {
                if (sizes[Dim(order[dim])] == 1) {
                    if (dim + 1 == dims)
                        continue;

                    crt_stride = bit_strides(Dim(order[dim + 1]));
                }

                VPUX_THROW_UNLESS(level < SimplifiedTensorLayout::STRIDING_LEVELS, "Max striding levels exceeded");

                *rt_strides[level] = crt_stride;
                *rt_dims[level] = (previous_size * previous_stride) / (level ? *rt_strides[level - 1] : CHAR_BIT);
                ++level;
            }

            previous_size = crt_size;
            previous_stride = crt_stride;
        }

        line_stride_ = line_stride_in_bits / CHAR_BIT;
        plane_stride_ = plane_stride_in_bits / CHAR_BIT;
        total_length_ = total_length_in_bits / CHAR_BIT;
    }

    unsigned int line_stride() const {
        return line_stride_;
    }
    unsigned int line_length() const {
        return line_length_;
    }
    unsigned int plane_stride() const {
        return plane_stride_;
    }
    unsigned int plane_length() const {
        return plane_length_;
    }
    unsigned int plane_count() const {
        return plane_length_ ? (total_length_ / plane_length_ / (line_length_ ? line_length_ : 1)) : 1;
    }
    unsigned int total_length() const {
        return total_length_;
    }

private:
    static constexpr auto STRIDING_LEVELS = 2;
    unsigned int line_stride_ = 0;
    unsigned int line_length_ = 0;
    unsigned int plane_stride_ = 0;
    unsigned int plane_length_ = 0;
    unsigned int total_length_ = 0;
};

}  // namespace

void vpux::VPUMI37XX::NNDMAOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    nn_public::VpuDMATask dmaTask;

    // safe init to zero the structure
    memset(reinterpret_cast<void*>(&dmaTask), 0, sizeof(dmaTask));

    const auto hasDescriptor = getDmaDescriptor().has_value();

    auto inputType = getInput().getType().cast<mlir::MemRefType>();

    dmaTask.barriers_sched_.start_after_ = checked_cast<uint16_t>(getStartAfter());
    dmaTask.barriers_sched_.clean_after_ = checked_cast<uint16_t>(getCleanAfter());

    auto& descriptor = dmaTask.transaction_;
    descriptor.cfg_link.cfg_bits.burst_length = 255;  // set burst lenght to max value
    descriptor.cfg_link.cfg_bits.barrier_en = 1;

    // In case of multicasting (multiple outputs) we will mask the destination with the multicast mask;

    if (getOutputBuffs().size() > 1)
        descriptor.dst = 0xC00000;

    // If this DMA Op is not used by any other Op this is the last Op in DMA chain and link_address shall be zero

    vpux::VPUMI37XX::NNDMAOp dmaUser = nullptr;

    for (auto user : getResult().getUsers()) {
        auto newDmaUser = mlir::dyn_cast<vpux::VPUMI37XX::NNDMAOp>(user);

        if (newDmaUser) {
            VPUX_THROW_UNLESS(!dmaUser, "VPUMI37XX::NNDMAOp '{0}' at loc '{1}' has more than one DMA user",
                              getOperation()->getName(), getLoc());

            dmaUser = newDmaUser;
        }
    }

    if (dmaUser) {
        auto dmaOpIndex = dmaUser.getResult().getType().cast<VPURegMapped::IndexType>();
        descriptor.link_address = static_cast<uint64_t>(dmaOpIndex.getValue());
    } else {
        descriptor.link_address = 0;
    }

    descriptor.cfg_link.cfg_bits.critical = 1;
    descriptor.cfg_link.cfg_bits.order_forced = !getIsOutOfOrder();
    descriptor.cfg_link.cfg_bits.skip_nr = 63;

    auto src_layout = SimplifiedTensorLayout(getInput());
    auto dst_layout = SimplifiedTensorLayout(getOutputBuffs()[0]);

    uint32_t src_width = src_layout.line_length();
    uint32_t dst_width = dst_layout.line_length();
    uint32_t src_stride = src_layout.line_stride();
    uint32_t dst_stride = dst_layout.line_stride();
    uint32_t num_planes = src_layout.plane_count();
    uint32_t src_plane_stride = src_layout.plane_stride();
    uint32_t dst_plane_stride = dst_layout.plane_stride();
    uint32_t size = src_layout.total_length();

    if (!hasDescriptor && !getCompression()) {
        if (!!src_plane_stride ^ !!dst_plane_stride) {
            if (src_plane_stride)
                num_planes = std::max(1u, src_layout.plane_count()), dst_plane_stride = size / num_planes;
            else
                num_planes = std::max(1u, dst_layout.plane_count()), src_plane_stride = size / num_planes;
        }

        VPUX_THROW_UNLESS(num_planes > 0, "Encountered num planes = {0}", num_planes);

        if (src_width == src_stride)
            src_width = src_stride = size / num_planes;

        if (dst_width == dst_stride)
            dst_width = dst_stride = size / num_planes;
    }

    if (hasDescriptor) {
        const auto dmaDescriptor = getDmaDescriptor().value();
        descriptor.length = checked_cast<uint32_t>(dmaDescriptor.getLen().getInt());
        descriptor.attr2d.src_width = checked_cast<uint32_t>(dmaDescriptor.getSrcWidth().getInt());
        descriptor.attr2d.dst_width = checked_cast<uint32_t>(dmaDescriptor.getDstWidth().getInt());
        descriptor.attr2d.src_stride = checked_cast<int32_t>(dmaDescriptor.getSrcStride().getInt());
        descriptor.attr2d.dst_stride = checked_cast<int32_t>(dmaDescriptor.getDstStride().getInt());
        descriptor.src_plane_stride = checked_cast<int32_t>(dmaDescriptor.getSrcPlaneStride().getInt());
        descriptor.dst_plane_stride = checked_cast<int32_t>(dmaDescriptor.getDstPlaneStride().getInt());
        descriptor.num_planes = checked_cast<uint32_t>(dmaDescriptor.getNumPlanes().getInt());
    } else {
        const auto elemSize = vpux::getElemTypeSize(inputType);
        auto totalSizeBits = alignMemSize(inputType.getNumElements() * elemSize, Byte(1));

        descriptor.length = vpux::Byte(totalSizeBits).count();
        descriptor.attr2d.src_width = src_width;
        descriptor.attr2d.dst_width = dst_width;
        descriptor.attr2d.src_stride = checked_cast<int32_t>(src_stride);
        descriptor.attr2d.dst_stride = checked_cast<int32_t>(dst_stride);
        descriptor.src_plane_stride = checked_cast<int32_t>(src_plane_stride);
        descriptor.dst_plane_stride = checked_cast<int32_t>(dst_plane_stride);
        descriptor.num_planes = num_planes;
    }

    --descriptor.num_planes;
    if (!descriptor.attr2d.src_width && !descriptor.attr2d.dst_width && !descriptor.attr2d.src_stride &&
        !descriptor.attr2d.dst_stride) {
        descriptor.num_planes = descriptor.src_plane_stride = descriptor.dst_plane_stride = 0;
        descriptor.cfg_link.cfg_bits.type = 0;
    } else if (!descriptor.num_planes) {
        descriptor.src_plane_stride = descriptor.dst_plane_stride = 0;
        descriptor.cfg_link.cfg_bits.type = 1;
    } else {
        descriptor.cfg_link.cfg_bits.type = 1;
    }

    if (getCompression()) {
        descriptor.cfg_link.cfg_bits.dec_en = 1;
        VPUX_THROW_UNLESS(descriptor.num_planes == 0,
                          "For DMA compression to be possible, the computed num_planes for the transaction needs to be "
                          "0, got {0}",
                          checked_cast<uint8_t>(descriptor.num_planes));

        // Ensure plane strides are set to 0 and set transaction type to 1D
        descriptor.src_plane_stride = descriptor.dst_plane_stride = 0;
        descriptor.cfg_link.cfg_bits.type = 0;
    }

    auto& barrierConsMask =
            descriptor.cfg_link.cfg_bits.type ? descriptor.barriers.cons_mask : descriptor.barriers1d.cons_mask;
    auto& barrierProdMask =
            descriptor.cfg_link.cfg_bits.type ? descriptor.barriers.prod_mask : descriptor.barriers1d.prod_mask;

    barrierConsMask = VPUMI37XX::computeMask(getWaitBarriers());
    barrierProdMask = VPUMI37XX::computeMask(getUpdateBarriers());

    uint8_t* ptrCharTmp = reinterpret_cast<uint8_t*>(&dmaTask);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUMI37XX::NNDMAOp::getBinarySize() {
    return sizeof(nn_public::VpuDMATask);
}

size_t vpux::VPUMI37XX::NNDMAOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuDMATask);
}

mlir::FailureOr<uint64_t> vpux::VPUMI37XX::NNDMAOp::getOffsetOfWithinOperation(mlir::Value val) {
    if (val == getInput()) {
        return offsetof(nn_public::VpuDMATask, transaction_) + offsetof(vpu_dma_descriptor_t, src);
    } else if (val == getOutputBuffs()[0]) {
        return offsetof(nn_public::VpuDMATask, transaction_) + offsetof(vpu_dma_descriptor_t, dst);
    } else if (val == getPreviousDMAIdx()) {
        return offsetof(nn_public::VpuDMATask, transaction_);
    }

    return mlir::failure();
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::NNDMAOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

vpux::ELF::SectionFlagsAttr vpux::VPUMI37XX::NNDMAOp::getAccessingProcs() {
    return (ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA);
}

vpux::ELF::SectionFlagsAttr vpux::VPUMI37XX::NNDMAOp::getUserProcs() {
    return (ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA);
}

vpux::VPURegMapped::TaskType vpux::VPUMI37XX::NNDMAOp::getTaskType() {
    return vpux::VPURegMapped::TaskType::DMA;
}
