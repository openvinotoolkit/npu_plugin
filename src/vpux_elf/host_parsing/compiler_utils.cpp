#include <nce_2p7_hw.h>
#include <relocation.h>

// There are 3 relocations for a DMA descriptor:
// desc.src - final input address
// desc.dst - final destination address
// desc.link_address - address of the next descriptor in the linked list

// Barriers
// desc.barriers.cons_mask / desc.barriers.prod_mask
// For a single tile inference, physical barriers can be 0-31 (tile 0) or 32-63 (tile 1).
// Compiler should generate a version of mapped inference with each version of barriers, runtime will choose which MappedInference to use based on the available tile
// If this is a dual tile inference then no barriers need to be moved becase all 0-63 are available
//
// To handle the "port" argument, compiler needs to maintain separate descriptor lists per port (DMA engine)
//
// Barriers are a bit tricky. The DmaDescriptor itself needs a mask of physical barrier IDs (1 per bit) to use.
// The DmaWrapper needs the increasing "virtual" ID to keep track of the execution state.

uint64_t barriersToMask(std::vector<uint8_t> &physicalIDs) {
    uint64_t mask = 0;
    for (uint8_t id : physicalIDs)
        mask |= (1 << id);

    return mask;
}

void patchDmaDescriptor(DmaDescriptor &desc, uint64_t src, uint64_t dst, uint64_t link = 0)
{
    desc.src = src;
    desc.dst = dst;
    desc.link_address = link;
}

DmaDescriptor createDmaDescriptor(uint32_t size, bool compressed, bool uint64_t waitBarriersMask, uint64_t updateBarriersMask)
{
    return createDmaDescriptorExt(size, 0, 0, 0, 0, 1, 0, 0, compressed, false, waitBarriersMask, updateBarriersMask);
}

DmaDescriptor createDmaDescriptorExt(uint32_t size,
uint32_t srcWidth = 0, int32_t srcStride = 0, uint32_t dstWidth = 0, int32_t dstStride = 0,
uint8_t numPlanes = 1, int32_t srcPlaneStride = 0, int32_t dstPlaneStride = 0, bool compressed = false, bool orderForced = false,
uint64_t waitBarriersMask = 0, uint64_t updateBarriersMask = 0)
{
    DmaDescriptor desc = {0};
    desc.link_address = linked_desc;
    desc.src = 0;
    desc.dst = 0;
    desc.length = size;
    desc.num_planes = numPlanes;
    desc.src_plane_stride = srcPlaneStride;
    desc.dst_plane_stride = dstPlaneStride;
    desc.attr2d.src_width = srcWidth;
    desc.attr2d.src_stride = srcStride;
    desc.attr2d.dst_width = dstWidth;
    desc.attr2d.dst_stride = dstStride;
    desc.cfg_bits.barrier_en = 1;
    desc.cfg_bits.critical = 1;
    desc.cfg_bits.order_forced = orderForced;
    desc.cfg_bits.burst_length = 16,
    desc.cfg_bits.dec_en = compressed;
    desc.barriers.cons_mask = waitBarriersMask;
    desc.barriers.prod_mask = updateBarriersMask;
}
