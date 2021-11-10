#include <data_types.h>
#include <utils.h>
#include <nce_2p7_hw.h>
#include <cassert>
#include <algorithm>
#include <memory.h>

namespace parsing_lib {
void convertDmaTask(const DMATask &t, host_parsing::DmaDescriptor &desc) {
    SimplifiedTensorLayout src_layout, dst_layout;
    decode_simplified_layout(t.src, src_layout);
    decode_simplified_layout(t.dst, dst_layout);

    uint32_t src_width = src_layout.line_length();
    uint32_t dst_width = dst_layout.line_length();
    uint32_t src_stride = src_layout.line_stride();
    uint32_t dst_stride = dst_layout.line_stride();
    uint32_t num_planes = src_layout.plane_count();
    uint32_t src_plane_stride = src_layout.plane_stride();
    uint32_t dst_plane_stride = dst_layout.plane_stride();
    uint32_t size = src_layout.total_length();

    if (t.compression) {
        if (src_width || src_stride || dst_width || dst_stride || src_plane_stride || dst_plane_stride) {
            assert(false && "ERROR: Decompression is only supported on 1D transfers");
        }
    } else {
        if (!!src_plane_stride ^ !!dst_plane_stride) {
            if (src_plane_stride)
                num_planes = std::max(1u, src_layout.plane_count()), dst_plane_stride = size / num_planes;
            else
                num_planes = std::max(1u, dst_layout.plane_count()), src_plane_stride = size / num_planes;
        }

        assert(num_planes > 0);

        if (src_width == src_stride)
            src_width = src_stride = size / num_planes;

        if (dst_width == dst_stride)
            dst_width = dst_stride = size / num_planes;
    }

    // HW stores planes as actual_planes - 1
    if (num_planes > 0)
        num_planes--;

    memset(&desc, 0x0, sizeof(desc));

    if (!src_width && !dst_width && !src_stride && !dst_stride)
    {
        num_planes = src_plane_stride = dst_plane_stride = 0;
        desc.cfg_link.cfg_bits.type = 0; // 1D
    }
    else if (!num_planes)
    {
        src_plane_stride = dst_plane_stride = 0;
        desc.cfg_link.cfg_bits.type = 1; // 2D
    }
    else
    {
        desc.cfg_link.cfg_bits.type = 1; // 3D
    }

    desc.length = src_layout.total_length();
    desc.num_planes = num_planes;
    desc.src_plane_stride = src_plane_stride;
    desc.dst_plane_stride = dst_plane_stride;
    desc.attr2d.src_width = src_width;
    desc.attr2d.src_stride = src_stride;
    desc.attr2d.dst_width = dst_width;
    desc.attr2d.dst_stride = dst_stride;
    desc.cfg_link.cfg_bits.barrier_en = 1;
    desc.cfg_link.cfg_bits.critical = 1;
    desc.cfg_link.cfg_bits.order_forced = t.set_ord;
    desc.cfg_link.cfg_bits.burst_length = 15;
    desc.cfg_link.cfg_bits.dec_en = t.compression;
    desc.cfg_link.cfg_bits.skip_nr = (1 << 6) - 1;
    desc.barriers.cons_mask = 0;
    desc.barriers.prod_mask = 0;
}

void patchDmaTask(host_parsing::DmaDescriptor &desc, uint64_t src, uint64_t dst, uint64_t link_next) {
    desc.src = src;
    desc.dst = dst;
    desc.link_address = link_next;
}
} // namespace nce_lib
