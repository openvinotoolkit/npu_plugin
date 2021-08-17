#ifndef DMA_H
#define DMA_H
#include <stdint.h>

#define CMDA_BASE_ADDR 0x2EF60000

typedef struct {
    uint64_t link_address : 40;
    uint64_t rsvd : 23;
    uint64_t watermark : 1;
    uint64_t type : 2;
    uint64_t burst_length : 8;
    uint64_t critical : 1;
    uint64_t interrupt_en : 1;
    uint64_t interrupt_trigger : 7;
    uint64_t skip_nr : 7;
    uint64_t order_forced : 1;
    uint64_t watermark_en : 1;
    uint64_t huf_en : 1;
    uint64_t barrier_en : 1;
    uint64_t wrap_en : 1;
    uint64_t wrap_min : 6;
    uint64_t wrap_max : 6;
    uint64_t rsvd_1 : 21;
    uint64_t src;
    uint64_t dst;
    uint64_t len : 24;
    uint64_t rsvd_2 : 8;
    uint64_t num_planes : 8;
    uint64_t task_id : 24;
    uint64_t src_plane_stride : 32;
    uint64_t dst_plane_stride : 32;
    uint64_t barrier_producer_mask;
    uint64_t barrier_consumer_mask;
}dma_descriptor_common_t;

void create_simple_transfer(
        uint64_t src,
        uint64_t dst,
        uint32_t len,
        dma_descriptor_common_t *desc_ptr);

void create_transfer_with_barrier(
        uint64_t src,
        uint64_t dst,
        uint32_t len,
        uint32_t consumer_barrier_no,
        uint32_t producer_barrier_no,
        dma_descriptor_common_t *desc_ptr);

int start_dma_transactions(dma_descriptor_common_t *desc_ptr);
#endif
