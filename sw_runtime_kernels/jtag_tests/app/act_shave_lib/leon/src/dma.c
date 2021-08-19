#include "dma.h"
#include "common_functions.h"

void create_simple_transfer(
        uint64_t src,
        uint64_t dst,
        uint32_t len,
        dma_descriptor_common_t *desc_ptr){
    desc_ptr->src = src;
    desc_ptr->dst = dst;
    desc_ptr->len = len;
    desc_ptr->src_plane_stride = 0;
    desc_ptr->dst_plane_stride = 0;

    desc_ptr->link_address = 0;
    desc_ptr->rsvd = 0;
    desc_ptr->watermark = 0;
    desc_ptr->type = 0;
    desc_ptr->burst_length = 0;
    desc_ptr->critical = 0;
    desc_ptr->interrupt_en = 0;
    desc_ptr->interrupt_trigger = 0;
    desc_ptr->skip_nr = 0;
    desc_ptr->order_forced = 0;
    desc_ptr->watermark_en = 0;
    desc_ptr->huf_en = 0;
    desc_ptr->barrier_en = 0;
    desc_ptr->wrap_en = 0;
    desc_ptr->wrap_min = 0;
    desc_ptr->wrap_max = 0;
    desc_ptr->rsvd_1 = 0;
    desc_ptr->rsvd_2 = 0;
    desc_ptr->num_planes = 0;
    desc_ptr->task_id = 0;
    desc_ptr->barrier_producer_mask = 0ULL;
    desc_ptr->barrier_consumer_mask = 0ULL;
}

void create_transfer_with_barrier(
        uint64_t src,
        uint64_t dst,
        uint32_t len,
        uint32_t consumer_barrier_no,
        uint32_t producer_barrier_no,
        dma_descriptor_common_t *desc_ptr){
    create_simple_transfer(src,dst,len, desc_ptr);
    desc_ptr->barrier_en = 1;
    desc_ptr->barrier_producer_mask = 1ULL << producer_barrier_no;
    desc_ptr->barrier_consumer_mask = 1ULL << consumer_barrier_no;
}

int start_dma_transactions(dma_descriptor_common_t *desc_ptr){
    if(!desc_ptr){
        return 0;
    }

#ifdef _WIN32
  uint64_t src_addr = desc_ptr->src;
  uint64_t dst_addr = desc_ptr->dst;

  retarget_address(&src_addr);
  retarget_address(&dst_addr);

  uint8_t* src_ptr = (uint8_t*)src_addr;
  uint8_t* dst_ptr = (uint8_t*)dst_addr;

  for(int i = 0; i < desc_ptr->len; ++i){
    *dst_ptr++ = *src_ptr++;
  }

  if(desc_ptr->link_address){
    uint64_t next_desc_addr = desc_ptr->link_address;
    retarget_address(&next_desc_addr);
    do_transfer((dma_descriptor_common_t*)next_desc_addr);
  }
  return (int)desc_ptr->len;
#else
    // TODO trigger Simics DMA model
    volatile uint64_t * dma_cmda_crtl_reg = (uint64_t *)(CMDA_BASE_ADDR);
    *dma_cmda_crtl_reg = 0x1;
    /*volatile uint64_t * dma_cmda_clr_link2fifo_reg = (CMDA_BASE_ADDR + 0x250);
    *dma_cmda_clr_link2fifo_reg = ~0ULL;*/
    volatile uint64_t * dma_cmda_set_link2fifo_reg = (uint64_t *)(CMDA_BASE_ADDR + 0x240);
    volatile uint64_t * dma_cmda_clr_link2fifo_reg = (uint64_t *)(CMDA_BASE_ADDR + 0x250);
    volatile uint64_t * dma_cmda_fentry_reg = (uint64_t *)(CMDA_BASE_ADDR + 0x130);
    *dma_cmda_clr_link2fifo_reg = 0x1ff;
    *dma_cmda_set_link2fifo_reg = 0x1fe;

    /* set up P & C barrier[0] to non-zero - BARRIER_COUNT[0]@0x2ef34000 <- 0x00010001 */
    /* set up PBarr IRQ EN BARRIER_PIRQ_EN[63:0]@0x4a0+0x2ef34000 <- 0xFFFFFFFFFFFFFFFF */
    /* set up CBarr IRQ EN BARRIER_CIRQ_EN[63:0]@0x4c0+0x2ef34000 <- 0xFFFFFFFFFFFFFFFF */
    /* set PBarr IRQ SET EN @0x580+0x2ef34000 <- 0xFFFFFFFFFFFFFFFF */
    /* set CBarr IRQ SET EN @0x5C0+0x2ef34000 <- 0xFFFFFFFFFFFFFFFF */
    /* set PBarr Cdec @0x420+0x2ef34000 <- 0x0000000000000001 */
    /* could do the same to trigger a producer barrier interrupt @0x400+base */

    *dma_cmda_fentry_reg = (uint64_t)(uint32_t)desc_ptr;
#endif

    return -1;
}
