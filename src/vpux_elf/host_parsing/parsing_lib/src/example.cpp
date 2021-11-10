#include <convert.h>
#include <host_parsed_inference.h>

int main(int, char**)
{
    parsing_lib::DMATask t;
    t.src.data_dtype = parsing_lib::DType::U8;
    t.dst.data_dtype = parsing_lib::DType::U8;
    t.src.dimensions = std::vector<uint32_t>({1, 320,7,7});
    t.dst.dimensions = std::vector<uint32_t>({1, 320,7,7});
    t.src.strides = std::vector<float>({1.f, 15680.f, 1.f, 2240.f, 320.f});
    t.dst.strides = std::vector<float>({1.f, 15680.f, 1.f, 2240.f, 320.f});
    t.src.order = 0x1342;
    t.dst.order = 0x1342;
    t.compression = false;

    DmaDescriptor d;
    parsing_lib::convertDmaTask(t, d);
    parsing_lib::patchDmaTask(d, 0xCAFEBABE, 0xDEADBEEF, 0x0);

    uint32_t *desc = (uint32_t*) &d;
    for (int i = 0; i < 20; i++)
    {
        printf("%08x ", *desc++);
        if ((i+1) % 4 == 0)
            printf("\n");
    }

    // auto user_desc = &d;

    // printf(
    //     "|-------------------- DESCRIPTOR 0x%02X (0x%016X) INFO --------------------|\n"
    //     "|       SOURCE       |    DESTINATION     |        SIZE        |     NUM PLANES     |\n"
    //     "| 0x%016llX | 0x%016llX |     0x%08lX     |        0x%02lX        |\n"
    //     "|-----------------------------------------------------------------------------------|\n"
    //     "|      TASK ID       |  SRC PLANE STRIDE  |  DST PLANE STRIDE  |      WATERMARK     |\n"
    //     "|     0x%08lX     |     0x%08lX     |     0x%08lX     |        0x%02lX        |\n"
    //     "|-----------------------------------------------------------------------------------|\n"
    //     "|      TYPE      |       BL       |       CRT      |       IEN      |       IT      |\n"
    //     "|        %u       |      0x%02X      |        %u       |        %u       |      0x%02X     |\n"
    //     "|-----------------------------------------------------------------------------------|\n"
    //     "|      SKIP      |       ORD      |     WTM_EN     |      DEC_EN    |    BARR_EN    |\n"
    //     "|      0x%02X      |        %u       |        %u       |        %u       |        %u      |\n"
    //     "|-----------------------------------------------------------------------------------|\n"
    //     "|            BARRIER PROD MASK            |            BARRIER CONS MASK            |",
    //     0, user_desc, user_desc->src, user_desc->dst, user_desc->length, user_desc->num_planes, user_desc->task_id,
    //     user_desc->src_plane_stride, user_desc->dst_plane_stride, user_desc->watermark,
    //     (uint8_t)user_desc->cfg_link.cfg_bits.type, (uint8_t)user_desc->cfg_link.cfg_bits.burst_length,
    //     (uint8_t)user_desc->cfg_link.cfg_bits.critical, (uint8_t)user_desc->cfg_link.cfg_bits.interrupt_en,
    //     (uint8_t)user_desc->cfg_link.cfg_bits.interrupt_trigger, (uint8_t)user_desc->cfg_link.cfg_bits.skip_nr,
    //     (uint8_t)user_desc->cfg_link.cfg_bits.order_forced, (uint8_t)user_desc->cfg_link.cfg_bits.watermark_en,
    //     (uint8_t)user_desc->cfg_link.cfg_bits.dec_en, (uint8_t)user_desc->cfg_link.cfg_bits.barrier_en);

    // if (user_desc->cfg_link.cfg_bits.type == 0) {
    //     printf(
    //             "\n|            0x%016llX           |            0x%016llX           |\n"
    //             "|-----------------------------------------------------------------------------------|\n",
    //             user_desc->barriers1d.prod_mask, user_desc->barriers1d.cons_mask);
    // } else {
    //     printf(
    //             "\n|            0x%016llX           |            0x%016llX           |\n"
    //             "|-----------------------------------------------------------------------------------|\n"
    //             "|      SRC WIDTH     |     SRC STRIDE     |      DST WIDTH     |     DST STRIDE     |\n"
    //             "|     0x%08lX     |     0x%08lX     |     0x%08lX     |     0x%08lX     |\n"
    //             "|-----------------------------------------------------------------------------------|\n",
    //             user_desc->barriers.prod_mask, user_desc->barriers.cons_mask, user_desc->attr2d.src_width,
    //             user_desc->attr2d.src_stride, user_desc->attr2d.dst_width, user_desc->attr2d.dst_stride);
    // }
}
