#include <nce_2p7_hw.h>
#include <data_types.h>

namespace parsing_lib {
void convertDmaTask(const DMATask &t, DmaDescriptor &desc);
void patchDmaTask(DmaDescriptor &desc, uint64_t src, uint64_t dst, uint64_t link_next);
}
