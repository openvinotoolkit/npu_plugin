/*
* {% copyright %}
*/
#include <Dma.h>

namespace nn {
    namespace inference_context {
        // Register RTEMS ISR for context violations
        void hook_isr_for_nce_context_violation();

        // RTEMS handler for context violations
        void contexViolationIsr(void *source);

        // Initialize the buffer dma reads from for wiping CMX
        void init_cmx_wipe_buffer();

        // Wipe CMX on Context change
        DmaStatus wipe_cmx(char tile, uint32_t dest, uint32_t size);
    } // namespace inference_context
} // namespace nn
