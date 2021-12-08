/*
 * {% copyright %}
 */
#ifndef NN_DMA_FEEDER_H_
#define NN_DMA_FEEDER_H_

#include <nn_ring_buffer.h>
#include <nn_memory.h>
#include <Dma.h>

namespace nn {
namespace inference_runtime {
class FasterDmaFeeder {
public:
    typedef DmaJobHandle Handle;
    typedef DmaDescriptorConfig Transaction;

    typedef void (*Callback)(void *user_context, Handle *handle, unsigned int transaction_count);

    struct HandleWrapper {
        Handle handle_;
        unsigned int transaction_count_;
        uint32_t aperture_offset_;

        HandleWrapper();
        void reset();
        void setApertureOffset(uint32_t aperture_offset);

        void append(const HandleWrapper &wrapper);
        void append(Transaction *transaction, DmaDescriptor *storage);
        void init(DmaDescriptor *head, DmaDescriptor *tail, uint32_t numTransactions);
    };

    FasterDmaFeeder();
    FasterDmaFeeder(unsigned char engine, unsigned char agent);

    void set_context_SSID(uint32_t vpu_ssid);
    void set_context_aperture_offset(uint32_t aperture_offset);
    void set_callback(Callback callback, void *user_context);
    void enqueue(const HandleWrapper &wrapper);
    void debug_info() const;

private:
    // This in fact decays to nothing on single-core systems.
    // Keeping it here just as a reference for the name.
    RTEMS_INTERRUPT_LOCK_MEMBER(dma_feeder_lock_)

    DmaJobConfig config_;
    HandleWrapper wrappers_[2];
    unsigned int available_;
    Callback callback_;
    void *user_context_;
    unsigned char engine_;
    bool active_;

    Handle *dma_callback(Handle *);
    static void dma_callback(DmaJobHandle *current_handle, void *user_context, DmaJobHandle **next_handle);
};
} // namespace inference_runtime
} // namespace nn

#endif // NN_DMA_FEEDER_H_
