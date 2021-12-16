/*
* {% copyright %}
*/
#include "nn_dma_feeder.h"
#include <nn_math.h>
#include <nn_log.h>
#include <assert.h>
#include <nn_cache.h>

namespace nn
{
    namespace inference_runtime
    {
        FasterDmaFeeder::HandleWrapper::HandleWrapper() :
            handle_(),
            transaction_count_(0),
            aperture_offset_(0)
        {
            auto result = DmaJobCreate(&handle_);
            handle_.features_used |= HGL_DMA_FLAG_BARRIER_EN | HGL_DMA_FLAG_WATERMARK_EN | HGL_DMA_FLAG_CRITICAL_TASK | HGL_DMA_FLAG_DECODE_EN | HGL_DMA_FLAG_ORDER_FORCED;
            assert(result == DMA_SUCCESS);
            UNUSED(result);
        }

        void FasterDmaFeeder::HandleWrapper::setApertureOffset(uint32_t aperture_offset)
        {
            aperture_offset_ = aperture_offset;
            DmaJobSetUserDataOffset(&handle_, aperture_offset);
        }

        void FasterDmaFeeder::HandleWrapper::reset()
        {
            DmaJobCreate(&handle_);
            DmaJobSetUserDataOffset(&handle_, aperture_offset_);
            transaction_count_ = 0;
        }

        void FasterDmaFeeder::HandleWrapper::append(const FasterDmaFeeder::HandleWrapper &wrapper)
        {
            if (transaction_count_ > 0)
            {
                // The DmaJobLink API expects a fresh handle to link the other arguments into
                // handle_ may still have transactions when we go to append again so this call will fail
                DmaJobHandle newHandle;
                DmaJobCreate(&newHandle);

                DmaJobLink(&newHandle, 2, &handle_, &wrapper.handle_);

                // Copy the temporary handle back into our storage for later execution
                memcpy_s(&handle_, sizeof(handle_), &newHandle, sizeof(newHandle));
            }
            else
                memcpy_s(&handle_, sizeof(handle_), &wrapper.handle_, sizeof(wrapper.handle_));

            transaction_count_ += wrapper.transaction_count_;
        }

        void FasterDmaFeeder::HandleWrapper::append(FasterDmaFeeder::Transaction *transaction, DmaDescriptor *storage)
        {
            DmaJobAddDescriptor(&handle_, storage, transaction);
            ++transaction_count_;
        }

        void FasterDmaFeeder::HandleWrapper::init(DmaDescriptor *head, DmaDescriptor *tail, uint32_t numTransactions)
        {
            tail->link_address = 0;
            DmaJobAddDescriptorList(&handle_, head);
            transaction_count_ = numTransactions;
        }

        FasterDmaFeeder::FasterDmaFeeder() :
            config_({
                .link_agent = 0,
                .callback = &dma_callback,
                .context = this,
                .wait = false,
                .context_id = DMA_DEFAULT_CID}),
            wrappers_(),
            available_(0),
            callback_(nullptr),
            user_context_(nullptr),
            engine_(0),
            active_(false)
        {
            nnLog(MVLOG_INFO, "Instance created using link agent %d", config_.link_agent);
        }

        FasterDmaFeeder::FasterDmaFeeder(unsigned char engine, unsigned char agent) :
            config_({
                .link_agent = agent,
                .callback = &dma_callback,
                .context = this,
                .wait = false,
                .context_id = DMA_DEFAULT_CID}),
            wrappers_(),
            available_(0),
            callback_(nullptr),
            user_context_(nullptr),
            engine_(engine),
            active_(false)
        {
            nnLog(MVLOG_INFO, "Instance created using link agent %d", config_.link_agent);
        }

        void FasterDmaFeeder::set_context_SSID(uint32_t ssid)
        {
            if (active_)
                nnLog(MVLOG_WARN, "Changing SSID on a running DMA job!");

            config_.context_id = ssid;
        }

        void FasterDmaFeeder::set_callback(Callback callback, void *user_context)
        {
            callback_ = callback;
            user_context_ = user_context;
        }

        void FasterDmaFeeder::enqueue(const HandleWrapper &wrapper)
        {
            if (wrapper.transaction_count_ == 0)
                return;

            rtems_interrupt_lock_context lock_ctx;
            rtems_interrupt_lock_acquire(&dma_feeder_lock_, &lock_ctx);

            auto &current = wrappers_[available_];
            current.append(wrapper);

            if (!active_)
            {
                active_ = true;
                available_ ^= 1;
                rtems_interrupt_lock_release(&dma_feeder_lock_, &lock_ctx);

                nnLog(MVLOG_DEBUG, "Start DMA handle @ %p to link agent %d", &current.handle_, config_.link_agent);

                auto result = DmaJobStart(&current.handle_, HGL_DMA_TYPE_NCE, static_cast<HglDmaEngineId>(engine_), DMA_MODE_REAL_TIME, &config_);
                assert(result == DMA_SUCCESS);
                UNUSED(result);
            }
            else
            {
                rtems_interrupt_lock_release(&dma_feeder_lock_, &lock_ctx);
            }
        }

        void FasterDmaFeeder::debug_info() const
        {
            rtems_interrupt_lock_context lock_ctx;
            rtems_interrupt_lock_acquire(&dma_feeder_lock_, &lock_ctx);

            nnLog(MVLOG_WARN, "Agent: %u, Active: %u, AV: %u, Pending: %u, Waiting: %u", config_.link_agent, active_, available_, wrappers_[available_ ^ 1].transaction_count_, wrappers_[available_].transaction_count_);

            if (wrappers_[available_ ^ 1].transaction_count_ > 0)
            {
                const auto *handle = &wrappers_[available_ ^ 1].handle_;
                nnLog(MVLOG_WARN, "Pending handle: %p, Head: %p, Tail: %p, TaskId: %u, Processed: %u", handle, handle->head, handle->tail, handle->task_id, handle->status);

                for (const DmaDescriptor *transaction = (const DmaDescriptor*) ((uint32_t)handle->head + handle->user_data_offset);
                     transaction != nullptr;
                     transaction = reinterpret_cast<const DmaDescriptor *>(static_cast<uint32_t>(transaction->link_address + handle->user_data_offset)))
                {
                    unsigned int cons_mask = static_cast<unsigned int>(transaction->cfg_link.cfg_bits.type ? transaction->barriers.cons_mask : transaction->barriers1d.cons_mask);
                    unsigned int prod_mask = static_cast<unsigned int>(transaction->cfg_link.cfg_bits.type ? transaction->barriers.prod_mask : transaction->barriers1d.prod_mask);

                    nnLog(MVLOG_WARN, "Transaction: %p, wmark: %llu, src: %p, dst: %p, length: %8u, Ds: %u, wait: %3d, post: %3d",
                        transaction, transaction->watermark, static_cast<uint32_t>(transaction->src), static_cast<uint32_t>(transaction->dst), transaction->length,
                        1 + transaction->cfg_link.cfg_bits.type,
                        math::firstBitIndex(cons_mask),
                        math::firstBitIndex(prod_mask)
                    );

                    auto cfg = static_cast<HglDmaConfigBits>(transaction->cfg_link.cfg_bits);
                    nnLog(MVLOG_DEBUG , "type: %x burst_length: %x critical: %x interrupt_en: %x interrupt_trigger: %x skip_nr: %x order_forced: %x watermark_en: %x dec_en: %x barrier_en: %x",
                    cfg.type, cfg.burst_length, cfg.critical, cfg.interrupt_en, cfg.interrupt_trigger, cfg.skip_nr, cfg.order_forced, cfg.watermark_en, cfg.dec_en, cfg.barrier_en);

                    auto attrs_2d = static_cast<HglDma2DAttributes>(transaction->attr2d);
                    nnLog(MVLOG_DEBUG , "src_width: %x src_stride: %x dst_width: %x dst_stride: %x", attrs_2d.src_width, attrs_2d.src_stride, attrs_2d.dst_width, attrs_2d.dst_stride);

                    auto barriers1d = static_cast<HglDmaBarrierCfg>(transaction->barriers1d);
                    nnLog(MVLOG_DEBUG , "1d.pm: %x 1d.cm: %x", barriers1d.prod_mask, barriers1d.cons_mask);

                    auto barriers = static_cast<HglDmaBarrierCfg>(transaction->barriers);
                    nnLog(MVLOG_DEBUG , "2d.pm: %x 2d.cm: %x", barriers.prod_mask, barriers.cons_mask);
                }
            }

            rtems_interrupt_lock_release(&dma_feeder_lock_, &lock_ctx);
        }

        FasterDmaFeeder::Handle *FasterDmaFeeder::dma_callback(Handle *handle)
        {
            auto &finished = wrappers_[available_ ^ 1];
            assert(handle == &finished.handle_ && "Received unexpected handle");

            if (callback_)
                callback_(user_context_, handle, finished.transaction_count_);

            finished.transaction_count_ = 0;

            Handle *next = nullptr;
            auto &candidate = wrappers_[available_];

            rtems_interrupt_lock_context lock_ctx;
            rtems_interrupt_lock_acquire(&dma_feeder_lock_, &lock_ctx);

            if (candidate.transaction_count_ > 0)
            {
                next = &candidate.handle_;
                available_ ^= 1;
            }
            else
            {
                active_ = false;
            }

            rtems_interrupt_lock_release(&dma_feeder_lock_, &lock_ctx);

            return next;
        }

        void FasterDmaFeeder::dma_callback(DmaJobHandleType *current_handle, void *user_context, DmaJobHandleType **next_handle)
        {
            if (user_context) {
                auto next = reinterpret_cast<FasterDmaFeeder *>(user_context)->dma_callback(current_handle);
                *next_handle = next;
            } else {
                nnLog(MVLOG_ERROR, "DMA returned a nullptr for user_context");
            }
        }
    }
}
