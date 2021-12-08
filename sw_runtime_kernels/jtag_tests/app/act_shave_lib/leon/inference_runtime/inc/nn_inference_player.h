/*
 * {% copyright %}
 */
#ifndef NN_INFERENCE_PLAYER_H_
#define NN_INFERENCE_PLAYER_H_

#include <nn_inference_runtime_types.h>
#include <nn_ipc.h>
#include <nn_memory.h>
#include <nn_thread.h>
#include <nn_shave_manager.h>
#include <nn_hw_resources.h>
#include <nn_runtime_configs.h>
#include <nn_resource_manager.h>
#include <nn_resource_locator.h>
#include <nn_task_context.h>
#include <nn_dma_feeder.h>
#include <Dma.h>
#include <Barrier.h>
#include <array>
#include <rtems.h>
#include <nn_time.h>

namespace nn {
namespace inference_runtime {
class Player {
public:
    Player(const time::Ticker &ticker_, unsigned char tid);

    InferenceRequest::Code play(const ResourceRequirements &resources, const MappedInference &mapped,
                                const RuntimeMapping &hw, InferenceRequest &request,
                                const UserContextInfo &context_info);

    void reset();

    const unsigned int *timestamp() const { return &timestamp_; }

    static void barrierCallback(unsigned char barrier, BarrierType type, void *context);

private:
    struct ProcessIterator {
        unsigned int copy_index_;
        unsigned int start_index_;
        unsigned int complete_index_;

        ProcessIterator()
            : copy_index_(0)
            , start_index_(0)
            , complete_index_(0) {}
    };

    class MetadataFeeder {
    public:
        typedef DmaDescriptorConfig Transaction;
        explicit MetadataFeeder(ProcessIterator &iterator);
        Transaction *reset(const void *remote, unsigned int remote_count, void *local, unsigned int local_count,
                           unsigned int item_size);

        bool transferred(FasterDmaFeeder::HandleWrapper &wrapper);
        Transaction *copy(unsigned int count);
        Transaction *copy() { return copy(local_count_ >> 1); }

        void setAperturedDescriptor(DmaDescriptor *t) { transaction_ = t; }
        DmaDescriptor *getPhysDescriptor() const;
        void setApertureOffset(uint32_t aperture_offset) { aperture_offset_ = aperture_offset; }

    private:
        DmaDescriptorConfig descConfig_;
        DmaDescriptor *transaction_;
        ProcessIterator &iterator_;
        uint32_t aperture_offset_;
        const unsigned char *remote_;
        unsigned int remote_count_;
        unsigned int remote_index_;
        unsigned char *local_;
        unsigned int local_count_;
        unsigned int local_index_;
        unsigned int copy_count_;
        unsigned int item_size_;
    };

    enum {
        SHV_STORE_SIZE = 4,
        SHV_QUEUE_SIZE = 1,
    };

    unsigned int timestamp_;
    const time::Ticker &ticker_;
    UserContextInfo currentContextInfo_;
    bool inContextErrorState;
    const ResourceRequirements *resources_;
    const MappedInference *mapped_;
    unsigned int activations_;
    BarrierNotificationCbEn barrierNotification_;
    bool progressed_;
    unsigned char tid_;
    std::array<bool, MAX_DMA_ENGINES> leading_dma_complete_;
    ProcessIterator bar_i_, dma_i_[MAX_DMA_ENGINES], inv_i_, var_i_, /*akr_i_,*/ aki_i_;

    FasterDmaFeeder task_feeder_;
    std::array<FasterDmaFeeder, MAX_DMA_ENGINES> data_feeder_;

    std::array<MetadataFeeder, MAX_DMA_ENGINES> dmaFeeder_;
    MetadataFeeder invFeeder_;
    MetadataFeeder varFeeder_;
    // MetadataFeeder akrFeeder_;
    MetadataFeeder akiFeeder_;
    std::array<MetadataFeeder *, NUM_METADATA_FEEDERS> feeders_;

    const BarrierCfg *barrier_configs_;
    std::array<short, common_runtime::TOTAL_PHYSICAL_BARRIERS> barrier_to_virtual_id_;
    std::array<short, common_runtime::TOTAL_PHYSICAL_BARRIERS> vid_produced_by_barrier_;
    RuntimeMapping hw_;
    time::Ticker perfTicker_;
    InferenceRequest request_;

    InferenceRequest::Code play_();

    inline bool is_barrier_produced(unsigned short vid) const;
    void start_barriers();

    void start_leading_dma_tasks();
    void start_dma_tasks();

    void start_dpu_tasks();
    void free_dpu_tasks();

    void start_act_tasks();
    void free_act_tasks();

    void print();
    void debug_info() const;

    void taskFeederCallback(FasterDmaFeeder::Handle *handle, unsigned int transaction_count);
    static void taskFeederCallback(void *user_context, FasterDmaFeeder::Handle *handle, unsigned int transaction_count);

    template <unsigned int e>
    void dataFeederCallback(FasterDmaFeeder::Handle *, unsigned int transaction_count) {
        if (leading_dma_complete_[e])
            dma_i_[e].complete_index_ += transaction_count;

        progressed_ = true;
        leading_dma_complete_[e] = true;
    }

    template <unsigned int e>
    static void dataFeederCallback(void *user_context, FasterDmaFeeder::Handle *handle,
                                   unsigned int transaction_count) {
        if (Player *player = reinterpret_cast<Player *>(user_context))
            player->dataFeederCallback<e>(handle, transaction_count);
    }

    void barrierCallback(unsigned char barrier, BarrierType type);
};
} // namespace inference_runtime
} // namespace nn

#endif // NN_INFERENCE_PLAYER_H_
