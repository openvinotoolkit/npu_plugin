///*
// * {% copyright %}
// */
//#include "nn_inference_player.h"
//#include <nn_math.h>
//#include <nn_cache.h>
//#include <nn_fifo_manager.h>
//#include <assert.h>
//#include <HglBarrier.h>
//#include <Barrier.h>
//#include <nn_barrier.h>
//#include <nn_log.h>
//#include <nn_nce_lib_ir.h>
//#include <pipePrintInit.h>
//#include <nn_context_manager.h>
//#include <Dma.h>
//
//#ifdef NN_PROFILING
//#include <barectf-myriad.h>
//#else
//#define BARECTF(...)
//#define BARECTF_ISR(...)
//#endif
//
//#ifdef NN_PROFILING_ALL
//#define BARECTF_ALL BARECTF
//#define BARECTF_ISR_ALL BARECTF_ISR
//#else
//#define BARECTF_ALL(...)
//#define BARECTF_ISR_ALL(...)
//#endif
//
//using namespace std;
//
//namespace {
//using namespace nn::inference_runtime;
//using namespace nn::common_runtime;
//
//inline void dma_create(DmaDescriptorConfig &descCfg, const void *src, void *dst, unsigned int length) {
//    descCfg.src = (uint64_t)(uint32_t)src;
//    descCfg.dst = (uint64_t)(uint32_t)dst;
//    descCfg.size = length;
//    descCfg.src_width = descCfg.dst_width = 0;
//    descCfg.src_stride = descCfg.dst_stride = 0;
//    descCfg.num_planes = DMA_DEFAULT_PLANE_NUM;
//    descCfg.burst_size = DMA_DEFAULT_BURST_SIZE;
//    descCfg.feature_flags = HGL_DMA_FLAG_WATERMARK_EN | HGL_DMA_FLAG_CRITICAL_TASK;
//    descCfg.barrier_cfg = nullptr;
//}
//
//void show_barrier_state(unsigned int line, BarrierStatus result, unsigned char barrier) {
//    nnLog(MVLOG_ERROR, "BarrierConfig:%u returned %d for barrier %u", line, result, barrier);
//
//    unsigned short prod = 0, cons = 0;
//    result = BarrierGetCount(barrier, &prod, &cons);
//    nnLog(MVLOG_ERROR, "BarrierGetCount(%u, %u, %u)", barrier, prod, cons);
//    assert(result == BARRIER_SUCCESS);
//
//    bool phit = false, chit = false;
//    result = BarrierGetStatus(barrier, &phit, &chit);
//    nnLog(MVLOG_ERROR, "BarrierGetStatus(%u, %u, %u)", barrier, phit ? 1 : 0, chit ? 1 : 0);
//    assert(result == BARRIER_SUCCESS);
//}
//} // namespace
//
//namespace nn {
//namespace inference_runtime {
//using namespace dpu_runtime;
//using namespace inference_context;
//
//Player::MetadataFeeder::MetadataFeeder(ProcessIterator &iterator)
//    : transaction_(nullptr)
//    , iterator_(iterator)
//    , aperture_offset_(0)
//    , remote_(reinterpret_cast<const unsigned char *>(&remote_))
//    , remote_count_(0)
//    , remote_index_(0)
//    , local_(reinterpret_cast<unsigned char *>(&local_))
//    , local_count_(0)
//    , local_index_(0)
//    , copy_count_(0)
//    , item_size_(0) {}
//
//DmaDescriptorConfig *Player::MetadataFeeder::reset(const void *remote, unsigned int remote_count, void *local,
//                                                   unsigned int local_count, unsigned int item_size) {
//    nnLog(MVLOG_INFO, "%p, %u, %p, %u, %u", remote, remote_count, local, local_count, item_size);
//
//    iterator_ = ProcessIterator();
//    remote_ = reinterpret_cast<const unsigned char *>(remote);
//    remote_count_ = remote_count;
//    remote_index_ = 0;
//    local_ = reinterpret_cast<unsigned char *>(local);
//    local_count_ = local_count;
//    local_index_ = 0;
//    copy_count_ = 0;
//    item_size_ = item_size;
//
//    return copy(local_count_);
//}
//
//bool Player::MetadataFeeder::transferred(FasterDmaFeeder::HandleWrapper &wrapper) {
//    if (copy_count_ != 0) {
//        bool wtm = 0;
//        auto rc = DmaDescIsWatermarkSet(&wrapper.handle_, getPhysDescriptor(), &wtm);
//        if (rc != DMA_SUCCESS || !wtm)
//            return false;
//
//        cache::invalidate(local_ + local_index_ * item_size_,
//                          math::round_up<NN_CACHE_LINE_LENGTH>(transaction_->length));
//
//        remote_index_ += copy_count_;
//        local_index_ += copy_count_;
//
//        if (local_count_ <= local_index_)
//            local_index_ -= local_count_;
//
//        iterator_.copy_index_ += copy_count_;
//        copy_count_ = 0;
//
//        return true;
//    }
//
//    return false;
//}
//
//DmaDescriptorConfig *Player::MetadataFeeder::copy(unsigned int count) {
//    if (copy_count_ == 0 &&                                                        // not busy
//        remote_index_ < remote_count_ &&                                           // there are more items to copy
//        iterator_.copy_index_ + count <= iterator_.complete_index_ + local_count_) // there is space to store them
//    {
//        copy_count_ = std::min(count, remote_count_ - remote_index_);
//
//        dma_create(descConfig_, remote_ - aperture_offset_ + remote_index_ * item_size_,
//                   local_ + local_index_ * item_size_, item_size_ * copy_count_);
//
//        return &descConfig_;
//    }
//
//    return nullptr;
//}
//
//DmaDescriptor *Player::MetadataFeeder::getPhysDescriptor() const {
//    unsigned char *apertured_desc = reinterpret_cast<unsigned char *>(transaction_);
//    return reinterpret_cast<DmaDescriptor *>(apertured_desc - aperture_offset_);
//}
//
//Player::Player(const time::Ticker &ticker, unsigned char tid) :
//            timestamp_(0),
//            ticker_(ticker),
//            currentContextInfo_(),
//            inContextErrorState{false},
//            resources_(nullptr),
//            mapped_(nullptr),
//            activations_(0),
//            barrierNotification_{ .cb_en = 0 },
//            progressed_(true),
//            tid_(tid),
//            leading_dma_complete_
//            {
//                false,
//                false,
//            },
//            bar_i_(),
//            dma_i_(),
//            inv_i_(),
//            var_i_(),
//            // akr_i_(),
//            aki_i_(),
//            task_feeder_(1 % DMA_ENGINES, static_cast<unsigned char>(2 * tid)),
//            data_feeder_
//            {
//                FasterDmaFeeder(0, static_cast<unsigned char>(2 * tid + 1)),
//                FasterDmaFeeder(1, static_cast<unsigned char>(2 * tid + 1)),
//            },
//            dmaFeeder_
//            {
//                MetadataFeeder(dma_i_[0]),
//                MetadataFeeder(dma_i_[1]),
//            },
//            invFeeder_(inv_i_),
//            varFeeder_(var_i_),
//            // akrFeeder_(akr_i_),
//            akiFeeder_(aki_i_),
//            feeders_
//            {
//                &dmaFeeder_[0],
//                &dmaFeeder_[1],
//                &invFeeder_,
//                &varFeeder_,
//                // &akrFeeder_,
//                &akiFeeder_
//            },
//            barrier_configs_(nullptr),
//            barrier_to_virtual_id_(),
//            vid_produced_by_barrier_(),
//            hw_(),
//            perfTicker_(),
//            request_()
//{
//    barrierNotification_.consumer = 1;
//    barrierNotification_.producer = 1;
//
//    data_feeder_[0].set_callback(&dataFeederCallback<0>, this);
//    data_feeder_[1].set_callback(&dataFeederCallback<1>, this);
//
//    cache::flush(&timestamp_, sizeof(timestamp_));
//
//    reset();
//}
//
//InferenceRequest::Code Player::play(const ResourceRequirements &resources, const MappedInference &mapped,
//                                    const RuntimeMapping &hw, InferenceRequest &request,
//                                    const UserContextInfo &context_info) {
//    resources_ = &resources;
//    mapped_ = &mapped;
//    hw_ = hw;
//    request_ = request;
//    currentContextInfo_ = context_info;
//
//    nnLog(MVLOG_INFO, "Counts: DMAs: %d, Invariants: %d, Variants: %d, AKR: %d, AKI: %d, Barriers: %d",
//          mapped_->dmaTasks_.size(), mapped_->invariants_.size(), mapped_->variants_.size(),
//          mapped_->actKRanges_.size(), mapped_->actKInvocations_.size(), mapped_->barrierConfigs_.size());
//    nnLog(MVLOG_INFO, "Storage requirements: DMAs: %u, Invariants: %u, Variants: %u, AKI: %u, Barriers %u, Player %u",
//          (hw_.dma_[0].count() + hw_.dma_[1].count()) * sizeof(backend::DMATask),
//          hw_.inv_.count() * sizeof(backend::DPUInvariantWrapper),
//          hw_.var_.count() * sizeof(backend::DPUVariantWrapper),
//          //   hw_.akr_.count() * sizeof(backend::ActKernelRangeWrapper),
//          hw_.aki_.count() * sizeof(backend::ActKernelInvocationWrapper),
//          common_runtime::MAX_BARRIERS_PER_INFERENCE * 2 * sizeof(short), sizeof(Player));
//    nnLog(MVLOG_INFO, "SSID: %d", currentContextInfo_.ssid_);
//    nnLog(MVLOG_INFO, "Aperture Offset: %lx\n", currentContextInfo_.aperture_offset_);
//
//    BARECTF_ALL(leonnn_trace_resources_locked, (uint32_t)this, mapped_->resource_requirements_.nn_slice_count_);
//
//    // if (!shave_manager_.assignClusters(hw_.config_))
//    //     return RuntimeResult::OUT_OF_RESOURCES;
//
//    // Initialize metadata feeders with the descriptors in mapped user context
//    // Whole mapped inference is const, so remove it just for these descriptors
//    // and keep const safety elsewhere
//    for (uint32_t i = 0; i < NUM_METADATA_FEEDERS; i++) {
//        feeders_[i]->setApertureOffset(currentContextInfo_.aperture_offset_);
//        feeders_[i]->setAperturedDescriptor(const_cast<DmaDescriptor *>(&mapped_->feederDescriptors_[i]));
//    }
//
//    // Set the DMA context ID for this inference execution
//    task_feeder_.set_context_SSID(currentContextInfo_.ssid_);
//
//    for (auto &feeder : data_feeder_)
//        feeder.set_context_SSID(currentContextInfo_.ssid_);
//
//    barrier_configs_ = mapped_->barrierConfigs_.data();
//    bar_i_.copy_index_ = mapped_->barrierConfigs_.size();
//
//    BARECTF_ALL(leonnn_trace_inference_started, (uint32_t)this);
//
//    auto res = play_();
//
//    BARECTF_ALL(leonnn_trace_inference_finished, (uint32_t)this);
//
//    if (request_.request_.barrier_lift_times_)
//        cache::flush(request_.request_.barrier_lift_times_, sizeof(unsigned int) * mapped_->barrierConfigs_.size());
//
//    if (request_.request_.barrier_free_times_)
//        cache::flush(request_.request_.barrier_free_times_, sizeof(unsigned int) * mapped_->barrierConfigs_.size());
//
//    return res;
//}
//
//void Player::reset() {
//    bar_i_ = ProcessIterator();
//    fill(barrier_to_virtual_id_.begin(), barrier_to_virtual_id_.end(), -1);
//    fill(vid_produced_by_barrier_.begin(), vid_produced_by_barrier_.end(), -1);
//}
//
//InferenceRequest::Code Player::play_() {
//    nnLog(MVLOG_INFO, "NN::IR::Player: inference start");
//
//    static ViolationState ctxViolState;
//    FasterDmaFeeder::HandleWrapper wrapper;
//    wrapper.setApertureOffset(currentContextInfo_.aperture_offset_);
//
//    if (auto *task = invFeeder_.reset(mapped_->invariants_.data(), mapped_->invariants_.size(), hw_.inv_.tasks(),
//                                      hw_.inv_.count(), sizeof(backend::DPUInvariantWrapper)))
//        wrapper.append(task, invFeeder_.getPhysDescriptor());
//
//    if (auto *task = varFeeder_.reset(mapped_->variants_.data(), mapped_->variants_.size(), hw_.var_.tasks(),
//                                      hw_.var_.count(), sizeof(backend::DPUVariantWrapper)))
//        wrapper.append(task, varFeeder_.getPhysDescriptor());
//
//    // TODO: prove out working kernels without needing akr access, then delete all the akr stuff
//    // if (auto *task = akrFeeder_.reset(mapped_->actKRanges_.data(), mapped_->actKRanges_.size(), hw_.akr_.tasks(),
//    //                                   hw_.akr_.count(), sizeof(backend::ActKernelRangeWrapper)))
//    //     wrapper.append(task, akrFeeder_.getPhysDescriptor());
//
//    if (auto *task = akiFeeder_.reset(mapped_->actKInvocations_.data(), mapped_->actKInvocations_.size(),
//                                      hw_.aki_.tasks(), hw_.aki_.count(), sizeof(backend::ActKernelInvocationWrapper)))
//        wrapper.append(task, akiFeeder_.getPhysDescriptor());
//
//    task_feeder_.enqueue(wrapper);
//    wrapper.reset();
//
//    start_barriers();
//    start_leading_dma_tasks();
//
//    for (unsigned int e = 0; e < MAX_DMA_ENGINES; ++e)
//        if (auto *task = dmaFeeder_[e].reset(mapped_->dmaTasks_[e].data() + mapped_->leadingDmaTasks_[e],
//                                             mapped_->dmaTasks_[e].size() - mapped_->leadingDmaTasks_[e],
//                                             hw_.dma_[e].tasks(), hw_.dma_[e].count(), sizeof(backend::DMATask)))
//            wrapper.append(task, dmaFeeder_[e].getPhysDescriptor());
//
//    task_feeder_.enqueue(wrapper);
//    wrapper.reset();
//
//    for (activations_ = 0; mapped_->leadingDmaTasks_[0] + dma_i_[0].complete_index_ < mapped_->dmaTasks_[0].size() ||
//                           mapped_->leadingDmaTasks_[1] + dma_i_[1].complete_index_ < mapped_->dmaTasks_[1].size() ||
//                           var_i_.complete_index_ < mapped_->variants_.size() ||
//                           aki_i_.complete_index_ < mapped_->actKInvocations_.size();
//         ++activations_, (progressed_ ? progressed_ = false, print() : util::Thread::yield())) {
//        BARECTF_ALL(leonnn_trace_player_iteration_started, activations_);
//
//        free_dpu_tasks();
//        free_act_tasks();
//
//        BARECTF_ALL(leonnn_trace_player_fill_levels, activations_, dma_i_.start_index_ - dma_i_.complete_index_,
//                    var_i_.start_index_ - var_i_.complete_index_, aki_i_.start_index_ - aki_i_.complete_index_,
//                    bar_i_.start_index_ - bar_i_.complete_index_);
//
//        for (auto *feeder : feeders_)
//            progressed_ |= feeder->transferred(wrapper);
//
//        start_dma_tasks();
//        start_dpu_tasks();
//        start_act_tasks();
//
//        for (auto *feeder : feeders_)
//            if (auto *task = feeder->copy())
//                wrapper.append(task, feeder->getPhysDescriptor());
//
//        task_feeder_.enqueue(wrapper);
//        wrapper.reset();
//
//        BARECTF_ALL(leonnn_trace_player_iteration_finished, activations_);
//
//#ifdef NN_IR_VERBOSE_STALLS
//        if (activations_ >> 22) {
//            debug_info();
//            activations_ = 0;
//        }
//#endif
//        if (context::check_and_record_context_violation(ctxViolState)) {
//            if (ctxViolState.cid == currentContextInfo_.ssid_) {
//                inContextErrorState = true;
//
//                context::context_violation_irq_clear(ctxViolState.cid_viol_bitfield);
//                context::print_context_violation(ctxViolState);
//                // Flush pipeprint or we probably won't see the error message
//                leonPipePrintFlushBuffer();
//
//                return InferenceRequest::CONTEXT_VIOLATION;
//            }
//        }
//    }
//
//    // Fathom generates a trailing barrier to signal the whole inference is complete and to be consumed by the runtime
//    // dispatcher. Don't need ir for now, but the consumer count still has to be decremented to avoid surprises in the
//    // next inference.
//    if (bar_i_.copy_index_ > 0) {
//        auto physical = barrier_configs_[bar_i_.copy_index_ - 1].real_id_;
//        unsigned short consumers = 0;
//        util::getBarrierCounts(physical, nullptr, &consumers);
//
//        if (consumers > 0)
//            HglBarrierConsume(1ull << physical);
//    }
//
//    return InferenceRequest::COMPLETE;
//}
//
//bool Player::is_barrier_produced(unsigned short vid) const {
//    assert(vid < mapped_->barrierConfigs_.size());
//    return vid <= vid_produced_by_barrier_[barrier_configs_[vid].real_id_];
//}
//
//void Player::start_barriers() {
//    for (; bar_i_.start_index_ < resources_->nn_barriers_ && bar_i_.start_index_ < bar_i_.copy_index_;
//         ++bar_i_.start_index_) {
//        const auto &config = barrier_configs_[bar_i_.start_index_];
//        auto &vid = barrier_to_virtual_id_[config.real_id_];
//
//        vid = static_cast<short>(bar_i_.start_index_);
//
//        auto result =
//            BarrierConfig(config.real_id_, config.producer_count_, config.consumer_count_, barrierNotification_, this);
//
//        if (result != BARRIER_SUCCESS)
//            show_barrier_state(__LINE__, result, config.real_id_);
//
//        assert(result == BARRIER_SUCCESS);
//
//        BARECTF_ALL(leonnn_trace_barrier_configured, bar_i_.start_index_, config.real_id_, config.producer_count_,
//                    config.producer_count_);
//    }
//}
//
//void Player::start_leading_dma_tasks() {
//    for (unsigned int e = 0; e < MAX_DMA_ENGINES; ++e) {
//        if (mapped_->leadingDmaTasks_[e] > 0) {
//            leading_dma_complete_[e] = false;
//
//            FasterDmaFeeder::HandleWrapper wrapper;
//            wrapper.setApertureOffset(currentContextInfo_.aperture_offset_);
//            wrapper.init(
//                const_cast<DmaDescriptor *>(&mapped_->dmaTasks_[e][0].transaction_),
//                const_cast<DmaDescriptor *>(&mapped_->dmaTasks_[e][wrapper.transaction_count_ - 1].transaction_),
//                mapped_->leadingDmaTasks_[e]);
//
//            data_feeder_[e].enqueue(wrapper);
//        } else
//            leading_dma_complete_[e] = true;
//    }
//}
//
//void Player::start_dma_tasks() {
//    for (unsigned int e = 0; e < MAX_DMA_ENGINES; ++e) {
//        FasterDmaFeeder::HandleWrapper wrapper;
//        unsigned int before_start = dma_i_[e].start_index_;
//
//        for (; dma_i_[e].start_index_ < dma_i_[e].copy_index_; progressed_ = true, ++dma_i_[e].start_index_) {
//            auto &local_dma = hw_.dma_[e].task(dma_i_[e].start_index_);
//
//            if (bar_i_.start_index_ < local_dma.barriers_.start_after_)
//                break;
//
//            nnLog(MVLOG_DEBUG,
//                  "Starting up DMA %u on engine %u from 0x%llx to 0x%llx with %u bytes, wait mask: %llx, post mask: "
//                  "%llx, after: %u",
//                  dma_i_[e].start_index_, e, local_dma.transaction_.src, local_dma.transaction_.dst,
//                  local_dma.transaction_.length, local_dma.barriers_.wait_mask_, local_dma.barriers_.post_mask_,
//                  local_dma.barriers_.start_after_);
//
//            BARECTF_ALL(leonnn_trace_dma_task_pushed, dma_i_.start_index_);
//        }
//
//        if (before_start < dma_i_[e].start_index_) {
//            wrapper.init(&hw_.dma_[e].task(before_start).transaction_,
//                         &hw_.dma_[e].task(dma_i_[e].start_index_ - 1).transaction_,
//                         dma_i_[e].start_index_ - before_start);
//
//            data_feeder_[e].enqueue(wrapper);
//        }
//    }
//}
//
//void Player::start_dpu_tasks() {
//    for (; inv_i_.start_index_ < inv_i_.copy_index_; progressed_ = true, ++inv_i_.start_index_) {
//        const auto &local_inv = hw_.inv_.task(inv_i_.start_index_);
//
//        if (bar_i_.start_index_ < local_inv.invariant_.barriers_.start_after_)
//            break;
//    }
//
//    for (; var_i_.start_index_ < var_i_.copy_index_; progressed_ = true, ++var_i_.start_index_) {
//        auto &local_var = hw_.var_.task(var_i_.start_index_);
//
//        if (inv_i_.start_index_ <= local_var.invariant_index_)
//            break; // corresponding invariant is not ready to be used
//
//#ifdef NN_PRINT_DPU_REGISTERS
//        nn::nce_lib::DebugPrintRegister(local_var.variant_);
//        // Add a delay here to allow for all of the prints to complete from
//        // the previous workload
//        sleep(2);
//#endif
//
//        assert(local_var.variant_.cluster_ < FIFO_COUNT);
//
//        if (fifo::isSNNWorkFifoFull(local_var.variant_.cluster_))
//            break;
//
//        fifo::sendWorkToSNNs(local_var.variant_.cluster_, &local_var.variant_);
//
//        BARECTF_ALL(leonnn_trace_workload_pushed, local_inv.invariant_.cluster_, local_var.invariant_index_,
//                    var_i_.start_index_, (uint32_t)&local_var.variant_);
//    }
//}
//
//void Player::free_dpu_tasks() {
//    for (; inv_i_.complete_index_ < inv_i_.start_index_; progressed_ = true, ++inv_i_.complete_index_) {
//        auto &local_inv = hw_.inv_.task(inv_i_.complete_index_);
//
//        if (!is_barrier_produced(local_inv.invariant_.barriers_.clean_after_))
//            break;
//
//        var_i_.complete_index_ += local_inv.variant_count_;
//
//        BARECTF_ALL(leonnn_trace_workload_done, local_inv.invariant_.cluster_, inv_i_.complete_index_, -1, 0);
//
//#ifdef NN_DUMP_INTERMEDIATE_BUFFERS
//        nce_lib::dump_output(inv_i_.complete_index_, local_inv.invariant_);
//#endif
//    }
//}
//
//void Player::start_act_tasks() {
//    for (; aki_i_.start_index_ < aki_i_.copy_index_; progressed_ = true, ++aki_i_.start_index_) {
//        auto &local_aki = hw_.aki_.task(aki_i_.start_index_);
//
//        /*
//            Note: the local_aki.kInvo_'s reference to its range_ is a pointer to DDR.
//            It is expected that the ActSHV's caches will manage that access better than
//            if space were waisted to buffer the ActKernelRange in local CMX like we are
//            doing with local_aki above.
//        */
//
//        if (bar_i_.start_index_ < local_aki.kInvo_.barriers_.start_after_)
//            break;
//
//        nnLog(MVLOG_DEBUG, "Check Act work FIFO for tile %d", local_aki.tile_);
//        if (fifo::isASWorkFifoFull(local_aki.tile_))
//            break;
//
//        nnLog(MVLOG_DEBUG, "Sending act workload %p", &local_aki.kInvo_);
//        nnLog(MVLOG_DEBUG, "Act workload kernel window %p", &local_aki.kInvo_.range_->kernelEntry_);
//        nnLog(MVLOG_DEBUG, "Act workload kernel entry %p", &local_aki.kInvo_.range_->textWindowBase_);
//
//        fifo::sendWorkToASs(local_aki.tile_, &local_aki.kInvo_);
//
//        BARECTF_ALL(leonnn_trace_workload_pushed, local_aki.tile_, local_aki.kInvo_.task_index_, aki_i_.start_index_,
//                    (uint32_t)&local_aki.kInvo_);
//    }
//}
//
//void Player::free_act_tasks() {
//    for (; aki_i_.complete_index_ < aki_i_.start_index_; progressed_ = true, ++aki_i_.complete_index_) {
//        auto &local_aki = hw_.aki_.task(aki_i_.complete_index_);
//        UNUSED(local_aki);
//
//        if (!is_barrier_produced(local_aki.kInvo_.barriers_.clean_after_))
//            break;
//
//        aki_i_.complete_index_ += local_aki.kInvo_.invo_index_;
//
//        BARECTF_ALL(leonnn_trace_workload_done, local_aki.tile, aki_i_.complete_index_, -1, 0);
//    }
//}
//
//void Player::print() {
//    nnLog(MVLOG_INFO,
//          "DMA: (%u) %u / %u / %u / %u, (%u) %u / %u / %u / %u; INV: %u / %u / %u / %u; VAR: %u / %u / %u / %u; "
//          "AKI: %u / %u / %u / %u; BAR: - / %u / %u / %u",
//          leading_dma_complete_[0] ? mapped_->leadingDmaTasks_[0] : 0, dma_i_[0].complete_index_,
//          dma_i_[0].start_index_, dma_i_[0].copy_index_, mapped_->dmaTasks_[0].size() - mapped_->leadingDmaTasks_[0],
//          leading_dma_complete_[1] ? mapped_->leadingDmaTasks_[1] : 0, dma_i_[1].complete_index_,
//          dma_i_[1].start_index_, dma_i_[1].copy_index_, mapped_->dmaTasks_[1].size() - mapped_->leadingDmaTasks_[1],
//          inv_i_.complete_index_, inv_i_.start_index_, inv_i_.copy_index_, mapped_->invariants_.size(),
//          var_i_.complete_index_, var_i_.start_index_, var_i_.copy_index_, mapped_->variants_.size(),
//          aki_i_.complete_index_, aki_i_.start_index_, aki_i_.copy_index_, mapped_->actKInvocations_.size(),
//          bar_i_.start_index_, bar_i_.copy_index_, mapped_->barrierConfigs_.size());
//}
//
//void Player::debug_info() const {
//    nnLog(MVLOG_WARN, "Inference stalling on thread %u, debug info follows for Player %p:", tid_, this);
//
//    nnLog(MVLOG_WARN,
//          "DMA: (%u) %u / %u / %u / %u, (%u) %u / %u / %u / %u; INV: %u / %u / %u / %u; VAR: %u / %u / %u / %u; "
//          "AKI: %u / %u / %u / %u; BAR: - / %u / %u / %u",
//          leading_dma_complete_[0] ? mapped_->leadingDmaTasks_[0] : 0, dma_i_[0].complete_index_,
//          dma_i_[0].start_index_, dma_i_[0].copy_index_, mapped_->dmaTasks_[0].size() - mapped_->leadingDmaTasks_[0],
//          leading_dma_complete_[1] ? mapped_->leadingDmaTasks_[1] : 0, dma_i_[1].complete_index_,
//          dma_i_[1].start_index_, dma_i_[1].copy_index_, mapped_->dmaTasks_[1].size() - mapped_->leadingDmaTasks_[1],
//          inv_i_.complete_index_, inv_i_.start_index_, inv_i_.copy_index_, mapped_->invariants_.size(),
//          var_i_.complete_index_, var_i_.start_index_, var_i_.copy_index_, mapped_->variants_.size(),
//          aki_i_.complete_index_, aki_i_.start_index_, aki_i_.copy_index_, mapped_->actKInvocations_.size(),
//          bar_i_.start_index_, bar_i_.copy_index_, mapped_->barrierConfigs_.size());
//
//    nnLog(MVLOG_WARN, "");
//    nnLog(MVLOG_WARN, "Barrier configuration:");
//
//    for (unsigned char i = 0; i < barrier_to_virtual_id_.size(); ++i) {
//        const auto &vid = barrier_to_virtual_id_[i];
//
//        if (vid >= 0) {
//            unsigned short prod = 0, cons = 0;
//            auto result = BarrierGetCount(i, &prod, &cons);
//            assert(result == BARRIER_SUCCESS);
//
//            UNUSED(result);
//
//            nnLog(MVLOG_WARN, "%2u: Virtual %3u, P: %2d, C: %2d, Original: P: %2d, C: %2d", i, vid, prod, cons,
//                  barrier_configs_[vid].producer_count_, barrier_configs_[vid].consumer_count_);
//        }
//    }
//
//    nnLog(MVLOG_WARN, "");
//    nnLog(MVLOG_WARN, "DmaFeeders:");
//
//    task_feeder_.debug_info();
//
//    for (unsigned int e = 0; e < MAX_DMA_ENGINES; ++e) {
//        data_feeder_[e].debug_info();
//
//        nnLog(MVLOG_WARN, "");
//        nnLog(MVLOG_WARN, "Pending DMA tasks on engine %u:", e);
//
//        for (unsigned int i = dma_i_[e].complete_index_; i < dma_i_[e].start_index_; ++i) {
//            const auto &task = hw_.dma_[e].task(i);
//            const auto *transaction = &task.transaction_;
//
//            nnLog(MVLOG_WARN,
//                  "%u: Transaction: %p, src: %llx, dst: %llx, length: %8u, wait_mask: %08x, post_mask: %08x, "
//                  "start_after: %3u wtm: %d",
//                  i, transaction, transaction->src, transaction->dst, transaction->length, task.barriers_.wait_mask_,
//                  task.barriers_.post_mask_, task.barriers_.start_after_, transaction->watermark);
//        }
//    }
//
//    nnLog(MVLOG_WARN, "");
//    nnLog(MVLOG_WARN, "Pending DPU tasks (invariants):");
//
//    for (unsigned int i = inv_i_.complete_index_; i < inv_i_.start_index_; ++i) {
//        const auto &task = hw_.inv_.task(i);
//
//        nnLog(MVLOG_WARN,
//              "%u: @ %p, cluster: %u, variant_count: %u, wait_mask: %08x, post_mask: %08x, start_after: %3u, "
//              "clean_after: %3u",
//              i, &task.invariant_, task.invariant_.cluster_, task.variant_count_, task.invariant_.barriers_.wait_mask_,
//              task.invariant_.barriers_.post_mask_, task.invariant_.barriers_.start_after_,
//              task.invariant_.barriers_.clean_after_);
//    }
//
//    nnLog(MVLOG_WARN, "");
//}
//
//void Player::barrierCallback(unsigned char barrier, BarrierType type) {
//    // Sometimes barriers get produced and consumed between the ISR checking the production and consumption states.
//    // This leads to possibly not getting a production callback, but only a consumption callback.
//    // Make sure to mark VID in either case.
//
//    if (inContextErrorState)
//        return;
//
//    auto &vid = barrier_to_virtual_id_[barrier];
//    assert(0 <= vid && vid < static_cast<short>(mapped_->barrierConfigs_.size()) &&
//           "Cannot trace back real barrier id to virtual id");
//
//    const auto &config = barrier_configs_[vid];
//
//    if (type == BARRIER_CONSUMER) {
//        if (request_.request_.barrier_free_times_)
//            request_.request_.barrier_free_times_[vid] = static_cast<unsigned int>(ticker_.ticks());
//
//        if (0 <= config.next_same_id_) {
//            const auto &next_config = barrier_configs_[config.next_same_id_];
//            assert((next_config.real_id_) == barrier);
//
//            auto result = BarrierConfig(barrier, next_config.producer_count_, next_config.consumer_count_,
//                                        barrierNotification_, this);
//
//            if (result != BARRIER_SUCCESS)
//                show_barrier_state(__LINE__, result, barrier);
//
//            assert(result == BARRIER_SUCCESS);
//            vid = config.next_same_id_;
//
//            for (; bar_i_.start_index_ < bar_i_.copy_index_ &&
//                   static_cast<int>(bar_i_.start_index_) <=
//                       barrier_to_virtual_id_[barrier_configs_[bar_i_.start_index_].real_id_];
//                 ++bar_i_.start_index_)
//                ;
//        }
//
//        ++timestamp_;
//        cache::flush(&timestamp_, sizeof(timestamp_));
//
//    } else {
//        if (request_.request_.barrier_lift_times_)
//            request_.request_.barrier_lift_times_[vid] = static_cast<unsigned int>(ticker_.ticks());
//
//        BARECTF_ISR(leonnn_isr_trace_barrier_prod_done, vid);
//
//        assert(vid_produced_by_barrier_[barrier] < vid &&
//               "VID to barrier mapping is supposed to follow a strictly increasing order");
//        vid_produced_by_barrier_[barrier] = vid;
//    }
//}
//
//void Player::barrierCallback(unsigned char barrier, BarrierType type, void *context) {
//    if (Player *player = reinterpret_cast<Player *>(context))
//        player->barrierCallback(barrier, type);
//}
//} // namespace inference_runtime
//} // namespace nn
