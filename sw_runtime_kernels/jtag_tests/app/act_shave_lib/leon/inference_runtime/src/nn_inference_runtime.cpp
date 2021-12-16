/*
 * {% copyright %}
 */
#include "nn_inference_runtime.h"
#include <nn_inference_player.h>
#include <nn_context_manager.h>
#include <nn_shave_manager.h>
#include <nn_ipc_config.h>
#include <nn_time.h>
#include <nn_math.h>
#include <nn_counter.h>
#include <nn_fifo.h>
#include <nn_fifo_configs.h>
#include <DrvRegUtilsDefines.h>
#include <nn_log.h>
#include <type_traits>

#ifdef NN_ENABLE_STACK_CHECKER
#include <rtems/stackchk.h>
#endif

#ifdef NN_PROFILING
#include <barectf-myriad.h>
#else
#define BARECTF(...)
#endif

#ifdef NN_PROFILING_ALL
#define BARECTF_ALL BARECTF
#else
#define BARECTF_ALL(...)
#endif

using namespace std;

namespace {
using namespace nn::inference_runtime;
using namespace nn::common_runtime;

const unsigned int IR_EVENT = RTEMS_EVENT_17;
const unsigned int WORK_QUEUE_LENGTH = IR_WORKER_COUNT * 2;

#if !defined(CONFIG_TARGET_SOC_3600) && !defined(CONFIG_TARGET_SOC_3710) && !defined(CONFIG_TARGET_SOC_3720)
const uint32_t NN_CMX_BASE = 0x3e000000;
#else
const uint32_t NN_CMX_BASE = 0x2e000000;
#endif
#if defined(NN_ENABLE_SCALABILITY_REPORTING)
const uint32_t NN_LOG_BUFFER_SIZE = 0x800;
#endif /* NN_ENABLE_SCALABILITY_REPORTING */
} // namespace

namespace nn {
namespace inference_runtime {
using namespace ipc::messages;

InferenceRuntime::ShutdownNotifier::ShutdownNotifier(ipc::Channel &channel)
    : channel_(channel) {}

InferenceRuntime::ShutdownNotifier::~ShutdownNotifier() {
    channel_.send(ShutdownNotification());
}

InferenceRuntime::InferenceRuntime()
    : controlChannel_((ipc::init(), ipc::ChannelId::Control), &ipcControlCallback, this)
    , shutdownNotifier_(controlChannel_)
    , nnCmx_(util::MemoryMap::project<NNCmxMemoryMap>(NN_CMX_BASE))
    , globalAreas_(nnCmx_)
    , runtimeMappings_()
    , startupNotification_()
    , workloadChannel_(ipc::ChannelId::Workload, &ipcWorkloadCallback, this)
    , resource_manager_(createMask(MAX_CLUSTERS))
    , shave_manager_(globalAreas_)
    , params_()
    , workers_()
    , workersReady_(0)
    , active_(true)
    , workQueue_(WORK_QUEUE_LENGTH)
    , workQueueController_(WORK_QUEUE_LENGTH)
#ifdef NN_ENABLE_SCALABILITY_REPORTING
    , nnLogBuffer_(nn::memory::cache_aligned_alloc(NN_LOG_BUFFER_SIZE), NN_LOG_BUFFER_SIZE)
    , logger_(reinterpret_cast<uint32_t *>(nnLogBuffer_.addr32()), NN_LOG_BUFFER_SIZE,
              util::SharedBuffer::Process::PRODUCER)
#endif /* NN_ENABLE_SCALABILITY_REPORTING */
    , runContext_(IR_EVENT)
    , violatingMask(0) {
    nnLog(MVLOG_INFO, "IR ctor, priority %u", static_cast<unsigned int>(util::Thread::priority()));
    memory::print_heap_stats();

    util::fifoConfig(fifo::FIFO0_PARTS, fifo::FIFO1_PARTS, fifo::FIFO3_PARTS);
    {
        BarrierSetupData config = {
            .callback = &Player::barrierCallback,
            .irq_priority = min<unsigned char>(
                NUM_INT_LEVELS, DMA_DEFAULT_LEON_IRQ_PRIO + 1) // Barrier interrupt to be higher priority than DMA
        };

        auto result = BarrierInit(&config);
        assert(result == BARRIER_SUCCESS && "Could not initialize barrier driver");

        UNUSED(result);
    }

    // LNN side of DmaOpen
    // Since this is a constructor there's no way to report if it fails
    {
        auto result = DmaOpen(HGL_DMA_TYPE_NCE, HGL_DMA_ENGINE_0);
        if (result != DMA_SUCCESS)
            nnLog(MVLOG_ERROR, "Couldn't open DMA engine 0: %d", result);

        result = DmaOpen(HGL_DMA_TYPE_NCE, HGL_DMA_ENGINE_1);
        if (result != DMA_SUCCESS)
            nnLog(MVLOG_ERROR, "Couldn't open DMA engine 1: %d", result);
    }

    context::init_context_default_state(globalAreas_);

#ifdef NN_PROFILING
    init_barectf();
    BARECTF_ALL(leonnn_trace_lnn_overhead_start, 0);
    BARECTF_ALL(leonnn_trace_lnn_overhead_end, 0);
#endif

    for (unsigned int i = 0; i < runtimeMappings_.size(); ++i) {
        ClusterMapper::Config config(i);
        if (config.valid())
            runtimeMappings_[i] = RuntimeMapping(globalAreas_, config);
    }
    cache::flush(&runtimeMappings_, math::round_up<NN_CACHE_LINE_LENGTH>(sizeof(runtimeMappings_)));

    for (unsigned int i = 0; i < workers_.size(); ++i) {
        auto &param = params_[i];
        param.ir_ = this;
        param.tid_ = i;

        auto &worker = workers_[i];
        worker.set_priority(util::Thread::priority() - 1);
        worker.create(rtems_build_name('I', 'R', 'W', '0' + i));
        worker.start(&workerThread, &param);

        workersReady_.lock();
    }

    // we recycle the workersReady_ semaphore for context cleanup sync, so 1+ here
    workersReady_.unlock();

    cache::flush(&globalAreas_, math::round_up<NN_CACHE_LINE_LENGTH>(sizeof(globalAreas_)));
#ifdef NN_ENABLE_SCALABILITY_REPORTING
    cache::flush(&nnLogBuffer_, math::round_up<NN_CACHE_LINE_LENGTH>(sizeof(nnLogBuffer_)));
    startupNotification_.nnLogBuffer_ = const_cast<Buffer *>(&nnLogBuffer_);
#endif /* NN_ENABLE_SCALABILITY_REPORTING */

#ifdef NN_ENABLE_STACK_CHECKER
    rtems_stack_checker_report_usage();
#endif

    startupNotification_.staticMapping_ = &globalAreas_;
    cache::flush(&startupNotification_, math::round_up<NN_CACHE_LINE_LENGTH>(sizeof(startupNotification_)));
    controlChannel_.send(startupNotification_);
}

InferenceRuntime::~InferenceRuntime() {
    nnLog(MVLOG_INFO, "IR dtor");

#ifdef NN_ENABLE_STACK_CHECKER
    rtems_stack_checker_report_usage();
#endif

    active_ = false;
    workQueue_.broadcast(nullptr);

    for (unsigned int i = 0; i < workers_.size(); ++i) {
        if (workers_[i].joinable())
            workers_[i].join();
    }

    // LNN side of DmaClose
    {
        auto result = DmaClose(HGL_DMA_TYPE_NCE, HGL_DMA_ENGINE_0);
        if (result != DMA_SUCCESS)
            nnLog(MVLOG_ERROR, "Couldn't close DMA engine 0: %d", result);

        result = DmaClose(HGL_DMA_TYPE_NCE, HGL_DMA_ENGINE_1);
        if (result != DMA_SUCCESS)
            nnLog(MVLOG_ERROR, "Couldn't close DMA engine 1: %d", result);
    }

#ifdef NN_PROFILING
    fini_barectf();
#endif
}

void InferenceRuntime::run() {
    runContext_.remap();

    while (active())
        runContext_.wait();
}

void InferenceRuntime::feed(WorkRequest &wr) {
    dispatch(wr);
}

void InferenceRuntime::handleControl(void *message, unsigned int size) {
    nnLog(MVLOG_DEBUG, "IR::handle");

    switch (size) {
        case sizeof(ShutdownRequest): {
            active_ = false;
            runContext_.notify();
            break;
        }

        case sizeof(ResourceLimitRequest): {
            if (auto request = static_cast<const ResourceLimitRequest *>(message)) {
                cache::invalidate(request, math::round_up<NN_CACHE_LINE_LENGTH>(sizeof(ResourceLimitRequest)));

                // TODO: Power management stuff shouldn't happen here
#ifndef NN_ACTIVE_TILE_MANAGEMENT
                ResourceLock lock(resource_manager_);
                shave_manager_.startNNShavesForTiles();
#endif /* NN_ACTIVE_TILE_MANAGEMENT */

                controlChannel_.send(*request);
            }

            break;
        }

        default:
            break;
    }
}

void handlePreemtpion(void *message, unsigned int size) {
    // TODO: impl
    UNUSED(message);
    UNUSED(size);
}

void InferenceRuntime::dispatch(WorkRequest &wr) {
    workQueueController_.lock();
    bool success = workQueue_.push(&wr);
    nnLog(MVLOG_DEBUG, "IR::dispatch pushed a work request, success: %d", success);
}

void InferenceRuntime::ipcControlCallback(void *message, uint32_t size, void *context) {
    nnLog(MVLOG_INFO, "IR::ipcControlCallback w/ message %p, size %lu", message, size);

    if (message != nullptr)
        if (auto *ir = reinterpret_cast<InferenceRuntime *>(context))
            ir->handleControl(message, size);
}

void InferenceRuntime::ipcWorkloadCallback(void *message, uint32_t size, void *context) {
    nnLog(MVLOG_DEBUG, "IR::ipcWorkloadCallback w/ message %p, size %lu", message, size);

    if (size == sizeof(WorkRequest)) {
        if (auto *ir = reinterpret_cast<InferenceRuntime *>(context)) {
            if (WorkRequest *wr = reinterpret_cast<WorkRequest *>(message)) {
                cache::invalidate(*wr);
                ir->dispatch(*wr);
            }
        }
    }
}

bool InferenceRuntime::flushResources(ResourceLock &lock, const WorkRequest *request) {
    bool context_success = false;

    if (context::configured_for_single_context()) {
        shave_manager_.stopActShavesForTiles();
        context::flush_tiles_of_context(globalAreas_);

        if (request->resources_.nn_slice_count_ > 1) {
            context_success = context::prepare_tiles_for_context(request->context_info_.ssid_, globalAreas_);
        } else {
            context::configure_nce_shave_l2_for_user_context_per_tile();

            ClusterMapper::Config config(lock.resources().clusterMask());
            auto tile = config.index();
            context_success = context::prepare_tile_for_context(tile, request->context_info_.ssid_, globalAreas_);
        }
    } else {
        if (request->resources_.nn_slice_count_ == 1) {
            ClusterMapper::Config config(lock.resources().clusterMask());
            auto tile = config.index();
            shave_manager_.stopActShavesForTile(tile);
            context_success = context::flush_tile_of_context(globalAreas_, tile);
            context_success &= context::prepare_tile_for_context(tile, request->context_info_.ssid_, globalAreas_);
        } else {
            // TODO: check that the assumption that all (2) tiles are halted here is correct
            shave_manager_.stopActShavesForTiles();
            context::flush_tiles_of_context(globalAreas_);
            context::configure_nce_shave_l2_for_single_user_context();
            context_success = context::prepare_tiles_for_context(request->context_info_.ssid_, globalAreas_);
        }
    }

#ifdef NN_ENABLE_CONTEXT_DEBUGGING
    nnLog(MVLOG_INFO, "CONTEXT_SUCCESS: %d SLICES: %d\n", context_success, request->resources_.nn_slice_count_);

    context::debug_print_L2_config();
    context::debug_print_tile_config();
#endif
    return context_success;
}

inline void InferenceRuntime::enqueueContextViolation(ResourceMask rm) {
    violatingMask |= rm;
}

bool InferenceRuntime::handleContextViolation() {
    bool ret{false};

    if (violatingMask) {
        // for now, halt everything (i.e. assume violationMask has all tiles)
        ResourceLock lock(resource_manager_);

        /*
         * This is where we could reset the DMA engine to forgo a KMD NCE reset
         * if we could cleanly flush the DMA engines or reset them from LNN
         */

        context::flush_tiles_of_context(globalAreas_);

        // here we enter IR failed state, no future inferences are possible without NCE rest

        // violatingMask = 0;
    }

    return ret;
}

inline void InferenceRuntime::waitForContextViolationClear() {
    while (violatingMask)
        util::Thread::yield();
}

bool InferenceRuntime::remapResources(ResourceLock &lock, const WorkRequest *request) {
    TileContextState currContext(request->context_info_.ssid_, context::get_host_ID_mapping(request->context_info_.ssid_));
    auto mask = lock.resources().clusterMask();
    bool didRemap{false};
    uint32_t res{0};

    // For the tiles currently reserved by lock (bitmask), check if the
    // 5 bit (VPU) or 20 bit (host/MMU) context SSIDs have changed. If so,
    // we have remapped resources and need to do a context flush.
    do {
        if (mask & 1) {
            didRemap |= (prevContext_[res] != currContext);
            prevContext_[res] = currContext;
        }

        res++;
        mask = mask >> 1;
    } while (mask && res < MAX_TILES);

    return didRemap;
}

bool InferenceRuntime::lockResources(ResourceLock &lock, const WorkRequest *request, bool *didRemap) {
    bool remapped = false;
    bool success = false;

    waitForContextViolationClear();

    if (request->flush_required_) {
#ifdef NN_ACTIVE_TILE_MANAGEMENT
        success = lock.lockByMask(request->tile_mask_);
#else
        success = lock.lock(request->resources_.nn_slice_count_);
#endif /* NN_ACTIVE_TILE_MANAGEMENT */
        remapped = true; // force remapped consequence

        if (success) {
            remapResources(lock, request);
            success = flushResources(lock, request);
        }
    } else {
#ifdef NN_ACTIVE_TILE_MANAGEMENT
        success = lock.lockByMask(request->tile_mask_);
#else
        success = lock.lockWithAffinity(request->resources_.nn_slice_count_, request->context_info_.ssid_);

        if (!success)
        {
            // If no tiles with the same context were available, see if any tiles are available and
            // switch context on them
            success = lock.lock(request->resources_.nn_slice_count_);
        }
#endif /* NN_ACTIVE_TILE_MANAGEMENT */

        if (success && (remapped = remapResources(lock, request))) {
            // corner case if 5bit <-> 20bit ID mapping changed
            success = flushResources(lock, request);
        }
    }

    *didRemap = remapped;
    return success;
}

void InferenceRuntime::updateActShaves(const ResourceMask rm, const ActKernelRuntimeConfigs &cfgs,
                                       const bool forceRestart) {
    switch (rm) {
        case 0b01:
            shave_manager_.startActShavesForTile(0, cfgs, forceRestart);
            break;
        case 0b10:
            shave_manager_.startActShavesForTile(1, cfgs, forceRestart);
            break;
        case 0b11:
            break;
            shave_manager_.startActShavesForTiles(cfgs, forceRestart);
        default:
            break;
    }
}

void InferenceRuntime::workerThread(ThreadParam *param) {
    InferenceRuntime &ir = *param->ir_;
    time::Ticker ticker;

    ResourceLock lock(ir.resource_manager_);
    memory::cache_aligned_unique_ptr<Player> player(new (memory::cache_aligned)
                                                        Player(ticker, static_cast<unsigned char>(param->tid_)));
    nnLog(MVLOG_INFO, "IR worker thread %u started, priority %u", param->tid_,
          static_cast<unsigned int>(util::Thread::priority()));

    ir.workersReady_.unlock();

    while (ir.active()) {
        WorkRequest *request = nullptr;
        bool success = ir.workQueue_.pop(request);
        if (success)
            ir.workQueueController_.unlock();

        nnLog(MVLOG_DEBUG, "IR worker thread %u pulled a work request at %p, success %d", param->tid_, request,
              success);

        if (success && request != nullptr) {
            if (request->phase_ == WorkRequest::RUN) {
                BARECTF(leonnn_trace_inference_requested, param->tid_, (uint32_t)request);

                ticker.start();

                assert(request->inference_request_);
                cache::invalidate(*request->inference_request_);

                if (request->resources_.nn_slice_length_ <= ir.globalAreas_.workareas_[0].size()) {
                    bool didRemap;

                    nnLog(MVLOG_INFO, "IR Thread %u: Assigning context resources for inference @ %p with tile mask 0x%x, context ID 0x%x", param->tid_, request, request->tile_mask_, request->context_info_.ssid_);
                    if (ir.lockResources(lock, request, &didRemap)) {
                        // TODO: Use PowerManager here. Power on reserved tiles

                        request->inference_request_->response_.code_ = InferenceRequest::RUNNING;
                        request->inference_request_->response_.timestamp_ = player->timestamp();
                        cache::flush(request->inference_request_->response_);

                        const auto &resources = lock.resources();
                        ClusterMapper::Config config = resources.clusterMask();
                        const auto &rtm = ir.runtimeMappings_[config];
                        const auto &mapped = request->mapped_[config.index()];

                        nnLog(MVLOG_WARN, "IR Thread %u: Executing inference @ %p (mapped @ %p) with tile mask: 0x%x", param->tid_, request, mapped, resources.clusterMask());

                        if (mapped.actKInvocations_.size() != 0) {
                            ir.updateActShaves((ResourceMask)config, mapped.actRtConfigs_, didRemap);
                        }

#ifdef NN_ACTIVE_TILE_MANAGEMENT
                        ir.shave_manager_.startNNShavesForTileMask(resources.clusterMask());
#endif /* NN_ACTIVE_TILE_MANAGEMENT */

#ifdef NN_ENABLE_SCALABILITY_REPORTING
                        ir.logger_.push(InferenceRequestLoggerUpdate(request->created_ticks_, InferenceRequest::RUNNING,
                                                                     util::sampleFRC()));
#endif /* NN_ENABLE_SCALABILITY_REPORTING */
                        auto code = player->play(request->resources_, mapped, rtm, *request->inference_request_,
                                                 request->context_info_);

#ifdef NN_ACTIVE_TILE_MANAGEMENT
                        ir.shave_manager_.stopNNShavesForTileMask(resources.clusterMask());
#endif /* NN_ACTIVE_TILE_MANAGEMENT */

                        if (code == InferenceRequest::CONTEXT_VIOLATION) {
                            ir.workersReady_.lock();
                            bool firstViolator{ir.violatingMask == 0};
                            ir.enqueueContextViolation(config);
                            ir.workersReady_.unlock();

                            lock.release();

                            if (firstViolator) {
                                while (!ir.resource_manager_.allResourcesFree())
                                    util::Thread::yield();

                                ir.handleContextViolation();
                                code = InferenceRequest::CONTEXT_VIOLATION_IR_HALTED;
                            }

                            // We could recover from this violation if handleContextViolation() could reset the DMA
                            // engines
                        } else {
                            lock.release();
                        }
                        request->inference_request_->response_.code_ = code;
                        request->inference_request_->response_.timestamp_ = nullptr;
#ifdef NN_ENABLE_SCALABILITY_REPORTING
                        ir.logger_.push(InferenceRequestLoggerUpdate(request->created_ticks_, code, util::sampleFRC()));
#endif /* NN_ENABLE_SCALABILITY_REPORTING */
                    } else {
                        request->inference_request_->response_.code_ = InferenceRequest::OUT_OF_RESOURCES;
#ifdef NN_ENABLE_SCALABILITY_REPORTING
                        ir.logger_.push(InferenceRequestLoggerUpdate(
                            request->created_ticks_, InferenceRequest::OUT_OF_RESOURCES, util::sampleFRC()));
#endif /* NN_ENABLE_SCALABILITY_REPORTING */
                    }
                } else {
                    request->inference_request_->response_.code_ = InferenceRequest::OUT_OF_MEMORY;
#ifdef NN_ENABLE_SCALABILITY_REPORTING
                    ir.logger_.push(InferenceRequestLoggerUpdate(request->created_ticks_,
                                                                 InferenceRequest::OUT_OF_MEMORY, util::sampleFRC()));
#endif /* NN_ENABLE_SCALABILITY_REPORTING */
                }
                auto ticks = ticker.ticks();
                request->inference_request_->response_.lnn_ticks_ = static_cast<unsigned int>(ticks);
                cache::flush(request->inference_request_->response_);

                ir.workloadChannel_.send(*request);

                BARECTF(leonnn_trace_inference_answered, param->tid_);
                nnLog(MVLOG_PERF, "\tIR Thread %u took %llu clock cycles for inference @ %p", param->tid_, ticks, request);

                player->reset();
            } else if (request->phase_ == WorkRequest::PREPARE) {
                cache::invalidate(&request->resources_,
                                  math::round_up<NN_CACHE_LINE_LENGTH>(sizeof(request->resources_)));

                const unsigned int configs = ClusterMapper().config_count(request->resources_.nn_slice_count_);
                cache::invalidate(request->mapped_, configs * sizeof(MappedInference));

                for (unsigned int i = 0; i < configs; ++i) {
                    const auto &mapped = request->mapped_[i];

                    cache::invalidate(mapped.barrierConfigs_);
                }

                ir.workloadChannel_.send(*request);

                nnLog(MVLOG_INFO, "IR worker thread %u synchronized cache for inference @ %p", param->tid_, request);
            } else if (request->phase_ == WorkRequest::PREFETCH) {
                nnLog(MVLOG_ERROR, "PREFETCH not supported, no CSRAM available");
                request->inference_request_->response_.code_ = InferenceRequest::INTERNAL_ERROR;
            } else {
                request->inference_request_->response_.code_ = InferenceRequest::INTERNAL_ERROR;
            }
        }
    }

    nnLog(MVLOG_INFO, "IR worker thread %u finished", param->tid_);
}

Service &Service::instance() {
    static Service singleton;
    return singleton;
}

Service::Service()
    : ir_(new (memory::cache_aligned) InferenceRuntime) {}

} // namespace inference_runtime
} // namespace nn
