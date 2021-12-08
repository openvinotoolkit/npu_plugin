/*
 * {% copyright %}
 */
#ifndef NN_INFERENCE_RUNTIME_H_
#define NN_INFERENCE_RUNTIME_H_

#include <array>
#include <vector>
#include <stdint.h>
#include <rtems.h>
#include <nn_hw_resources.h>
#include <nn_runtime_configs.h>
#include <nn_ipc.h>
#include <nn_ipc_messages.h>
#include <nn_cache.h>
#include <nn_message_queue.h>
#include <nn_task_context.h>
#include <nn_semaphore.h>
#include <nn_thread.h>
#include <nn_memory.h>
#include <nn_inference_runtime_types.h>
#include <nn_resource_manager.h>
#include <nn_shave_manager.h>
#include <nn_cmx_memory_map.h>
#include <nn_resource_locator.h>
#ifdef NN_ENABLE_SCALABILITY_REPORTING
#include <nn_cached_shared_buffer.h>
#endif /* NN_ENABLE_SCALABILITY_REPORTING */

namespace nn {
namespace inference_runtime {

using namespace common_runtime;
using namespace act_runtime;

class InferenceRuntime {
public:
    ~InferenceRuntime();

    void run();

    void feed(WorkRequest &wr);

private:
    struct ShutdownNotifier {
        explicit ShutdownNotifier(ipc::Channel &channel);
        ~ShutdownNotifier();

    private:
        ipc::Channel &channel_;
    };

    struct ThreadParam {
        InferenceRuntime *ir_;
        unsigned int tid_;
    };

    ipc::Channel controlChannel_;
    ShutdownNotifier shutdownNotifier_;
    NNCmxMemoryMap *nnCmx_;
    StaticMapping globalAreas_ NN_CACHE_ALIGNED;
    std::array<RuntimeMapping, MAX_CLUSTERS * MAX_CLUSTERS> runtimeMappings_ NN_CACHE_ALIGNED;
    ipc::messages::StartupNotification startupNotification_ NN_CACHE_ALIGNED;
    ipc::Channel workloadChannel_ NN_CACHE_ALIGNED;
    ResourceManager resource_manager_;
    shaves::ShaveManager shave_manager_;
    std::array<ThreadParam, IR_WORKER_COUNT> params_;
    std::array<nn::util::Thread, IR_WORKER_COUNT> workers_;
    util::Semaphore workersReady_;
    volatile bool active_;
    util::MessageQueue<WorkRequest *> workQueue_;
    util::Semaphore workQueueController_;
#ifdef NN_ENABLE_SCALABILITY_REPORTING
        const Buffer nnLogBuffer_;
        CachedSharedBuffer<common_runtime::InferenceRequestLoggerUpdate> logger_;
#endif
    util::TaskContext runContext_;
    common_runtime::TileContextState prevContext_[MAX_TILES];
    uint32_t violatingMask;

    InferenceRuntime();
    InferenceRuntime(const InferenceRuntime &) = delete;
    InferenceRuntime &operator=(const InferenceRuntime &) = delete;

    inline bool active() const { return active_; }

    void handleControl(void *message, unsigned int size);
    void handlePreemtpion(void *message, unsigned int size);
    void dispatch(WorkRequest &wr);

    static void ipcControlCallback(void *message, uint32_t size, void *context);
    static void ipcWorkloadCallback(void *message, uint32_t size, void *context);
    static void workerThread(ThreadParam *param);
    static void prefetch(unsigned int tid, const WorkRequest &wr);

    bool lockResources(ResourceLock &lock, const WorkRequest *wr, bool *didRemap);
    bool flushResources(ResourceLock &lock, const WorkRequest *wr);
    bool remapResources(ResourceLock &lock, const WorkRequest *wr);
    void updateActShaves(const ResourceMask rm, const ActKernelRuntimeConfigs &cfgs, const bool forceRestart);
    void enqueueContextViolation(ResourceMask rm);
    void waitForContextViolationClear();
    bool handleContextViolation();

    friend class Service;
};

class Service {
public:
    static Service &instance();
    InferenceRuntime &runtime() { return *ir_; }

private:
    memory::cache_aligned_unique_ptr<InferenceRuntime> ir_;

    Service();

    Service(const Service &) = delete;
    Service &operator=(const Service &) = delete;
};
} // namespace inference_runtime
} // namespace nn

#endif // NN_INFERENCE_RUNTIME_H_
