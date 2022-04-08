/*
* {% copyright %}
*/
#pragma once

#include <Ipc.h>
#include <OsDrvMutex.h>
#include <OsDrvSvu.h>
#include <dma_leon.h>
#include <nn_message_queue.h>
#include <nn_thread.h>
#include <sw_layer.h>
#include <sw_nn_runtime_types.h>
#include <sw_shave_res_manager.h>

namespace nn {
namespace shave_lib {

class UPALayerRunner {
  public:
    UPALayerRunner();
    ~UPALayerRunner();

    bool hasResources() const;
    uint32_t getControllerShaveID() const;
    bool resizeShavePool(unsigned int total_shaves);
    void flushShaveL2DataCache(void);
    void flushShaveL2InstructionCache(void);

  private:
    static nn::util::MessageQueue<uint16_t> upaMsgQueue;

    nn::util::Thread upaThread;
    memory::cache_aligned_vector<ShaveResource> resources;
    memory::cache_aligned_vector<ResMgrHandler> resourcesHandles;
    sDrvSvuHandler_t upaShaveHandle;
    OsDrvMutexHandler mtxHandle;
    bool keepRunning;

    void allocateResources();
    rtems_status_code  allocateShaveResource();
    void allocateShaveResources();
    void setupAndStartResources(unsigned int shaveFrom);
    void freeResources();
    void freeShaveResources();

    void setupShave(const ShaveResource &res, const SoftKernel &kernel);
    void shaveCleanup(unsigned int shave_id);

    static void monitorMsgQueue(UPALayerRunner *upa);
    static void voidIrqHandlerExitHook(void *source);

    UPALayerRunner(UPALayerRunner &&) = delete;
    UPALayerRunner(const UPALayerRunner &) = delete;
    UPALayerRunner &operator =(UPALayerRunner &&) = delete;
    UPALayerRunner &operator =(const UPALayerRunner &) = delete;
};
} // namespace shave_lib
} // namespace nn
