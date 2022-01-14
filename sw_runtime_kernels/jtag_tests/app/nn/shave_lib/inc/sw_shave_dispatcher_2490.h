/*
* {% copyright %}
*/
#pragma once

#include <nn_inference_runtime_types.h>
#include <nn_memory.h>
#include <nn_message_queue.h>
#include <nn_relocation.h>

#include "sw_layer.h"
#include "upa_layer_runner.h"

#include <nn_message_queue.h>
#include <nn_semaphore.h>
#include <nn_thread.h>

#include <memory>
#include <mutex>

#define SHAVE_LAYER_LOAD_ERROR 0

namespace nn {
namespace shave_lib {
class SWShaveDispatcher {
  public:
    static std::shared_ptr<SWShaveDispatcher> getInstance();

    SWShaveDispatcher(SWShaveDispatcher &&) = delete;
    SWShaveDispatcher(const SWShaveDispatcher &) = delete;
    SWShaveDispatcher &operator=(SWShaveDispatcher &&) = delete;
    SWShaveDispatcher &operator=(const SWShaveDispatcher &) = delete;

    SWShaveDispatcher(/**/);
    ~SWShaveDispatcher();

    /**
     * Registers the SoftChannel Handle with the Dispatcher and initializes
     * the shave_lib backend
     */
    void initSWShaveDispatcher();

    /**
     * Terminate the shave_lib backend
     */
    void terminateSWShaveDispatcher();

    /**
     * Resizes SHAVE pool
     * @param[in] - total_shaves - Number of SHAVEs requested by the inference.
     */
    bool resizeShavePool(unsigned int total_shaves);

    /**
     * Returns true if the minimum resources required to execute a SL are available
     */
    bool hasResources() const;

    /**
     * @returns the SVU shaveID that has taken the role of controller
     */
    unsigned char getControllerShaveID() const;

    /**
     * Flush and invalidate the L2 datacache of all the associated shaves
     */
    void flushShaveL2DataCache();

    /**
     * Invalidate the L2 instruction cache of all the associated shaves
     */
    void flushShaveL2InstructionCache();

    /**
     * The IRS should call this method to give the UPA Dispatcher a soft layer task
     */
    bool enqueueLayerExec(SoftLayerExec *slExec);

    /**
     * Blocks until a soft layer task is completed enqueued by enqueueLayerExec(...)
     * @returns returns the first completed SLE or nullptr if the queue is empty
     */
    SoftLayerExec *dequeueCompletedLayerExec();

  private:
    std::mutex runnerMutex;
    memory::cache_aligned_unique_ptr<UPALayerRunner> upaRunner;
    nn::util::MessageQueue<SoftLayerExec *> sleq;
};
} // namespace shave_lib
} // namespace nn
