/*
* {% copyright %}
*/
#include "sw_shave_dispatcher.h"

#include "upa_fifo.h"
#include <assert.h>
#include <mutex>
#include <mvLog.h>
#include <nn_cache.h>
#include <nn_resources.h>
#include <thread>

//#define NN_SHAVE_WORKER_USE_DMA

using nn::inference_runtime::frontend::SoftLayerTask;

namespace nn {
namespace shave_lib {

std::shared_ptr<SWShaveDispatcher> SWShaveDispatcher::getInstance() {
    static std::shared_ptr<SWShaveDispatcher> holder(new (memory::cache_aligned) SWShaveDispatcher, memory::cache_aligned_deleter<SWShaveDispatcher>());
    return holder;
}

SWShaveDispatcher::~SWShaveDispatcher() {}

void SWShaveDispatcher::initSWShaveDispatcher() {
    std::lock_guard<std::mutex> lg(runnerMutex);
    if (!upaRunner)
         upaRunner.reset(new (memory::cache_aligned) UPALayerRunner);
}

bool SWShaveDispatcher::resizeShavePool(unsigned int total_shaves){
    std::lock_guard<std::mutex> lg(runnerMutex);
    if (upaRunner){
        return upaRunner->resizeShavePool(total_shaves);
    }
    return false;
}

void SWShaveDispatcher::terminateSWShaveDispatcher() {
    std::lock_guard<std::mutex> lg(runnerMutex);

    if (upaRunner)
        upaRunner.reset();
}

bool SWShaveDispatcher::hasResources() const { return upaRunner ? upaRunner->hasResources() : false; }

unsigned char SWShaveDispatcher::getControllerShaveID() const {
    return (unsigned char)upaRunner->getControllerShaveID();
}

void SWShaveDispatcher::flushShaveL2DataCache() {
    if (upaRunner)
        upaRunner->flushShaveL2DataCache();
}

void SWShaveDispatcher::flushShaveL2InstructionCache() {
    if (upaRunner)
        upaRunner->flushShaveL2InstructionCache();
}

// Enqueues the SLE to the controller shave
// Note that this means a preamble may need to be run if NO_SVU_NN_CONTROLLER is enabled
bool SWShaveDispatcher::enqueueLayerExec(SoftLayerExec *slExec) {
    assert(upaRunner && "shave_lib subsystem must be initalized");

    sleq.push(slExec);
    return nn::util::upaFifoWrite(upaRunner->getControllerShaveID(), (uint32_t)slExec);
}

// Track enqueued SLEs and check each if sle->completed_ == true; return that SLE
// Note: this is blocking.
SoftLayerExec *SWShaveDispatcher::dequeueCompletedLayerExec() {
    assert(upaRunner && "shave_lib subsystem must be initalized");

    SoftLayerExec *slExec {nullptr};
    sleq.pop(slExec);

    if (slExec){
        while (!slExec->completed_) {
            // TODO: Flush Shave L2 cache here.

            nn::cache::invalidate(slExec, sizeof(SoftLayerExec));
        }
    }

    return slExec;
}

// private //

SWShaveDispatcher::SWShaveDispatcher() : runnerMutex{}, upaRunner(), sleq(CMX_FIFO_NB_ENTRIES) {}

} // namespace shave_lib
} // namespace nn
