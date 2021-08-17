#include "act_shave_dispatcher.h"
#include "act_shave_runtime.h"
#include <DrvRegUtils.h>
#include <registersVpu.h>
#include <VcsHooksApi.h>
#include <swcLeonUtils.h>
#include <nn_cache.h>

using namespace nn::act_shave_lib;

std::shared_ptr<ACTShaveDispatcher> ACTShaveDispatcher::getInstance() {
    static std::shared_ptr<ACTShaveDispatcher> holder(new (memory::cache_aligned) ACTShaveDispatcher, memory::cache_aligned_deleter<ACTShaveDispatcher>());
    return holder;
}
ACTShaveDispatcher::ACTShaveDispatcher() {}
ACTShaveDispatcher::~ACTShaveDispatcher() {
    terminateSWShaveDispatcher();
}

void ACTShaveDispatcher::initSWShaveDispatcher() {
    std::lock_guard<std::mutex> startStopShaveAccess(runnerMutex);
    if (shavesStarted) {
        return;
    }
//    printf("INIT dispatcher -  starting activation shaves\n");

    // check whether shaves started already
    startActShave();
//    printf("started act shaves\n");
    shavesStarted = true;
}

void ACTShaveDispatcher::terminateSWShaveDispatcher() {
    std::lock_guard<std::mutex> startStopShaveAccess(runnerMutex);
//    printf("TERMINATE dispatcher -  stopping activation shaves\n");
    stopActShave();
    shavesStarted = false;
}

bool ACTShaveDispatcher::resizeShavePool(unsigned int /*total_shaves*/) {
    return true;
}
bool ACTShaveDispatcher::hasResources() const {
    return true;
}
unsigned char ACTShaveDispatcher::getControllerShaveID() const {
    return 0;
}
void ACTShaveDispatcher::flushShaveL2DataCache(){}

void ACTShaveDispatcher::flushShaveL2InstructionCache(){}

bool ACTShaveDispatcher::enqueueLayerExec(nn::shave_lib::SoftLayerExec *slExec) {
    std::lock_guard<std::mutex> shaveQueueAccess(runnerMutex);
    shv_job_header hdr = {};

    hdr.shv_kernel_address = reinterpret_cast<uint64_t>(slExec->layer_->kernelEntry);
    hdr.kernel_arg_address = reinterpret_cast<uint64_t>(slExec->layer_->params.layerParams);
    hdr.shv_pre_address = reinterpret_cast<uint64_t>(slExec->layer_->pre);
    hdr.aba_pointer = reinterpret_cast<uint64_t>(&slExec->abs_addr_);

    jobs.push_back({slExec, hdr});
    swcLeonDataCacheFlush();
//    printf("enqueueLayerExec: %p\n ", slExec->layer_->kernelEntry);
    return enqueueShaveJob(&jobs.back().job_header);
}
//#include "param_postops.h"

nn::shave_lib::SoftLayerExec *ACTShaveDispatcher::dequeueCompletedLayerExec() {
    if (jobs.empty()) {
        return nullptr;
    }

    std::unique_lock<std::mutex> jobQueueAccess(runnerMutex);
    if (jobs.empty()) {
        return nullptr;
    }
   // auto params = (nn::shave_lib::PostOpsParams*)  0x853FD6C0;

    auto firstJob =  jobs.begin();
    jobs.pop_front();
    jobQueueAccess.unlock();

    while(firstJob->sleq != nullptr && ! *((uint32_t*)firstJob->job_header.job_completed_pointer)) {
        nn::cache::invalidate(*firstJob->sleq);
    }

//    printf("INFO: dequeueCompletedLayerExec() -- DONE\n");
    fflush(stdout);

    return firstJob->sleq;
}
