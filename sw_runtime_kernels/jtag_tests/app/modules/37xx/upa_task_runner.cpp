// {% copyright %}

#include <sw_nn_runtime_types.h>
#include "upa_task_runner.hpp"
//#include "act_shave_dispatcher.h"
#include <nn_shave_manager.h>
//#include <nn_cmx_memory_map.h>
#include <nn_cache.h>
#include <nn_time.h>

//volatile u32 __attribute__((section(".nncmx.data0"))) shaveErrors;




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





static SoftLayerExec __attribute__((section(".nncmx0.shared.data"))) sl;
static Layer __attribute__((section(".nncmx0.shared.data"))) layer;
using namespace nn;

bool UPATaskRunner::enqueTask(Op * operation,
                              const std::vector<OpTensor> &inputs,
                              const std::vector<OpTensor> &outputs,
                              int /*numSHAVEs*/,
                              PerformanceData *perfData) {

//    static std::shared_ptr<nn::act_shave_lib::ACTShaveDispatcher> actDisp;
    nn::common_runtime::NNCmxMemoryMap *nnCmx = util::MemoryMap::project<NNCmxMemoryMap>(NN_CMX_BASE);
    alignas(NN_CACHE_LINE_LENGTH) nn::common_runtime::StaticMapping globalAreas(nnCmx);
    nn::inference_runtime::shaves::ShaveManager shaveManager(globalAreas);

    memset(&sl, 0, sizeof(sl));
    memset(&layer, 0, sizeof(layer));

    memory::cache_aligned_vector<TensorRef> &layerImputs = layer.getInputs();
    memory::cache_aligned_vector<TensorRef> &layerOutputs = layer.getOutputs();
    layerImputs.resize(inputs.size());
    layerOutputs.resize(outputs.size());
    for (unsigned i = 0; i < inputs.size(); i++) {
        layerImputs[i] = inputs[i];
    }
    for (unsigned i = 0; i < outputs.size(); i++) {
        layerOutputs[i] = outputs[i];
    }

    sl.counters_ = perfData->perfCounters;

    auto totalByteSize = [](const OpTensor & b) {
        return b.getFullDataSize();
    };

    auto &addrs = sl.abs_addr_;

    int addrsIdx = 0;
    for (auto && input : inputs) {
        addrs.inputs_[addrsIdx] = reinterpret_cast<const unsigned char *>(input.addr);
        nn::cache::flush(input.addr, totalByteSize(input));
        addrsIdx ++;
    }
    addrsIdx = 0;
    for (auto && output : outputs) {
        addrs.outputs_[addrsIdx] = reinterpret_cast<unsigned char *>(output.addr);
        nn::cache::invalidate(output.addr, totalByteSize(output));
        addrsIdx ++;
    }

    operation->parse(&layer);

    sl.layer_ = &layer;

    nnLog(MVLOG_DEBUG, "Enqueuing SL @ %p\n", &sl);
    nnLog(MVLOG_DEBUG, "           L @ %p\n", (void *)sl.layer_);
    nnLog(MVLOG_DEBUG, "         ABA @ %p\n", &sl.abs_addr_);

#if DEBUG_KERNELS
    leonPipePrintFlushBuffer();
#endif

////     Initialize ACT dispatcher
//    actDisp = nn::act_shave_lib::ACTShaveDispatcher::getInstance();
//    actDisp->initSWShaveDispatcher();
//#ifdef NN_MAX_UPA_SHAVE_POOL_SIZE
//    actDisp->resizeShavePool(NN_MAX_UPA_SHAVE_POOL_SIZE);
//#endif
//    actDisp->flushShaveL2DataCache();
//
//    nn::time::Timer timer;
//
////     Enqueue layer
//    if (!actDisp->enqueueLayerExec(&sl)) {
//        return false;
//    }
//
//    timer.start();
//
////     Await layer completion - TODO possible async
//    if (nullptr == actDisp->dequeueCompletedLayerExec()) {
//        return false;
//    }
//
//    perfData->elapsedTimeNs = timer.elapsedNs();
//
//    _enqued = true;

    return true;
}

bool UPATaskRunner::dequeResult() {
    return _enqued;
}
