// {% copyright %}

#include "upa_task_runner.hpp"
#include "sw_shave_dispatcher.h"
#include "mvTensorUtil.h"
#include <nn_cache.h>
#include <Fp16Convert.h>

#include <nn_time.h>

using namespace nn;
using namespace nn::shave_lib;
using namespace nn::memory;

static SoftLayerExec __attribute__((section(".cmx_direct.data"))) sl;
static Layer __attribute__((section(".cmx_direct.data"))) layer;

bool UPATaskRunner::enqueTask(Op * operation,
                              const std::vector<OpTensor> &inputs,
                              const std::vector<OpTensor> &outputs,
                              int numSHAVEs,
                              PerformanceData *perfData) {

    static std::shared_ptr<nn::shave_lib::SWShaveDispatcher> upaDisp;

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
    layer.maxShaves = numSHAVEs;

    sl.layer_ = &layer;

    nnLog(MVLOG_DEBUG, "Enqueuing SL @ %p\n", &sl);
    nnLog(MVLOG_DEBUG, "           L @ %p\n", (void *)sl.layer_);
    nnLog(MVLOG_DEBUG, "         ABA @ %p\n", &sl.abs_addr_);

#if DEBUG_KERNELS
    leonPipePrintFlushBuffer();
#endif

    upaDisp = nn::shave_lib::SWShaveDispatcher::getInstance();
    upaDisp->initSWShaveDispatcher();
#ifdef NN_MAX_UPA_SHAVE_POOL_SIZE
    upaDisp->resizeShavePool(NN_MAX_UPA_SHAVE_POOL_SIZE);
#endif
    upaDisp->flushShaveL2DataCache();

    nn::time::Timer timer;

    // Enqueue layer
    if (!upaDisp->enqueueLayerExec(&sl)) {
        return false;
    }

    timer.start();

    // Await layer completion - TODO possible async
    if (nullptr == upaDisp->dequeueCompletedLayerExec()) {
        return false;
    }

    perfData->elapsedTimeNs = timer.elapsedNs();

    _enqued = true;

    return true;
}

bool UPATaskRunner::dequeResult() {
    return _enqued;
}
