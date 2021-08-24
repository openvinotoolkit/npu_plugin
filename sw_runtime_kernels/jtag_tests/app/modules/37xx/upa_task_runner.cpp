// {% copyright %}

#include <sw_nn_runtime_types.h>

#include "upa_task_runner.hpp"

//#include "layers/svuSLKernels_EP.h"
#include <layer_loader.h>

//#include <upa_layer_runner.h>
//#include <nn_runtime_types.h>
#include "act_shave_dispatcher.h"

//#include "mvTensorUtil.h"
#include <nn_cache.h>
#include "commonBuilder.hpp"
//#include <Fp16Convert.h>

#include <nn_time.h>

//using namespace nn::shave_lib;
//using namespace nn::memory;


//volatile u32 __attribute__((section(".nncmx.data0"))) shaveErrors;

static SoftLayerExec __attribute__((section(".nncmx0.shared.data"))) sl;
static Layer __attribute__((section(".nncmx0.shared.data"))) layer;
using namespace nn;

bool UPATaskRunner::enqueTask(std::unique_ptr<MVCNN::UPALayerTaskT> && task,
                              const std::vector<Buffer> &inputs,
                              const std::vector<Buffer> &outputs,
                              int /*numSHAVEs*/,
                              PerformanceData *perfData) {
    for (auto && input : inputs) {
        task->inputs.push_back(std::move(CommonFBFuilder::buildTensorReferenceT(input)));
    }
    for (auto && output : outputs) {
        task->outputs.push_back(std::move(CommonFBFuilder::buildTensorReferenceT(output)));
    }

    flatbuffers::FlatBufferBuilder _fbb;
    auto upa_task = MVCNN::UPALayerTask::Pack(_fbb, task.release());
    _fbb.Finish(upa_task);
//
    static std::shared_ptr<nn::act_shave_lib::ACTShaveDispatcher> actDisp;

    memset(&sl, 0, sizeof(sl));
    memset(&layer, 0, sizeof(layer));

    sl.counters_ = perfData->perfCounters;

    auto serializedUPATask = flatbuffers::GetRoot<MVCNN::UPALayerTask>(_fbb.GetBufferPointer());

    nn::shave_lib::LayerLoader::parseUPALayer(serializedUPATask, &layer);

//    LayerLoader::parseUPALayer(serializedUPATask, &layer);
//    layer.setExecCleanup();

    auto totalByteSize = [](const Buffer & b) {
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

    sl.layer_ = &layer;
//    layer.maxShaves = numSHAVEs;

    nnLog(MVLOG_DEBUG, "Enqueuing SL @ %p\n", &sl);
    nnLog(MVLOG_DEBUG, "           L @ %p\n", (void *)sl.layer_);
    nnLog(MVLOG_DEBUG, "         ABA @ %p\n", &sl.abs_addr_);
//    nnLog(MVLOG_DEBUG, "    preamble @ %x\n", sl.layer_->pre);
//    nnLog(MVLOG_DEBUG, "      kernel @ %x\n", sl.layer_->kernelEntry);

#if DEBUG_KERNELS
    leonPipePrintFlushBuffer();
#endif

//     Initialize ACT dispatcher
    actDisp = nn::act_shave_lib::ACTShaveDispatcher::getInstance();
    actDisp->initSWShaveDispatcher();
#ifdef NN_MAX_UPA_SHAVE_POOL_SIZE
    actDisp->resizeShavePool(NN_MAX_UPA_SHAVE_POOL_SIZE);
#endif
    actDisp->flushShaveL2DataCache();

    nn::time::Timer timer;

//     Enqueue layer
    if (!actDisp->enqueueLayerExec(&sl)) {
        return false;
    }

    timer.start();

//     Await layer completion - TODO possible async
    if (nullptr == actDisp->dequeueCompletedLayerExec()) {
        return false;
    }

    perfData->elapsedTimeNs = timer.elapsedNs();

//    if (layer.lyrClean)
//        (layer.lyrClean)(&layer.params);

    _enqued = true;

    return true;
}





bool UPATaskRunner::enqueTask(Op * operation,
                              const std::vector<Buffer> &inputs,
                              const std::vector<Buffer> &outputs,
                              int /*numSHAVEs*/,
                              PerformanceData *perfData) {

    static std::shared_ptr<nn::act_shave_lib::ACTShaveDispatcher> actDisp;

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

//    operation->parse(&layer);

//    LayerLoader::parseUPALayer(serializedUPATask, &layer);
//    layer.setExecCleanup();

    auto totalByteSize = [](const Buffer & b) {
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
//    layer.maxShaves = numSHAVEs;

    nnLog(MVLOG_DEBUG, "Enqueuing SL @ %p\n", &sl);
    nnLog(MVLOG_DEBUG, "           L @ %p\n", (void *)sl.layer_);
    nnLog(MVLOG_DEBUG, "         ABA @ %p\n", &sl.abs_addr_);
//    nnLog(MVLOG_DEBUG, "    preamble @ %x\n", sl.layer_->pre);
//    nnLog(MVLOG_DEBUG, "      kernel @ %x\n", sl.layer_->kernelEntry);

#if DEBUG_KERNELS
    leonPipePrintFlushBuffer();
#endif

//     Initialize ACT dispatcher
    actDisp = nn::act_shave_lib::ACTShaveDispatcher::getInstance();
    actDisp->initSWShaveDispatcher();
#ifdef NN_MAX_UPA_SHAVE_POOL_SIZE
    actDisp->resizeShavePool(NN_MAX_UPA_SHAVE_POOL_SIZE);
#endif
    actDisp->flushShaveL2DataCache();

    nn::time::Timer timer;

//     Enqueue layer
    if (!actDisp->enqueueLayerExec(&sl)) {
        return false;
    }

    timer.start();

//     Await layer completion - TODO possible async
    if (nullptr == actDisp->dequeueCompletedLayerExec()) {
        return false;
    }

    perfData->elapsedTimeNs = timer.elapsedNs();

//    if (layer.lyrClean)
//        (layer.lyrClean)(&layer.params);

    _enqued = true;

    return true;
}




bool UPATaskRunner::dequeResult() {
    return _enqued;
}
