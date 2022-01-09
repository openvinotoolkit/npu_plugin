// {% copyright %}

//#ifdef CONFIG_TARGET_SOC_3720
__attribute__((aligned(1024)))
#include "sk.nnActEntry.3010xx.text.xdat"
void * sk_nnActEntry_3010xx_text_ref = (void*)sk_nnActEntry_3010xx_text;
//#endif
//extern void*  (shvNN0_act_shave_runtime_shaveMain);
//extern void const *shvNN0_nnActEntry;


#include <sw_nn_runtime_types.h>
#include <sw_shave_lib_common.h>
#include <HglShaveCommon.h>
#include "upa_task_runner.hpp"
//#include "act_shave_dispatcher.h"
#include <nn_shave_manager.h>
#include <nn_fifo_manager.h>
//#include <nn_cmx_memory_map.h>
#include <nn_cache.h>
#include <nn_time.h>
#include <CustomCpp.h>

//volatile u32 __attribute__((section(".nncmx.data0"))) shaveErrors;


unsigned char __attribute__((section(".nncmx0.shared.data"), aligned(64))) actShaveData[SHAVE_LIB_DATA_SIZE];
unsigned int actShaveDataReserved = 0;


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


//extern void const *shvNN0_nnActEntry;


static SoftLayerExec __attribute__((section(".nncmx0.shared.data"))) sl;
static Layer __attribute__((section(".nncmx0.shared.data"))) layer;
using namespace nn;
extern bool HglShaveAccessAllowed[HGL_SHAVE_TYPE_NB];

nn::common_runtime::NNCmxMemoryMap *nnCmx = util::MemoryMap::project<NNCmxMemoryMap>(NN_CMX_BASE);

std::shared_ptr<nn::common_runtime::StaticMapping> getStaticMapping(nn::common_runtime::NNCmxMemoryMap *nnCmx) {
    static std::shared_ptr<nn::common_runtime::StaticMapping> holder(new (memory::cache_aligned) nn::common_runtime::StaticMapping(nnCmx), memory::cache_aligned_deleter<nn::common_runtime::StaticMapping>());
    return holder;
}

std::shared_ptr<nn::inference_runtime::shaves::ShaveManager> getShaveManager(std::shared_ptr<nn::common_runtime::StaticMapping> mapping) {
    static std::shared_ptr<nn::inference_runtime::shaves::ShaveManager> holder(new (memory::cache_aligned) nn::inference_runtime::shaves::ShaveManager(*mapping), memory::cache_aligned_deleter<nn::inference_runtime::shaves::ShaveManager>());
    return holder;
}














bool UPATaskRunner::enqueTask(Op * operation,
                              const std::vector<OpTensor> &inputs,
                              const std::vector<OpTensor> &outputs,
                              int /*numSHAVEs*/,
                              PerformanceData *perfData) {

    actShaveDataReserved = 0;

//    static std::shared_ptr<nn::act_shave_lib::ACTShaveDispatcher> actDisp;
    HglShaveAccessAllowed[1] = false;
    HglShaveAccessAllowed[2] = true;
    cache::flush(HglShaveAccessAllowed, sizeof(bool) * HGL_SHAVE_TYPE_NB);
//    printf("!!!!!!!!!! before UPATaskRunner::enqueTask !!!!!!!!!!!!!\n");
//    nn::common_runtime::NNCmxMemoryMap *nnCmx = util::MemoryMap::project<NNCmxMemoryMap>(NN_CMX_BASE);
//    printf("!!!!!!!!!! %s:%d !!!!!!!!!!!!!\n", __FILE__, __LINE__);
    auto globalAreas = getStaticMapping(nnCmx);
//    printf("!!!!!!!!!! %s:%d !!!!!!!!!!!!!\n", __FILE__, __LINE__);
    auto shaveManager = getShaveManager(globalAreas);
//    printf("!!!!!!!!!! %s:%d !!!!!!!!!!!!!\n", __FILE__, __LINE__);

    nn::act_runtime::ActKernelRuntimeConfigs actRtConfigs;  // Initialize properly
//    printf("!!!!!!!!!! %s:%d !!!!!!!!!!!!!\n", __FILE__, __LINE__);

    actRtConfigs.useScheduleEmbeddedRt_ = true;

    CustomCpp * customOp = static_cast<CustomCpp*>(operation);



//    // TODO: this can be made better by sharing a common stack set for all mappings
//    for (unsigned int j = 0; j < common_runtime::AS_TOTAL; ++j) {
//        rtcfg.stackFrames_[j] = wrapper.stacks_[j].resolve(nnrd).addr32();
//    }
//
//    nnLog(MVLOG_DEBUG, "actRt Buffer index: 0x%" PRIx32"", wrapper.kernelDataBuffer_.index());
//    nnLog(MVLOG_DEBUG, "actRt Buffer location: 0x%" PRIx32"", wrapper.kernelDataBuffer_.location());
//    nnLog(MVLOG_DEBUG, "actRt Buffer offset: 0x%" PRIx32"", wrapper.kernelDataBuffer_.offset(RelativeAddress::Class::Data));
//
//    rtcfg.actRtWindowBase_ = reinterpret_cast<unsigned char *>(wrapper.kernelDataBuffer_.resolve(nnrd).addr32());
//

//    actRtConfigs.actRtWindowBase_ = reinterpret_cast<unsigned char *>(sk_nnActEntry_3010xx_text);





    printf("!!!!!!!!!! %s:%d !!!!!!!!!!!!! sk_nnActEntry_3010xx_text_ref %p, &sk_nnActEntry_3010xx_text_ref %p, sk_nnActEntry_3010xx_text %p\n", __FILE__, __LINE__
            , sk_nnActEntry_3010xx_text_ref, &sk_nnActEntry_3010xx_text_ref, sk_nnActEntry_3010xx_text);//, shvNN0_nnActEntry, &shvNN0_nnActEntry);
    actRtConfigs.runtimeEntry_ = reinterpret_cast<nn::act_runtime::actRuntimeEntry>(sk_nnActEntry_3010xx_text);
    actRtConfigs.actRtWindowBase_ = reinterpret_cast<unsigned char*>(sk_nnActEntry_3010xx_text);
//    actRtConfigs.runtimeEntry_ = reinterpret_cast<nn::act_runtime::actRuntimeEntry>(shvNN0_nnActEntry);

    operation->parse(&layer);

    cache::flush(actRtConfigs);
    cache::flush(globalAreas);
    cache::flush(shaveManager);
    cache::flush(*globalAreas);
    cache::flush(*shaveManager);
    printf("!!!!!!!!!! %s:%d !!!!!!!!!!!!!\n", __FILE__, __LINE__);
//    return true;
    printf("!!!!!!!!!! before start !!!!!!!!!!!!!\n");
    shaveManager->startActShavesForTile(0, actRtConfigs, true);
    printf("!!!!!!!!!! after start !!!!!!!!!!!!!\n");
    act_runtime::ActKernelRange kRange = {nn::act_runtime::ActWLType::WL_KERNEL,
                                            reinterpret_cast<act_runtime::actKernelEntry>(customOp->ops.kernel),
                                            reinterpret_cast<act_runtime::actKernelTextBuffer>(customOp->ops.kernel),
                                            customOp->ops.kernelDataLen,
                                            0};
    cache::flush(kRange);

    const BarrierUserConfig userBariers = {0, 0, 0, 0, 0};
//        PhysicalBarrierMask wait_mask_;
//        PhysicalBarrierMask post_mask_;
//        unsigned short start_after_;
//        unsigned short clean_after_;
//        unsigned int virtual_dep_;
//    };
    cache::flush(userBariers);

    const BarrierGpioConfig gpioBarriers = {0, 0};
//    {
//        unsigned char group_;
//        unsigned char mask_;
//    };
    cache::flush(gpioBarriers);


    act_runtime::ActKernelInvocation kInvo = {&kRange,
            (void*)(customOp->ops.paramData),
            reinterpret_cast<act_runtime::actKernelDataBuffer>(customOp->ops.paramData),
            userBariers, gpioBarriers, 0
    };
    cache::flush(kInvo);
//    extern "C" struct ActKernelInvocation {
//        ActKernelRange *range_{nullptr};
//        act_kernel_args *kernelArgs_{nullptr};
//        actKernelDataBuffer dataWindowBase_{nullptr};
//
//        BarrierUserConfig barriers_{};
//        BarrierGpioConfig barriers_gpio_{};
//        unsigned int invo_index_{0};
//    };



    fifo::sendWorkToASs(0/*local_aki.tile_*/, &kInvo);

//    sleep(10);

    shaveManager->stopActShavesForTiles();
    printf("!!!!!!!!!! after stop !!!!!!!!!!!!!\n");
//    shaveManager.stopActShavesForTile(TILE_0);

    uint32_t * tmp = (uint32_t*)0x2e014000;
#define N_OF_LOGS 10
    printf( "!!!!!!!!!!!!!!!!!!!!!!!! Was I there: !!!!!!!!!!!!!!!!!!!!!!!!\n");
    cache::invalidate(tmp, sizeof(uint32_t));
    cache::invalidate(tmp, tmp[0] * sizeof(uint32_t));
    for (int i = 0; i < tmp[0]; i++) {
        printf( "\t\t%d) %d 0x%x\n", i, tmp[i], tmp[i]);
    }


    _enqued = true;
    return true;


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

//    operation->parse(&layer);
//
//    sl.layer_ = &layer;
//
//    nnLog(MVLOG_DEBUG, "Enqueuing SL @ %p\n", &sl);
//    nnLog(MVLOG_DEBUG, "           L @ %p\n", (void *)sl.layer_);
//    nnLog(MVLOG_DEBUG, "         ABA @ %p\n", &sl.abs_addr_);
//
//#if DEBUG_KERNELS
//    leonPipePrintFlushBuffer();
//#endif
//
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
    _enqued = true;

    return true;
}

bool UPATaskRunner::dequeResult() {
    return _enqued;
}
