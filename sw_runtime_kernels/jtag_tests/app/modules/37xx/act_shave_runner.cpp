//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

__attribute__((aligned(1024)))
#include "sk.nnActEntry.3010xx.text.xdat"
#include <sw_nn_runtime_types.h>
#include <sw_shave_lib_common.h>
#include <HglShaveCommon.h>
#include <HglShaveL1Cache.h>
#include "shave_task_runner.hpp"
#include <nn_shave_manager.h>
#include <nn_fifo_manager.h>
#include <nn_cache.h>
#include <nn_time.h>
#include <CustomCpp.h>
#include <leonUtils.h>

#include <HglBarrier.h>

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
constexpr uint32_t MAX_WAIT_ITERATIONS=1000000;
} // namespace

static SoftLayerExec __attribute__((section(".nncmx0.shared.data"))) sl;
static Layer __attribute__((section(".nncmx0.shared.data"))) layer;

using namespace nn;
extern bool HglShaveAccessAllowed[HGL_SHAVE_TYPE_NB];

//  FIXME: Temporarily are located on CMX due to problem of ACT_SHAVE cache invalidation
nn::act_runtime::ActKernelRuntimeConfigs actRtConfigs __attribute__((section(".nncmx0.shared.data")));   // Initialize properly
act_runtime::ActKernelRange kRange __attribute__((section(".nncmx0.shared.data")));
act_runtime::ActKernelInvocation kInvo __attribute__((section(".nncmx0.shared.data")));

nn::common_runtime::NNCmxMemoryMap *nnCmx = util::MemoryMap::project<NNCmxMemoryMap>(NN_CMX_BASE);

std::shared_ptr<nn::common_runtime::StaticMapping> getStaticMapping(nn::common_runtime::NNCmxMemoryMap *nnCmx) {
    static std::shared_ptr<nn::common_runtime::StaticMapping> holder(new (memory::cache_aligned) nn::common_runtime::StaticMapping(nnCmx), memory::cache_aligned_deleter<nn::common_runtime::StaticMapping>());
    return holder;
}

std::shared_ptr<nn::inference_runtime::shaves::ShaveManager> getShaveManager(std::shared_ptr<nn::common_runtime::StaticMapping> mapping) {
    static std::shared_ptr<nn::inference_runtime::shaves::ShaveManager> holder(new (memory::cache_aligned) nn::inference_runtime::shaves::ShaveManager(*mapping), memory::cache_aligned_deleter<nn::inference_runtime::shaves::ShaveManager>());
    return holder;
}

const int ConsumerNum = 0;
const int ProducerNum = 1;
const int ConsumerMask = (1 << ConsumerNum);
const int ProducerMask = (1 << ProducerNum);

bool ShaveTaskRunner::enqueTask(Op * operation,
                              const std::vector<OpTensor> &inputs,
                              const std::vector<OpTensor> &outputs,
                              int /*numSHAVEs*/,
                              PerformanceData *perfData) {

    actShaveDataReserved = 0;

    HglShaveAccessAllowed[1] = false;
    HglShaveAccessAllowed[2] = true;
    cache::flush(HglShaveAccessAllowed, sizeof(bool) * HGL_SHAVE_TYPE_NB);
    auto globalAreas = getStaticMapping(nnCmx);
    _shaveManager = getShaveManager(globalAreas);

    actRtConfigs.useScheduleEmbeddedRt_ = true;

    CustomCpp * customOp = static_cast<CustomCpp*>(operation);

    actRtConfigs.runtimeEntry_ = reinterpret_cast<nn::act_runtime::actRuntimeEntry>(sk_nnActEntry_3010xx_text);
    actRtConfigs.actRtWindowBase_ = reinterpret_cast<unsigned char*>(sk_nnActEntry_3010xx_text);

    operation->parse(&layer);

    cache::flush(actRtConfigs);
    cache::flush(globalAreas);
    cache::flush(_shaveManager);
    cache::flush(*globalAreas);
    cache::flush(*_shaveManager);
    _shaveManager->startActShavesForTile(0, actRtConfigs, true);

    act_runtime::ActKernelRange kRng = {nn::act_runtime::ActWLType::WL_KERNEL,
                                            reinterpret_cast<act_runtime::actKernelEntry>(customOp->ops.kernel),
                                            reinterpret_cast<act_runtime::actKernelTextBuffer>(customOp->ops.kernel),
                                            customOp->ops.kernelDataLen,
                                            0};

    kRange = kRng;
    cache::flush(kRange);

    const BarrierUserConfig userBariers = {ConsumerMask, ProducerMask, 0, 0, 0};
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

    HglBarrierReset(ConsumerMask);
    HglBarrierReset(ProducerMask);
    HglBarrierSetProdConsCounts(ConsumerNum, 1, 1);
    HglBarrierSetProdConsCounts(ProducerNum, 1, 0);

    act_runtime::ActKernelInvocation kInv = {&kRange,
            (void*)(customOp->ops.paramData),
            reinterpret_cast<act_runtime::actKernelDataBuffer>(customOp->ops.paramData),
            userBariers, gpioBarriers, 0
    };
    kInvo = kInv;
    cache::flush(kInvo);
    leonDataCacheFlush();
    fifo::sendWorkToASs(0/*local_aki.tile_*/, &kInvo);

    HglBarrierProduce(ConsumerMask);
    _enqued = true;
    return true;
}

bool ShaveTaskRunner::dequeResult() {
    if (_enqued) {
        HglBarrierWait(ProducerMask);

        nnLog(MVLOG_DEBUG, "After send waiting is done");

        _shaveManager->stopActShavesForTiles();
        _enqued = false;
    }
    return true;
}
