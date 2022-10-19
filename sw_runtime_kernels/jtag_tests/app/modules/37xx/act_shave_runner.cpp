//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

__attribute__((aligned(1024)))
#include "sk.nnActEntry.3720xx.text.xdat"
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
#include <OsDrvBootFreq.h>

#include <nn_perf_manager.h>

namespace {
using namespace nn::inference_runtime;
using namespace nn::common_runtime;

const uint32_t NN_CMX_BASE = 0x2e000000;
} // namespace

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

perf::ActPerfReport __attribute__((section(".nncmx0.shared.data"), aligned(64))) actPerfReport{0, 0, 0, 0, 0, 0};

PerformanceData* perfData = nullptr;

bool ShaveTaskRunner::enqueTask(Op * operation,
                              const std::vector<OpTensor> &inputs,
                              const std::vector<OpTensor> &outputs,
                              int /*numSHAVEs*/,
                              PerformanceData *_perfData) {
    (void)inputs;
    (void)outputs;

    HglShaveAccessAllowed[1] = false;
    HglShaveAccessAllowed[2] = true;
    cache::flush(HglShaveAccessAllowed, sizeof(bool) * HGL_SHAVE_TYPE_NB);
    auto globalAreas = getStaticMapping(nnCmx);
    _shaveManager = getShaveManager(globalAreas);

    actRtConfigs.useScheduleEmbeddedRt_ = true;

    CustomCpp * customOp = static_cast<CustomCpp*>(operation);

    actRtConfigs.runtimeEntry_ = reinterpret_cast<nn::act_runtime::actRuntimeEntry>(sk_nnActEntry_3720xx_text);
    actRtConfigs.actRtWindowBase_ = reinterpret_cast<unsigned char*>(sk_nnActEntry_3720xx_text);

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

    HglBarrierReset64(ConsumerMask);
    HglBarrierReset64(ProducerMask);
    HglBarrierSetProdConsCounts(ConsumerNum, 1, 1);
    HglBarrierSetProdConsCounts(ProducerNum, 1, 0);

    perfData = _perfData;

    act_runtime::ActKernelInvocation kInv = {&kRange,
            (void*)(customOp->ops.paramData),
            reinterpret_cast<act_runtime::actKernelDataBuffer>(customOp->ops.paramData),
            userBariers, gpioBarriers, (perfData ? &actPerfReport : 0)
    };
    kInvo = kInv;
    cache::flush(kInvo);
    leonDataCacheFlush();
    fifo::sendWorkToASs(0/*local_aki.tile_*/, &kInvo);

    HglBarrierProduce64(ConsumerMask);

    _enqued = true;
    return true;
}

bool ShaveTaskRunner::dequeResult() {
    if (_enqued) {
        HglBarrierWait64(ProducerMask);

        nnLog(MVLOG_DEBUG, "After send waiting is done");

        _shaveManager->stopActShavesForTiles();
        _enqued = false;
    }

    if (perfData) {
        if (perfData->perfCounters)
            perfData->perfCounters->cycles = actPerfReport.duration;
        perfData->elapsedTimeNs = (actPerfReport.duration * 1000.0f) / OsDrvBootCalculateBootFrequency();
        perfData = nullptr;
    }

    return true;
}
