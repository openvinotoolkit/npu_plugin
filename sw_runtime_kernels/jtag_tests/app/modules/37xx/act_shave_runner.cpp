//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

__attribute__((aligned(1024)))
#include "sk.nnActEntry.3720xx.text.xdat"
#include <nn_runtime_types.h>
#include "shave_task_runner.hpp"
#include <nn_shave_manager.h>
#include <nn_fifo_manager.h>
#include <nn_cache.h>
#include <nn_time.h>
#include <CustomCpp.h>
#include <leonUtils.h>

#include <HglBarrier.h>
#include <OsDrvBootFreq.h>

// #include <nn_perf_manager.h>

using namespace nn;
using namespace nn_public;
void const *actShvx_nnActEntry;
uint32_t lnn_to_lrt_line[16]; // dummy for watchdog

const int ConsumerNum = 0;
const int ProducerNum = 1;
const int ConsumerMask = (1 << ConsumerNum);
const int ProducerMask = (1 << ProducerNum);

// perf::ActPerfReport __attribute__((section(".nncmx0.shared.data"), aligned(64))) actPerfReport{0, 0, 0, 0, 0, 0};

// PerformanceData* perfData = nullptr;

ShaveTaskRunner::ShaveTaskRunner()
    : nnCmx_(util::MemoryMap::project<common_runtime::NNCmxMemoryMap>(NN_CMX_BASE))
    , globalAreas_(nnCmx_)
    , shave_manager_(globalAreas_, reinterpret_cast<common_runtime::ParsedShaveElfs *>(&shaveElfs))
    , actRtConfigs()
    , kRange()
    , kInvo() {
    cache::flush(&globalAreas_, math::round_up<NN_CACHE_LINE_LENGTH>(sizeof(globalAreas_)));

    // Allocate all Shave stack frames
    nn::memory::shared_unique_ptr<unsigned char> actShv0Stack(
        new (nn::memory::shared) unsigned char[SHAVE_STACK_SIZE]);
    nn::memory::shared_unique_ptr<unsigned char> actShv1Stack(
        new (nn::memory::shared) unsigned char[SHAVE_STACK_SIZE]);
    nn::memory::shared_unique_ptr<unsigned char> actShv2Stack(
        new (nn::memory::shared) unsigned char[SHAVE_STACK_SIZE]);
    nn::memory::shared_unique_ptr<unsigned char> actShv3Stack(
        new (nn::memory::shared) unsigned char[SHAVE_STACK_SIZE]);

    actRtConfigs.stackFrames_[0] = reinterpret_cast<uint32_t>(actShv0Stack.get());
    actRtConfigs.stackFrames_[1] = reinterpret_cast<uint32_t>(actShv1Stack.get());
    actRtConfigs.stackFrames_[2] = reinterpret_cast<uint32_t>(actShv2Stack.get());
    actRtConfigs.stackFrames_[3] = reinterpret_cast<uint32_t>(actShv3Stack.get());
    actRtConfigs.stackSize_ = SHAVE_STACK_SIZE;

    actRtConfigs.useScheduleEmbeddedRt_ = true;
    actRtConfigs.runtimeEntry_ = reinterpret_cast<uint32_t>(&sk_nnActEntry_3720xx_text);
    actRtConfigs.actRtWindowBase_ = reinterpret_cast<uint32_t>(&sk_nnActEntry_3720xx_text);
    cache::flush(actRtConfigs);
}

ShaveTaskRunner::~ShaveTaskRunner() {}

bool ShaveTaskRunner::enqueTask(CustomCppLayerParams ops) {//,
                              //PerformanceData *_perfData) {
    uint8_t tile = 0;

    shave_manager_.startActShavesForTiles((common_runtime::TileMask)(1 << tile), actRtConfigs);

    kRange.type_ = act_runtime::ActWLType::WL_KERNEL;
    kRange.kernelEntry_.ptr = ops.kernel;
    kRange.textWindowBase_ = ops.kernel;
    kRange.codeSize_ = ops.kernelDataLen;
    kRange.dataSecSize_ = 0;
    kRange.kInvoCount_ = 1;

    cache::flush(kRange);

    const act_runtime::TaskBarrierDependecy userBariers = {ConsumerMask, ProducerMask, 0, 0};
    const common_runtime::TaskSchedulingBarrierConfig schBarriers = {0, 1};

    cache::flush(userBariers);
    cache::flush(schBarriers);

    HglBarrierReset64(ConsumerMask);
    HglBarrierReset64(ProducerMask);
    HglBarrierSetProdConsCounts(ConsumerNum, 1, 1);
    HglBarrierSetProdConsCounts(ProducerNum, 1, 0);

    // perfData = _perfData;

    kInvo.range_ = reinterpret_cast<uint32_t>(&kRange);
    kInvo.kernelArgs_ = reinterpret_cast<uint32_t>(ops.paramData);
    kInvo.dataWindowBase_ = reinterpret_cast<uint32_t>(ops.paramData);
    kInvo.perfPacketOut_ = nullptr;
    kInvo.barriers_ = userBariers;
    kInvo.barriers_sched_ = schBarriers;
    kInvo.invoIndex_ = 0;
    kInvo.invoTile_ = 0;
    kInvo.kRangeIndex_ = 0;

    cache::flush(kInvo);

    common_runtime::fifo::sendWorkToASs(tile, &kInvo);

    HglBarrierProduce64(ConsumerMask);

    _enqued = true;
    return true;
}

bool ShaveTaskRunner::dequeResult() {
    if (_enqued) {
        HglBarrierWait64(ProducerMask);

        nnLog(MVLOG_DEBUG, "After send waiting is done");

        shave_manager_.stopActShavesForTiles();
        _enqued = false;
    }

    // if (perfData) {
    //     if (perfData->perfCounters)
    //         perfData->perfCounters->cycles = actPerfReport.duration;
    //     perfData->elapsedTimeNs = (actPerfReport.duration * 1000.0f) / OsDrvBootCalculateBootFrequency();
    //     perfData = nullptr;
    // }

    return true;
}
