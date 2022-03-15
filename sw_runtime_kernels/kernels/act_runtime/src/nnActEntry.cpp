//
// Copyright 2022 Intel Corporation.
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

#include <DrvRegUtils.h>
//#include <DrvSvuL1Cache.h>
#include <cpuWhoAmI.h>
#include <nn_barrier.h>
#include <nn_fifo.h>
#include <nn_fifo_manager.h>
#include <nn_perf_manager.h>
#include <nn_counter.h>
#include <string.h>

#include <nnActRtUtils.h>
#include <nnActRtPerf.h>
#include <nnActRtDebug.h>

#include <sys/__moviconfig.h>

#define P_CFG_SETTING ~0b011110

#if defined(USE_SHAVE_NN_PRINT)
#include <stdio.h>
#define PRINTF(...) printf(__VA_ARGS__)
#else
#define PRINTF(...)
#endif

using namespace nn::act_runtime;
using namespace nn::util;
using namespace nn::common_runtime::fifo;

extern "C" void nnActEntry(void *config, void *scratch) {
    const SHVFifoConfig fifoCfg = unpackSHVConfig(reinterpret_cast<uint32_t>(config));
    const uint32_t wlFifoAddr = computeFifoRecieveAddress(fifoCfg.work.fifo, fifoCfg.work.index);
    const uint32_t ctFifoAddr = computeFifoRecieveAddress(fifoCfg.ctrx.fifo, fifoCfg.ctrx.index);
    const uint32_t prFifoAddr = computeFifoRecieveAddress(fifoCfg.perf.fifo, fifoCfg.perf.index);

    // TODO: double check that this is working now with latest tools
    const unsigned int shaveIndex = cpuWhoAmI() - PROCESS_ACT_SHAVE0;
    // const unsigned int shaveIndex = __builtin_shave_getcpuid();
    UNUSED(prFifoAddr);

    ActKernelInvocation *ki{nullptr};
    ActKernelRange *kr{nullptr};

    ActPerfReport pr;
    char packedPr[sizeof(ActPerfReport)];
    uint32_t perfPackedSize{0};
    uint32_t perfMetricMask{0};

    auto handleCtrl = [&](uint32_t fifo_val) {
        const ASCtrlMessage ctrl = unpackASCtrlMessage(reinterpret_cast<uint32_t>(fifo_val));

        switch (ctrl.message) {
            case SHVCtrlMessage::HWStatsEnable:
                break;
            case SHVCtrlMessage::PreemptHaltAndAck:
                break;
            case SHVCtrlMessage::EnablePerfStream:
                perfMetricMask = ctrl.payload;
                perfPackedSize = actPRPackedSize(perfMetricMask);
                configCounters(perfMetricMask);
                break;
            case SHVCtrlMessage::DisablePerfStream:
                perfMetricMask = 0;
                perfPackedSize = 0;
                break;

            default:
                break;
        }
    };

    auto handleKRChange = [&]() {
        // do something with the previous kRange
        // TODO: maybe do perf roll-up and push to perf FIFO?

        /*
         * TODO: we also need to prefetch the .text to L2
         * Note that a-shvs will share the same iL2 partition (per tile), so we may be spamming the prefetch here.
         *   Use a free HW mutex?
         */
        kr = ki->range_;

        setShaveWindow(1, kr->textWindowBase_);

        // sDrvPfetchDl1LineL();
        // sDrvPfetchDl2(ki->data_);
    };

    auto writeDbgVal = [scratch](uint64_t val) {
        static volatile uint64_t *dbg = reinterpret_cast<uint64_t *>(scratch);
        *(dbg++) = val;
    };

    auto writeFRC = [writeDbgVal](void) {
#ifdef CONFIG_VALIDATION_APP_ENABLED
        writeDbgVal(nn::util::sampleFRC());
#else
        UNUSED(writeDbgVal);
#endif
    };

#ifdef ACT_RT_DEBUG
    /*
     * WARNING: This debug helper will almost certainly corrupt an inference and is _not_ safe to call by multiple
     * shaves from any tile > 0. Use at your own risk. It's only intended as a fast debugging tool to avoid MoviDebug's
     * complicated debuging features.
     */
    auto cmxDebugStride = [](uint32_t value) {
        // NOTE!: .data* sections are windowed to same window as .text for the ActRT.
        //        That means all shaves share the same .data!
        static uint32_t *debug{(uint32_t *)(0x2E000000 + 1024 * 1024 - 1024)};
        static uint32_t next{0};

        if (next < 1024) {
            *reinterpret_cast<uint32_t *>((reinterpret_cast<uint32_t>(debug) + next)) = value;
            next += 4;
        }
    };

    auto waitWL = [&]() {
        uint32_t ct;

        do {
            ki = reinterpret_cast<ActKernelInvocation *>(GET_REG_WORD_VAL(wlFifoAddr));
            ct = GET_REG_WORD_VAL(ctFifoAddr);

            if (ct != 0) {
                handleCtrl(ct);
            }
        } while (ki == 0);
    };
#else
    auto waitWL = [&]() {
        if (fifoWaitGpioWithCtrl(fifoCfg.work.fifo, fifoCfg.ctrx.fifo)) {
            ki = reinterpret_cast<ActKernelInvocation *>(GET_REG_WORD_VAL(wlFifoAddr));
        } else {
            ki = nullptr;
            handleCtrl(GET_REG_WORD_VAL(ctFifoAddr));
        }
        writeFRC();
    };
#endif

    auto execWL = [&]() {
        const auto &barriers = ki->barriers_;
        const auto &barriers_gpio = ki->barriersGpio_;

        setShaveWindow(2, ki->dataWindowBase_);

        waitBarrier(barriers, barriers_gpio, shaveIndex);
        HglBarrierConsume(barriers.wait_mask_);

        if (perfMetricMask) {
            resetCounters(pr);

            (kr->kernelEntry_)(ki->kernelArgs_);

            recordCounters(pr);
            packActPerfReport(perfMetricMask, pr, reinterpret_cast<void *>(packedPr));

            if (ki->perfPacketOut_) {
                memcpy_s(ki->perfPacketOut_, sizeof(ActPerfReport), reinterpret_cast<const void *>(packedPr),
                         perfPackedSize);
            } else {
                // TODO: stream it out
            }
        } else {
            writeFRC();
#if defined(LOW_LEVEL_TESTS_PERF)
            if (ki->perfPacketOut_) {
                perfMetricMask = (0b1 << perf::FRC_TIMESTAMP_EN) | (0b1 << perf::FRC_DURATION_EN);
                perfPackedSize = actPRPackedSize(perfMetricMask);
                //configCounters(perfMetricMask); // the call hangs but not need for TIMESTAMP|DURATION
                resetCounters(pr);
            }
#endif // LOW_LEVEL_TESTS_PERF
            (kr->kernelEntry_)(ki->kernelArgs_);
#if defined(LOW_LEVEL_TESTS_PERF)
            if (ki->perfPacketOut_) {
                recordCounters(pr);
                packActPerfReport(perfMetricMask, pr, reinterpret_cast<void *>(packedPr));
                memcpy_s(ki->perfPacketOut_, sizeof(ActPerfReport), reinterpret_cast<const void *>(packedPr),
                         perfPackedSize);
            }
#endif // LOW_LEVEL_TESTS_PERF
        }

        HglBarrierProduce(barriers.post_mask_);
    };

    setFPRound(P_CFG_SETTING);

    do {
        waitWL();

        if (ki) {
            if (ki->range_ != kr)
                handleKRChange();

            switch (kr->type_) {
                case ActWLType::WL_KERNEL: {
                    execWL();
                    break;
                }
#ifdef NN_ENABLE_CONTEXT_DEBUGGING
                case ActWLType::WL_DEBUG: {
                    execDebug(kr, shaveIndex, fifoCfg);
                    break;
                }
#endif
                case ActWLType::WL_UNKNOWN: {
                    break;
                }
                default:
                    break;
            }
        }
    } while (true);
}
