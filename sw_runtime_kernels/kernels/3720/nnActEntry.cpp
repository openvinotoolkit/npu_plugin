/*
 * {% copyright %}
 */

#include <DrvRegUtils.h>
#include <DrvSvuL1Cache.h>
#include <cpuWhoAmI.h>
#include <nn_barrier.h>
#include <nn_fifo.h>
#include <nn_fifo_manager.h>
#include <nnActRtDebug.h>

#include <sys/__moviconfig.h>

#if defined(USE_SHAVE_NN_PRINT)
#include <stdio.h>
#define PRINTF(...) printf(__VA_ARGS__)
#else
#define PRINTF(...)
#endif

using namespace nn::act_runtime;
using namespace nn::util;
using namespace nn::common_runtime::fifo;

#ifdef NN_ACT_PROFILING
void execPerf(ActWorkload *ki) {
    // TODO: wrap this up with other perf flags?
    PRINTF("Profiling on ACT_SHAVE Using not implemented!\n");
}
#endif

inline void waitBarrier(const BarrierUserConfig &bar, const BarrierGpioConfig &gpio, unsigned int shave_index) {
    // TODO: enable GPIO monitor when shave_index is confirmed working
    if (false && gpio.group_ > 0) {
        HglBarrierMonitorSelect(shave_index, gpio.group_ - 1);
        waitBarrierGpio(gpio.mask_);
    } else
        HglBarrierWait(bar.wait_mask_);
}

// Set the window address by writing to the local address space of the current SHAVE
inline void setShaveWindow(uint32_t windowNumber, void *targetWindowBaseAddr) {
    switch (windowNumber) {
        case 0:
            asm volatile("lsu0.sta.32 %[addr], SHAVE_LOCAL, 0x10" ::[addr] "r"(targetWindowBaseAddr));
            break;
        case 1:
            asm volatile("lsu0.sta.32 %[addr], SHAVE_LOCAL, 0x14" ::[addr] "r"(targetWindowBaseAddr));
            break;
        case 2:
            asm volatile("lsu0.sta.32 %[addr], SHAVE_LOCAL, 0x18" ::[addr] "r"(targetWindowBaseAddr));
            break;
        case 3:
            asm volatile("lsu0.sta.32 %[addr], SHAVE_LOCAL, 0x1c" ::[addr] "r"(targetWindowBaseAddr));
            break;
    }
}

inline void setFPRound(const unsigned int actShvID) {
    // Set FP round to x86 compatibility mode for verification
    auto myBase{0};

    switch (actShvID) {
        case 0:
            myBase = ACT_SHV_0_BASE;
            break;
        case 1:
            myBase = ACT_SHV_1_BASE;
            break;
        case 2:
            myBase = ACT_SHV_2_BASE;
            break;
        case 3:
            myBase = ACT_SHV_3_BASE;
            break;
        default:
            myBase = ACT_SHV_0_BASE;
            break;
    }

    auto reg = GET_REG_WORD_VAL(myBase + P_CFG_OFFSET);
    reg = reg & (~0b011110); // F2INTRND; 0x0 - round to nearest even
    SET_REG_WORD(myBase + P_CFG_OFFSET, reg);
}

extern "C" void nnActEntry(void *config) {
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

    auto handleCtrl = [&]() {
        const SNNCtrlMessage ctrl = unpackSNNCtrlMessage(reinterpret_cast<uint32_t>(GET_REG_WORD_VAL(ctFifoAddr)));

        switch (ctrl.message) {
            case nn::SHVCtrlMessage::HWStatsEnable:
                break;
            case nn::SHVCtrlMessage::PreemptHaltAndAck:
                break;
            case nn::SHVCtrlMessage::EnablePerfStream:
                break;
            case nn::SHVCtrlMessage::DisablePerfStream:
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

#ifdef ACT_RT_DEBUG
    /*
     * WARNING: This debug helper will almost certainly corrupt an inference and is _not_ safe to call by multiple
     * shaves from any tile > 0. Use at your own risk. It's only intended as a fast debugging tool to avoid MoviDebug's
     * complicated debuging features.
     */
    auto cmxDebugStride = [&](uint32_t value) {
        // NOTE!: .data* sectios are windowed to same window as .text for the ActRT.
        //        That means all shaves share the same .data!
        static uint32_t *debug{(uint32_t *)(0x2E000000 + 1024 * 1024 - 1024)};
        static uint32_t next{0};

        if (next < 1024) {
            *reinterpret_cast<uint32_t *>((reinterpret_cast<uint32_t>(debug) + next)) = value;
            next += 4;
        }
    };

    auto waitWL = [&]() {
        do {
            ki = reinterpret_cast<ActKernelInvocation *>(GET_REG_WORD_VAL(wlFifoAddr));
        } while (ki == 0);
    };
#else
    auto waitWL = [&]() {
        if (fifoWaitGpioWithCtrl(fifoCfg.work.fifo, fifoCfg.ctrx.fifo)) {
            ki = reinterpret_cast<ActKernelInvocation *>(GET_REG_WORD_VAL(wlFifoAddr));
        } else {
            ki = nullptr;
            handleCtrl();
        }
    };
#endif

    auto execWL = [&]() {
        const auto &barriers = ki->barriers_;
        const auto &barriers_gpio = ki->barriers_gpio_;

        setShaveWindow(2, ki->dataWindowBase_);

        waitBarrier(barriers, barriers_gpio, shaveIndex);
        HglBarrierConsume(barriers.wait_mask_);

        (kr->kernelEntry_)(ki->kernelArgs_);

        HglBarrierProduce(barriers.post_mask_);
    };

    setFPRound(shaveIndex);

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
