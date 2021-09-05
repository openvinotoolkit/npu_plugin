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

// #ifdef NN_ACT_PROFILING
// void execPerf(ActWorkload *ki) {
//     // TODO: wrap this up with other perf flags?
//     PRINTF("Profiling on ACT_SHAVE Using not implemented!\n");
// }
// #endif

extern "C" void nnActEntry(void *config) {
    const SHVFifoConfig fifoCfg = unpackSHVConfig(reinterpret_cast<uint32_t>(config));
    const unsigned int wlFifoAddr = computeFifoRecieveAddress(fifoCfg.work.fifo, fifoCfg.work.index);
    const unsigned int ctFifoAddr = computeFifoRecieveAddress(fifoCfg.ctrx.fifo, fifoCfg.ctrx.index);
    const unsigned int prFifoAddr = computeFifoRecieveAddress(fifoCfg.perf.fifo, fifoCfg.perf.index);
    const unsigned int shaveIndex = cpuWhoAmI() - PROCESS_ACT_SHAVE0;
    UNUSED(prFifoAddr);
    UNUSED(shaveIndex);

    ActKernelInvocation *ki{nullptr};
    ActKernelRange *kr{nullptr};

    auto readWL = [&]() { return reinterpret_cast<ActKernelInvocation *>(GET_REG_WORD_VAL(wlFifoAddr)); };

    auto waitWL = [&]() {
        fifoWaitGpio(fifoCfg.work.fifo);
        return readWL();
    };

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

        // sDrvPfetchDl1LineL();
        // sDrvPfetchDl2(ki->data_);
    };

    // Set the window address by writing to the local address space of the current SHAVE
    auto setShaveWindows = [&](uint32_t windowNumber, uint32_t targetWindowBaseAddr) {
        switch (windowNumber)
        {
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
    };
    UNUSED(setShaveWindows);

    auto execWL = [&]() {
        // TODO: run the kernel
        (kr->kernelEntry_)(ki->kernelArgs_.args_, ki->kernelArgs_.numArgs_);
    };

    while (true) {
        if (!isfifoEmpty(fifoCfg.ctrx.fifo)) {
            handleCtrl();
        }
        ki = waitWL();

        while (ki) {
            if (ki->range_ != kr)
                handleKRChange();

            switch (kr->type_) {
                case ActWLType::KERNEL: {
                    execWL();
                    break;
                }
#ifdef NN_ENABLE_CONTEXT_DEBUGGING
                case ActWLType::DEBUG: {
                    execDebug(kr, shaveIndex, fifoCfg);
                    break;
                }
#endif
                case ActWLType::UNKNOWN: {
                    break;
                }
                default:
                    break;
            }

            ki = readWL();
        }
    }
}
