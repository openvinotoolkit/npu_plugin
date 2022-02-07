#define TRY_HARDCODED_PERF (1)

/*
 * {% copyright %}
 */

#include <DrvRegUtils.h>
#include <DrvSvuL1Cache.h>
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

enum class ActRtTag : uint8_t {
    INVALID = 0x00,
    DEBUG_FIFO_WL_DEQUE,
    DEBUG_BARRIER_CONSUME,
    DEBUG_WL_BEGIN,
    DEBUG_BARRIER_PRODUCE,
};

extern "C" void nnActEntry(void *config, void *scratch) {
    uint32_t * tmp = (uint32_t *)0x2e014000;
    uint32_t& debInd = *tmp;
    debInd = 1;
    tmp[debInd++] = 111111;
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
        tmp[debInd++] = 222220;
        tmp[debInd++] = fifo_val;
        /*const*/ ASCtrlMessage ctrl = unpackASCtrlMessage(fifo_val);

//ctrl.message = SHVCtrlMessage::EnablePerfStream;

        switch (ctrl.message) {
            case SHVCtrlMessage::HWStatsEnable:
                tmp[debInd++] = 222221;
                break;
            case SHVCtrlMessage::PreemptHaltAndAck:
                tmp[debInd++] = 222222;
                break;
            case SHVCtrlMessage::EnablePerfStream:
                tmp[debInd++] = 2222230;
                perfMetricMask = ctrl.payload;
                tmp[debInd++] = 2222231;
                perfPackedSize = actPRPackedSize(perfMetricMask);
                tmp[debInd++] = 2222232;
                configCounters(perfMetricMask);
                tmp[debInd++] = 2222233;
                break;
            case SHVCtrlMessage::DisablePerfStream:
                tmp[debInd++] = 222224;
                perfMetricMask = 0;
                perfPackedSize = 0;
                break;
            default:
                tmp[debInd++] = 222225;
                break;
        }
        tmp[debInd++] = 222226;
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

        // TODO: We may need to flush dL2 after the kr is done...
    };

    auto writeDbgVal = [scratch](uint64_t val) {
        static volatile uint64_t *dbg = reinterpret_cast<uint64_t *>(scratch);
        *(dbg++) = val;
    };

    auto writeFRC = [writeDbgVal](ActRtTag tag) {
#ifdef CONFIG_ACTSHV_START_STOP_PERF_TEST
        uint64_t sample = nn::util::sampleFRC();
        uint64_t val = sample & 0x00FFFFFFFFFFFFFF | ((static_cast<uint64_t>(tag)) << 60);
        writeDbgVal(val);
#else
        UNUSED(writeDbgVal);
        UNUSED(tag);
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
#endif

#if 1
    auto _fifoWaitGpioWithCtrl = [&](unsigned int wlFifo, unsigned int ctrlFifo) -> bool {
        (void)ctrlFifo;
        // SNN fifo0 bit24 (shave nn monitor only)
        // SNN fifo1 bit25 (shave nn monitor only)
        // SNN fifo2 bit0
        const unsigned int wlFifoEmptyBit = wlFifo != 2 ? (1u << (24 + (wlFifo == 1))) : 1u;
        const unsigned int ctrlFifoEmptyBit = ctrlFifo != 2 ? (1u << (24 + (ctrlFifo == 1))) : 1u;
        const unsigned int fifoBits = wlFifoEmptyBit | ctrlFifoEmptyBit;

        tmp[debInd++] = 555550;
        tmp[debInd++] = wlFifoEmptyBit;
        tmp[debInd++] = ctrlFifoEmptyBit;
        tmp[debInd++] = fifoBits;

        fifoWaitGPIBits(fifoBits);
        return checkGPIBits(ctrlFifoEmptyBit);
    };
#endif

    auto waitWL = [&]() {
        tmp[debInd++] = 777710;
        tmp[debInd++] = fifoCfg.work.fifo;
        tmp[debInd++] = fifoCfg.work.index;
        tmp[debInd++] = fifoCfg.ctrx.fifo;
        tmp[debInd++] = fifoCfg.ctrx.index;
        if (_fifoWaitGpioWithCtrl(fifoCfg.work.fifo, fifoCfg.ctrx.fifo)) {
            ki = reinterpret_cast<ActKernelInvocation *>(GET_REG_WORD_VAL(wlFifoAddr));
            tmp[debInd++] = 777711;
        } else {
            ki = nullptr;
            handleCtrl(GET_REG_WORD_VAL(ctFifoAddr));
            tmp[debInd++] = 777712;
        }

        writeFRC(ActRtTag::DEBUG_FIFO_WL_DEQUE);
        tmp[debInd++] = 777719;
    };

#if defined(TRY_HARDCODED_PERF)
    auto _directAddrSta = [&](uint32_t addr, uint32_t val) {
        tmp[debInd++] = 5555700;
        switch (addr) {
            case SVU_PCC0_OFFSET:
                tmp[debInd++] = 5555701;
                SET_REG_WORD_SHAVE_LOCAL(SVU_PCC0_OFFSET, val);
                break;
            case SVU_PCC1_OFFSET:
                tmp[debInd++] = 5555702;
                SET_REG_WORD_SHAVE_LOCAL(SVU_PCC1_OFFSET, val);
                break;
            case SVU_PCC2_OFFSET:
                tmp[debInd++] = 5555703;
                SET_REG_WORD_SHAVE_LOCAL(SVU_PCC2_OFFSET, val);
                break;
            case SVU_PCC3_OFFSET:
                tmp[debInd++] = 5555704;
                SET_REG_WORD_SHAVE_LOCAL(SVU_PCC3_OFFSET, val);
                break;
        }
        tmp[debInd++] = 5555799;
    };
    auto _configCounters = [&](uint32_t metricMask) {
        uint32_t counterIndex{4};

        /*
         * NOTE: we must parse in PCCBitOffsets order (LSB upward) to match what the schema defines!
         * This order is used at perf payload parse time.
         */

        tmp[debInd++] = 5555501;
        tmp[debInd++] = metricMask;
        if (metricMask & (1 << BIT_STALL_CYCLE_CNT_EN)) {
            // [26] Instruction buffer Low during discontinuity
            // [25] Discontinuity Starve Stall
            // [24] Discontinuity Decode Stall (too much data in instruction buffer at end of delay slots)
            // [23] Discontinuity Fetch Stall
            // [22] Instruction buffer Low Stall
            // [21] LSU1 Access Stall
            // [20] LSU0 Access Stall
            // [19] LSU1 Stall (waiting for data)
            // [18] LSU0 Stall (waiting for data)
            // [17] Other interrupts
            // [16] SWIH

            counterIndex--;
            const uint32_t stCntTarget{(metricMask & (0b11111111111 << 16)) | 0b1};
            _directAddrSta(pcc_offsets[counterIndex], stCntTarget);
        }
        tmp[debInd++] = 5555502;
        if (metricMask & (1 << BIT_EXEC_INST_CNT_EN)) {
            counterIndex--;
            _directAddrSta(pcc_offsets[counterIndex], 0b1 << BIT_EXEC_INST_CNT_EN);
        }
        tmp[debInd++] = 5555503;
        if (metricMask & (1 << BIT_CLK_CYCLE_CNT_EN)) {
            counterIndex--;
            _directAddrSta(pcc_offsets[counterIndex], 0b1 << BIT_CLK_CYCLE_CNT_EN);
        }
        tmp[debInd++] = 5555504;
        if (metricMask & (1 << BIT_BRANCH_TAKEN_CNT_EN)) {
            counterIndex--;
            _directAddrSta(pcc_offsets[counterIndex], 0b1 << BIT_BRANCH_TAKEN_CNT_EN);
        }
        tmp[debInd++] = 5555505;
        if (counterIndex && (metricMask & (1 << BIT_INST_BRKP0_CNT_EN))) {
            counterIndex--;
            _directAddrSta(pcc_offsets[counterIndex], 0b1 << BIT_INST_BRKP0_CNT_EN);
        }
        tmp[debInd++] = 5555506;
        if (counterIndex && (metricMask & (1 << BIT_INST_BRKP1_CNT_EN))) {
            counterIndex--;
            _directAddrSta(pcc_offsets[counterIndex], 0b1 << BIT_INST_BRKP1_CNT_EN);
        }
        tmp[debInd++] = 5555507;
        if (counterIndex && (metricMask & (1 << BIT_DATA_BRKP0_CNT_EN))) {
            counterIndex--;
            _directAddrSta(pcc_offsets[counterIndex], 0b1 << BIT_DATA_BRKP0_CNT_EN);
        }
        tmp[debInd++] = 5555508;
        if (counterIndex && (metricMask & (1 << BIT_DATA_BRKP1_CNT_EN))) {
            counterIndex--;
            _directAddrSta(pcc_offsets[counterIndex], 0b1 << BIT_DATA_BRKP1_CNT_EN);
        }
        tmp[debInd++] = 5555509;
        if (counterIndex && (metricMask & (1 << BIT_GO_COUNT_EN))) {
            counterIndex--;
            _directAddrSta(pcc_offsets[counterIndex], 0b1 << BIT_GO_COUNT_EN);
        }
        tmp[debInd++] = 5555510;
        if (counterIndex && (metricMask & (1 << BIT_LSU0_RBYTE_CNT_EN))) {
            // [12:9] are accumulated into one counter, so don't decrament counterIndex
            _directAddrSta(pcc_offsets[counterIndex], 0b1 << BIT_LSU0_RBYTE_CNT_EN);
        }
        tmp[debInd++] = 5555511;
        if (counterIndex && (metricMask & (1 << BIT_LSU0_WBYTE_CNT_EN))) {
            const uint32_t cur = directAddrLda(pcc_offsets[counterIndex]);
            _directAddrSta(pcc_offsets[counterIndex], (0b1 << BIT_LSU0_WBYTE_CNT_EN) | cur);
        }
        tmp[debInd++] = 5555512;
        if (counterIndex && (metricMask & (1 << BIT_LSU1_RBYTE_CNT_EN))) {
            const uint32_t cur = directAddrLda(pcc_offsets[counterIndex]);
            _directAddrSta(pcc_offsets[counterIndex], (0b1 << BIT_LSU1_RBYTE_CNT_EN) | cur);
        }
        tmp[debInd++] = 5555513;
        if (counterIndex && (metricMask & (1 << BIT_LSU1_WBYTE_CNT_EN))) {
            const uint32_t cur = directAddrLda(pcc_offsets[counterIndex]);
            _directAddrSta(pcc_offsets[counterIndex], (0b1 << BIT_LSU1_WBYTE_CNT_EN) | cur);
        }
        tmp[debInd++] = 5555599;
    };
#endif // TRY_HARDCODED_PERF

    auto execWL = [&]() {
        const auto &barriers = ki->barriers_;
        const auto &barriers_gpio = ki->barriersGpio_;

        setShaveWindow(2, ki->dataWindowBase_);

        waitBarrier(barriers, barriers_gpio, shaveIndex);
        HglBarrierConsume(barriers.wait_mask_);
        writeFRC(ActRtTag::DEBUG_BARRIER_CONSUME);

        tmp[debInd++] = 333330;
        tmp[debInd++] = perfMetricMask;
        tmp[debInd++] = 333331;
        if (perfMetricMask) {
            tmp[debInd++] = 333332;
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
            tmp[debInd++] = 333333;
        } else {
            tmp[debInd++] = 333334;
            writeFRC(ActRtTag::DEBUG_WL_BEGIN);
#if defined(TRY_HARDCODED_PERF)
            tmp[debInd++] = 3333341;
            perfMetricMask = (0b1 << perf::FRC_TIMESTAMP_EN) | (0b1 << perf::FRC_DURATION_EN);
            tmp[debInd++] = 3333342;
            perfPackedSize = actPRPackedSize(perfMetricMask);
            tmp[debInd++] = 3333343;
            _configCounters(perfMetricMask);
            tmp[debInd++] = 3333344;
            resetCounters(pr);
            tmp[debInd++] = 3333345;
#endif // TRY_HARDCODED_PERF
            (kr->kernelEntry_)(ki->kernelArgs_);
#if defined(TRY_HARDCODED_PERF)
            tmp[debInd++] = 3333346;
            recordCounters(pr);
            tmp[debInd++] = 3333347;
            packActPerfReport(perfMetricMask, pr, reinterpret_cast<void *>(packedPr));
            tmp[debInd++] = 3333348;
            void* perfPacketOut = (void*)(0x2e014000 + 1024);
            tmp[debInd++] = 3333349;
//            memcpy_s(perfPacketOut, sizeof(ActPerfReport), reinterpret_cast<const void *>(packedPr), perfPackedSize);
            if (ki->perfPacketOut_) {
                tmp[debInd++] = (uint32_t)(ki->perfPacketOut_);
                memcpy_s(ki->perfPacketOut_, sizeof(ActPerfReport), reinterpret_cast<const void *>(packedPr),
                         perfPackedSize);
            }
#endif // TRY_HARDCODED_PERF
            tmp[debInd++] = 333335;
        }
        tmp[debInd++] = 333336;

        writeFRC(ActRtTag::DEBUG_BARRIER_PRODUCE);
        HglBarrierProduce(barriers.post_mask_);
        tmp[debInd++] = 333337;
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
