/*
* {% copyright %}
*/
#include "svu_nn_runtime.h"
#include "svu_nn_debug.h"
#include "svu_nn_irq.h"
#include "svu_nn_memory.h"
#include "svu_nn_util.h"

#include "sw_shave_lib_common.h"
#include "sw_shave_res_manager.h"
#include <assert.h>
#include <string.h>
#include <ShDrvCmxDma.h>
#include <svuCommonShave.h>
#include <nn_log.h>

#include "registersMyriad.h"
#include <DrvSvuCounters.h>
#include <lDrvSvuDefines.h>

extern "C" {
#include <CmxFifoShAsm.h>
#include <DrvSvuL1Cache.h>
#include <ShDrvMutex.h>
}

/**
 * NO_SVU_NN_CONTROLLER runs the NN SHAVE pool in very different dispatch model.
 * With NO_SVU_NN_CONTROLLER set, NN shaves expect all work done by the Preamble
 * and Cleanup to be performed externally before enqueuing SLTs to the CMXFIFOs.
 * This work can either be precomputed by the schedule (2.6+), or executed on
 * LNN via the Inference Runtime (2.0, 2.1).
 */
// #define NO_SVU_NN_CONTROLLER

// Block when reading from the CMX FIFO
#define CMX_FIFO_BLOCKING_READ 1

#ifdef SVU_STACK_USAGE_INSTRUMENTATION
#    define SVU_STACK_POINTER_REGISTER_ID (19) // Stack pointer register     I19
#    define SVU_REG_SLICE_SIZE (0x10000)
#    define SHAVE_BASE_ADDR(shaveId) (uint32_t)(SHAVE_0_BASE_ADR + (SVU_REG_SLICE_SIZE * shaveId))
#    define DCU_SVU_IRF(shave, regNum)                                                                                 \
        (SHAVE_BASE_ADDR(shave) + SLC_OFFSET_SVU + IRF_BASE + (4 * (regNum))) //  32 * 32 bits
extern uint32_t __stackHighWater;
extern uint32_t __stackMaximumExtent;
#endif

namespace nn {
namespace shave_lib {

SVUNNRuntime::SVUNNRuntime(svuNNRtInit *init)

// clang-format off
#ifdef NO_SVU_NN_CONTROLLER
    : isController{ false },
#else
    : isController{ isControllerShave(init) },
#endif
      lyr{ nullptr }, sle{ nullptr }, aba{}

#ifdef SVU_STACK_USAGE_INSTRUMENTATION
,   stackMaxExtent{ (uint32_t *)&__stackMaximumExtent },
    stackHighWater{ (uint32_t *)&__stackHighWater },
    stackStartAddr{ GET_REG_WORD_VAL(DCU_SVU_IRF(scGetShaveNumber(), SVU_STACK_POINTER_REGISTER_ID)) },
    stackSize{stackStartAddr - *stackMaxExtent}
#endif
// clang-format on
{
    memset(&perfCounters, 0, sizeof(ShavePerfCounters));

    if (isController) {
        workState = &init->rtState.workerRunState[0];
        preMapping = init->rtState.preMapping;
        // Reset the worker states
        memset(workState, (uint8_t)RtWorkerState::IDLE, TOTAL_NUM_SHAVES * sizeof(RtWorkerState));
        memset(preMapping, (int8_t)(INVALID_SHAVE_ID), TOTAL_NUM_SHAVES * sizeof(int8_t));
    } else {
        workState = init->rtState.workerRunState + scGetShaveNumber();
        preMapping = init->rtState.preMapping + scGetShaveNumber();
    }

    commonState = &init->rtState;
    svuMutexId = init->svuMutexId;
    execContext = init->rtState.execContext;
    dbgStateInit(init);

    // For accessing perfCounters from workers.
    init->rtState.svuNNRtRef[scGetShaveNumber()] = this;
}

SVUNNRuntime::~SVUNNRuntime() {}

void SVUNNRuntime::runRT() {
    do {
        sle = getNextSLE();
        dbgState(RtDbgState::ReceivedSLE, sle);

        if (sle == nullptr) {
            dbgState(RtDbgState::RuntimeTerminate, (uint32_t)isController);
            if (isController) handlePoolResize();
            break;
        }

        lyr = sle->layer_;
        dbgState(RtDbgState::DequeuedLayer, sle, lyr);

        if (isController && lyr->pre) {
            handlePoolResize();

            resetPerformanceCounters(true);

            aba = sle->abs_addr_;
            dbgState(RtDbgState::AbsoluteAddresses, &aba, lyr->params.paramsID);
            dbgState(RtDbgState::ExecPreamble, lyr->pre);

#ifdef CONFIG_NN_SVU_RUNTIME_DUMP_TENSORS
            dbgState(RtDbgState::SaveInputs, &aba, &lyr->params);
#endif
            resetStorageCounters();

            // To be executed in several stages resMgr->preNumOfStages should be set
            // in the first start of preamble function
            earlyStopRequested = false;
            for (int stage = 0; stage < preNumOfStages; stage++) {
                curStage = stage;
                (lyr->pre)(lyr->params.layerParams, this);
                accumulateControllerCounters();
                reportStackUsage();

                if (lyr->kernelEntry) {
                    if (earlyStopRequested) break;

                    dbgState(RtDbgState::KernelEntry, lyr->kernelEntry);
                    CmxFifoHandle handle;
                    handle.type = FIFO_TYPE;

                    int lastPreResIndex =
                        (commonState->preResources[0].shaveID == commonState->totResources[0].shaveID);
                    for (int i = commonState->preResCount - 1; i >= lastPreResIndex; i--) {
                        handle.fifo_id = commonState->preResources[i].shaveID;
                        workState[handle.fifo_id] = RtWorkerState::IN_PROGRESS;
                        CmxFifoWrite(&handle, (uint32_t *)(&sle), sizeof(sle));
                    }

                    if (lastPreResIndex) {
                        // the last resource is the controller
                        resetPerformanceCounters();
                        if (lyr->isMultistage) {
                            (lyr->kernelEntry)(reinterpret_cast<void *>(&(commonState->preResources[0])));
                        } else {
                            (lyr->kernelEntry)(reinterpret_cast<void *>(sParam));
                        }
                        finishCounters();
                        reportStackUsage();
                    }

                    waitForWorkers();
                }

                accumulateWorkerCounters();
#ifdef SVU_STACK_USAGE_INSTRUMENTATION
                const int32_t stackRemaining = *stackHighWater - *stackMaxExtent;
                const uint32_t percentUsed = (uint32_t)(((stackSize - stackRemaining) / (float)stackSize) * 100);
                if (stackRemaining < 0) {
                    dbgState(RtDbgState::StackCrash, percentUsed, (uint32_t)(stackRemaining * -1));
                } else if (stackRemaining <= stackSize * 0.05) {
                    dbgState(RtDbgState::StackEventExceeds, percentUsed, (uint32_t)stackRemaining);
                }
#endif

#ifdef CONFIG_NN_SVU_RUNTIME_DUMP_TENSORS
                dbgState(RtDbgState::SaveOutputs, &aba, &lyr->params);
#endif
            }
            resetPerformanceCounters();
            if (lyr->exeClean) {
                (lyr->exeClean)(reinterpret_cast<const SoftParams *>(sParam)->layerParams, this);
            }
            accumulateControllerCounters();

            reportCounters();
            // TODO: Extend the functionality of flushIfNeeded
            // to provide the flash between stages if needed.
            // For that modify layerRequiresCacheFlushOnCompletion field to enum or bitmap
            // to set/check several flags independently
            flushIfNeeded();
            sendComplete();
        } else {
#ifndef NO_SVU_NN_CONTROLLER
            *workState = RtWorkerState::IN_PROGRESS;
#endif
            dbgState(RtDbgState::KernelEntry, lyr->kernelEntry);
            resetPerformanceCounters();
            if (lyr->isMultistage) {
                (lyr->kernelEntry)(reinterpret_cast<void *>(&(commonState->preResources[*preMapping])));
            } else {
                (lyr->kernelEntry)(reinterpret_cast<void *>(sParam));
            }
            finishCounters();
            reportStackUsage();
            sendComplete();
        }
    } while (true);
}

/*
 * ************* ShaveResourceManager Impl *************
 */

const ShaveResource *SVUNNRuntime::requestShaves(unsigned int &numShaves) {
    assert(numShaves && "Requested shaves must be non-zero");

    // seperate preResources set with INVALID_SHAVE_ID is probably overkill
    numShaves = __builtin_shave_cmu_min_u32_rr_uint(numShaves, commonState->resCount);
    numShaves = __builtin_shave_cmu_min_u32_rr_uint(numShaves, lyr->maxShaves);
    commonState->preResCount = numShaves;

    const uint32_t preSize = numShaves * sizeof(ShaveResource);

    memcpy_s(commonState->preResources, preSize, (commonState->totResources + (commonState->resCount - numShaves)), preSize);

    for (unsigned int i = 0; i < commonState->resCount; i++) {
        preMapping[i] = (i < commonState->resCount - numShaves) ?
                INVALID_SHAVE_ID :
                i - (commonState->resCount - numShaves);
    }

    for (unsigned int i = commonState->preResCount; i < TOTAL_NUM_SHAVES; i++) {
        commonState->preResources[i].shaveID = INVALID_SHAVE_ID;
        commonState->preResources[i].resMgr = nullptr;
    }

    for (unsigned int i = 0; i < commonState->preResCount; i++) {
        commonState->preResources[i].resMgr = this;
    }

    return commonState->preResources;
}

// Return the shaves previously assigned through requestShaves()
const ShaveResource *SVUNNRuntime::getAssignedShaves(unsigned int &numShaves) const {
    numShaves = commonState->preResCount;
    return commonState->preResources;
}

unsigned int SVUNNRuntime::getMaximumShaves() const {
    return NN_MAX_UPA_SHAVE_POOL_SIZE;
}

/* FIXME: this cannot function as it did from LRT
 * To dynamically load ELFs we'll need to:
 * - make room for a "guest" data section in our data section
 * - copy to that here
 * - also make sure that it's windowed correctly
 * - (also the guest's code needs proper windowing)
 */
void SVUNNRuntime::setupShaveForKernel(const ShaveResource &res) { (void)res; }

void SVUNNRuntime::updateLayerParams(const ShaveResource &shave, LayerParams *lp) const {
    if (!lp)
        return;
    lp->availableCmxBytes = SHAVE_LIB_DATA_SIZE;
    lp->cmxData = (uint8_t *)getDataAddr(shave);
}

const unsigned char *SVUNNRuntime::getAbsoluteInputAddr(unsigned int idx) const { return aba.inputs_[idx]; }
unsigned char *SVUNNRuntime::getAbsoluteOutputAddr(unsigned int idx) const { return aba.outputs_[idx]; }

uint32_t SVUNNRuntime::getParamAddr(const ShaveResource &shave) const {
    // uint32_t paramWinAddr = KERNEL_DATA(sParam);
    uint32_t paramWinAddr = reinterpret_cast<uint32_t>(sParam);
    uint32_t winMask = ~(uint32_t)DATA_WINDOW_MASK;

    return shave.cmxSliceAddr() | (paramWinAddr & winMask);
}

uint32_t SVUNNRuntime::getDataAddr(const ShaveResource &shave) const {
    // uint32_t paramWinAddr = KERNEL_DATA(sData);
    uint32_t paramWinAddr = reinterpret_cast<uint32_t>(sData);
    uint32_t winMask = ~(uint32_t)DATA_WINDOW_MASK;

    return shave.cmxSliceAddr() | (paramWinAddr & winMask);
}

char *SVUNNRuntime::getExecContextBaseAddr() { return execContext; }

char *SVUNNRuntime::getRawExecContext(size_t size) {
    assert(size < SHAVE_LIB_EXEC_CONTEXT_SIZE && "Requested ExecContext may not exceed SHAVE_LIB_EXEC_CONTEXT_SIZE");
    return reinterpret_cast<char *>(getExecContextBaseAddr());
}

void SVUNNRuntime::requestCacheFlushForLayer(void) {
    lrtMessageSend(SVU_NN_TAG_FIELD_L2C_DATA_FLUSH, svuMutexId, &commonState->lrtInterruptServiced);
}

void SVUNNRuntime::invalidateL1L2InstCacheForAssignedShaves() const {
    lrtMessageSend(SVU_NN_TAG_FIELD_L2C_INSTR_FLUSH, svuMutexId, &commonState->lrtInterruptServiced);
}

void SVUNNRuntime::flushL1L2DataCacheForAssignedShaves() const {
    lrtMessageSend(SVU_NN_TAG_FIELD_L2C_DATA_FLUSH, svuMutexId, &commonState->lrtInterruptServiced);
}

/*
 * ********************* Private ***********************
 */

SoftLayerExec *SVUNNRuntime::getNextSLE() {
    uint32_t bytes;
    SoftLayerExec *ret;

    // Block and monitor CMX FIFO here.
    CmxFifoRead((uint32_t *)&ret, sizeof(ret), &bytes);

    return ret;
}

inline void SVUNNRuntime::handlePoolResize() {
    if (commonState->updateTotResources) {
        assert(commonState->newTotResources[0].shaveID == commonState->totResources[0].shaveID &&
               "Reassigning SVUNNRt controller not supported");

        const uint32_t totSize = commonState->newResCount * sizeof(ShaveResource);

        memcpy_s(commonState->totResources, totSize, commonState->newTotResources, totSize);

        for (int i = commonState->newResCount; i < TOTAL_NUM_SHAVES; i++) {
            commonState->totResources[i].shaveID = INVALID_SHAVE_ID;
        }

        commonState->resCount = commonState->newResCount;
        commonState->updateTotResources = false;
    }
}

void SVUNNRuntime::waitForWorkers() {
    static_assert(sizeof(svuNNRtCommonState::workerRunState) / sizeof(RtWorkerState) == 16,
                  "SVUNNRT state array needs to fit in a  uchar16");
    // FIXME: CFG_SHAVE_PRG_DATA_WIDTH may not mean what I think it means
    static_assert(sizeof(svuNNRtCommonState::workerRunState) == CFG_SHAVE_PRG_DATA_WIDTH / 8,
                  "SVUNNRT state array size expected to match vector unit word length");
    assert(isController && "SVU NN RT RT worker internal error");

    volatile uchar16 workers = *reinterpret_cast<uchar16 *>(workState);

    //  note that IDLE = 0
    while (__builtin_shave_sau_orx_x8_r(workers)) {
        workers = *reinterpret_cast<uchar16 *>(workState);
        // do something useful?
    };
}

void SVUNNRuntime::flushIfNeeded() {
    if (lyr->params.layerParams->layerRequiresCacheFlushOnCompletion) {
        requestCacheFlushForLayer();
    }
}

void SVUNNRuntime::sendComplete() {
#ifndef NO_SVU_NN_CONTROLLER
    if (isController) {
        // Set SoftLayer completion flag
        sle->completed_ = true;

        dbgState(RtDbgState::FinishedSLE, sle, lyr);
    } else {
        *workState = RtWorkerState::IDLE;
    }
#endif
}

#ifdef SVU_STACK_USAGE_INSTRUMENTATION
inline uint32_t SVUNNRuntime::getStackSizeConsumed(void) { return stackStartAddr - *stackHighWater; }
#endif

void SVUNNRuntime::reportStackUsage() {
#ifdef SVU_STACK_USAGE_INSTRUMENTATION
    dbgState(RtDbgState::StackInstrumentation, getStackSizeConsumed(), stackSize);
#endif
}

/*
 * ******************** Perf Stuff *********************
 */

inline void SVUNNRuntime::resetStorageCounters() {
    if (sle->counters_) {
        commonState->performanceCounters.instrs = 0;
        commonState->performanceCounters.cycles = 0;
        commonState->performanceCounters.stalls = 0;
        commonState->performanceCounters.branches = 0;
    }
}

inline void SVUNNRuntime::resetPerformanceCounters(bool resetCycles) {
    if (sle->counters_) {
        uint32_t svuBase = SHAVE_BASE_ADDR(scGetShaveNumber()) + SLC_OFFSET_SVU;

        SET_REG_WORD(svuBase + SVU_PC0, 0);
        if(resetCycles)
            SET_REG_WORD(svuBase + SVU_PC1, 0);
        SET_REG_WORD(svuBase + SVU_PC2, 0);
        SET_REG_WORD(svuBase + SVU_PC3, 0);
    }
}

inline void SVUNNRuntime::finishCounters() {
    if (sle->counters_) {
        uint32_t svuBase = SHAVE_BASE_ADDR(scGetShaveNumber()) + SLC_OFFSET_SVU;

        perfCounters.instrs = GET_REG_WORD_VAL(svuBase + SVU_PC0);
        perfCounters.cycles = GET_REG_WORD_VAL(svuBase + SVU_PC1);
        perfCounters.stalls = GET_REG_WORD_VAL(svuBase + SVU_PC2);
        perfCounters.branches = GET_REG_WORD_VAL(svuBase + SVU_PC3);
    }
}

inline void SVUNNRuntime::accumulateControllerCounters() {
    if (sle->counters_) {
        uint32_t svuBase = SHAVE_BASE_ADDR(scGetShaveNumber()) + SLC_OFFSET_SVU;

        commonState->performanceCounters.instrs += GET_REG_WORD_VAL(svuBase + SVU_PC0);
        commonState->performanceCounters.cycles += GET_REG_WORD_VAL(svuBase + SVU_PC1);
        commonState->performanceCounters.stalls += GET_REG_WORD_VAL(svuBase + SVU_PC2);
        commonState->performanceCounters.branches += GET_REG_WORD_VAL(svuBase + SVU_PC3);
    }
}

inline void SVUNNRuntime::accumulateWorkerCounters() {
    if (sle->counters_) {
        for (int i = commonState->preResCount - 1; i >= 0; i--) {
            uint32_t shaveID = commonState->preResources[i].shaveID;
            if (SVUNNRuntime* svunnrt = static_cast<SVUNNRuntime*> (commonState->svuNNRtRef[shaveID])) {
                commonState->performanceCounters.instrs += svunnrt->perfCounters.instrs;
                commonState->performanceCounters.stalls += svunnrt->perfCounters.stalls;
                commonState->performanceCounters.branches += svunnrt->perfCounters.branches;
            }
        }
    }
}

inline void SVUNNRuntime::reportCounters() {
    if (sle->counters_)
    {
        ShDrvCmxDmaTransactionHnd handle;
        ShDrvCmxDmaTransaction transaction;

        auto result = ShDrvCmxDmaCreateTransaction(&handle, &transaction,
            reinterpret_cast<uint8_t *>(&commonState->performanceCounters), reinterpret_cast<uint8_t *>(sle->counters_), sizeof(ShavePerfCounters));

        if (result == MYR_DRV_SUCCESS)
        {
            result = ShDrvCmxDmaStartTransfer(&handle);

            if (result == MYR_DRV_SUCCESS)
            {
                result = ShDrvCmxDmaWaitTransaction(&handle);

                if (result != MYR_DRV_SUCCESS)
                    nnLog(MVLOG_ERROR, "ShDrvCmxDmaWaitTransaction failed with error %u", result);
            }
            else
                nnLog(MVLOG_ERROR, "ShDrvCmxDmaStartTransfer failed with error %u", result);
        }
        else
            nnLog(MVLOG_ERROR, "ShDrvCmxDmaCreateTransaction failed with error %u", result);
    }
}

} // namespace shave_lib
} // namespace nn
