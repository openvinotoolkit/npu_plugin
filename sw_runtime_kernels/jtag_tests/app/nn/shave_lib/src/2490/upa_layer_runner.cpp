/*
* {% copyright %}
*/
#include "upa_layer_runner.h"
#include "HglCmxFifo.h"
#include "layer_loader.h"
#include "layers/svuSLKernels_EP.h"
#include "mvMacros.h"
#include "sw_layer.h"
#include "sw_layer_params.h"
#include "upa_fifo.h"
#include "CmxFifo.h"

#include <cassert>
#include <mvLog.h>
#include <nn_cache.h>
#include <pipePrintInit.h>
#include <type_traits>
#include <types.h>

#include "layers/svuSLKernels_EP.h"

// For windows...
extern "C" {
#include <DrvRegUtils.h>
#include <DrvSvuControl.h>
#include <DrvSvuCounters.h>
#include <DrvSvuL2CacheDefines.h>
#include <OsDrvCpr.h>
#include <OsDrvSvu.h>
#include <lDrvSvuDefines.h> // For SHAVE_BASE_ADDR
}

#ifdef CONFIG_USE_COMPONENT_VCSHOOKS
#    include <VcsHooksApi.h>
#else
#    define saveMemoryToFile(...)
#endif

#define NN_ALLOCATION_TIMEOUT_MS 1000

#define PACK_MSG(sh, tag) ((sh << 8) | tag)
#define UNPACK_SH(msg) ((msg >> 8) & 0xFF)
#define UNPACK_TAG(msg) (msg & 0xFF)

#define DEBUG_TRACE(state)                                                                                             \
    case RtDbgState::state:                                                                                            \
        nnLog(MVLOG_INFO, "SVU NN RT Trace Event: '" #state "' Sh: %d Value: %x ", shNum, dbgValue);                   \
        break;

#define DEBUG_TRACE2(state, MSG)                                                                                       \
    case RtDbgState::state:                                                                                            \
        nnLog(MVLOG_INFO, "SVU NN RT Trace Event: '" #state "' Sh: %d " MSG, shNum, dbgValue, dbgValue2);              \
        break;

namespace nn {
namespace shave_lib {

// Used for Shave interrupt callback
static volatile uint32_t svuTag;

// TODO: move this somewhere else ?
static svuNNRtInit __attribute__((section(".cmx_direct.data"))) svuRtInit;

// TODO: clarify appropriate msgqueue length...
nn::util::MessageQueue<uint16_t> UPALayerRunner::upaMsgQueue(SHAVE_LIB_SVU_LRT_MESSAGE_QUEUE_SIZE);

inline OsDrvCprDevice getShaveOsDrvCprDeviceID(uint32_t shaveID) {
    return static_cast<OsDrvCprDevice>(static_cast<uint32_t>(OS_DRV_CPR_DEV_UPA_SHAVE_0) + shaveID);
}

void UPALayerRunner::voidIrqHandlerExitHook(void *source) {
    uint8_t shaveNumber = (uint8_t)((uint32_t)source - LRT_IRQ_SHAVE);
    svuTag = DrvSvuGetSwiTag(shaveNumber);
    DrvSvuClearPendingIrqs(shaveNumber);
    UPALayerRunner::upaMsgQueue.push(PACK_MSG(shaveNumber, svuTag));
}

UPALayerRunner::UPALayerRunner() : resources(0), resourcesHandles(0), keepRunning(true) {
    nnLog(MVLOG_DEBUG, "ctor start");
    allocateResources();

    // Create thread for managing UPA cache flushing
    UPALayerRunner::upaThread.set_priority(util::Thread::priority() - 1); // TODO: check priority of this thread
    UPALayerRunner::upaThread.create(rtems_build_name('U', 'P', 'A', '0'));
    UPALayerRunner::upaThread.start(&monitorMsgQueue, this);

    // Start shaves from controller first (smallest to largest) because controller acquires the mutex
    setupAndStartResources(0);

    nnLog(MVLOG_DEBUG, "ctor end");
}

UPALayerRunner::~UPALayerRunner() {
    nnLog(MVLOG_DEBUG, "dtor start: Shutting down SVU NN runtime");

    uint32_t runningShaves;
    for (auto &res : resources) {
        nn::util::upaFifoWrite(res.shaveID, (uint32_t)NULL);
        // svuWaitShave will flush and invalidate L2 and L1 data cache.
        svuWaitShave(&upaShaveHandle, res.shaveID, RTEMS_NO_TIMEOUT, &runningShaves);
    }

    keepRunning = false;
    upaMsgQueue.push(0);
    if (upaThread.joinable())
        upaThread.join();

    freeResources();
    nnLog(MVLOG_DEBUG, "dtor end: UPA resources released");
}

void UPALayerRunner::monitorMsgQueue(UPALayerRunner *upa) {
    nnLog(MVLOG_INFO, "Started UPA MSG queue thread");
    uint16_t svuMsg;
    uint8_t shNum;
    uint8_t tag;
#ifdef DEBUG_NN_SVU_RUNTIME
    char fname[NN_CACHE_LINE_LENGTH] __attribute__((aligned(64)));
#endif

    for (upa->upaMsgQueue.pop(svuMsg); upa->keepRunning; upa->upaMsgQueue.pop(svuMsg)) {
        OsDrvMutexLock(&upa->mtxHandle);
#ifdef DEBUG_NN_SVU_RUNTIME
        svuRtInit.rtState.irq_rx++;
#endif
        shNum = UNPACK_SH(svuMsg);
        tag = UNPACK_TAG(svuMsg);

        switch (tag) {
        case SVU_NN_TAG_FIELD_SHUTDOWN: {
            nnLog(MVLOG_DEBUG, "UPA stopped");
            break;
        }
        case SVU_NN_TAG_FIELD_PREAMBLE: {
            nnLog(MVLOG_WARN, "Preamble requested from UPA");
            // TODO: Add preamble run here.
            break;
        }
        // Flush the Shave L1/L2 data cache
        case SVU_NN_TAG_FIELD_L2C_DATA_FLUSH: {
            nnLog(MVLOG_INFO, "L2 data cache flush requested from UPA Shave");
            upa->flushShaveL2DataCache();
            nnLog(MVLOG_INFO, "L2 data cache flush of Shave completed");
            break;
        }
        // Flush the Shave L1/L2 instruction cache
        case SVU_NN_TAG_FIELD_L2C_INSTR_FLUSH: {
            nnLog(MVLOG_INFO, "L2 instruction cache flush requested from UPA Shave");
            upa->flushShaveL2InstructionCache();
            nnLog(MVLOG_INFO, "L2 instruction cache flush of Shave completed");
            break;
        }
        case SVU_NN_TAG_FIELD_PRINT_RT_TRACE: {
#ifdef DEBUG_NN_SVU_RUNTIME
            static uint32_t layerCnt = 0;
            RtDbgState dbgState = svuRtInit.rtState.dbgState;
            uint32_t dbgValue = svuRtInit.rtState.dbgValue;
            uint32_t dbgValue2 = svuRtInit.rtState.dbgValue2;

            switch (dbgState) {
                DEBUG_TRACE(RuntimeInitStarting)
                DEBUG_TRACE(RuntimeInitComplete)
                DEBUG_TRACE(RuntimeStarting)
                DEBUG_TRACE(ReceivedSLE)
                DEBUG_TRACE2(DequeuedLayer, "SLE %x Lyr %x")
                DEBUG_TRACE(ExecPreamble)
                DEBUG_TRACE(KernelEntry)
                DEBUG_TRACE2(FinishedSLE, "SLE: %x Lyr: %x")
                DEBUG_TRACE(RuntimeComplete)
                DEBUG_TRACE(RuntimeTerminate)
                DEBUG_TRACE2(StackInstrumentation, "Used: %d Available: %d")
                DEBUG_TRACE2(StackEventExceeds, "Stack Used: %d%% Remains: %d B")
                DEBUG_TRACE2(StackCrash, "Stack Used: %d%% Wrote: %d B into data")
                DEBUG_TRACE(Marker02)
                DEBUG_TRACE(Marker03)
                DEBUG_TRACE(Marker04)
                DEBUG_TRACE(Marker05)
            case RtDbgState::AbsoluteAddresses: {
                AbsoluteAddresses *addrs = (AbsoluteAddresses *)dbgValue;
                cache::invalidate(addrs, sizeof(AbsoluteAddresses));
                nnLog(MVLOG_INFO, "Tensor addresses for %s:", LayerParser::getParamName(dbgValue2));
                for (int i = 0; i < MAX_INPUT_TENSORS; i++)
                    if (addrs->inputs_[i] != nullptr)
                        nnLog(MVLOG_INFO, "I %d: %p", i, addrs->inputs_[i]);

                for (int o = 0; o < MAX_OUTPUT_TENSORS; o++)
                    if (addrs->outputs_[o] != nullptr)
                        nnLog(MVLOG_INFO, "O %d: %p", o, addrs->outputs_[o]);
                break;
            }
            case RtDbgState::SaveInputs: {
                AbsoluteAddresses *addrs = (AbsoluteAddresses *)dbgValue;
                SoftParams *params = (SoftParams *)dbgValue2;
                cache::invalidate(addrs, sizeof(AbsoluteAddresses));
                cache::invalidate(params, sizeof(SoftParams));
                nnLog(MVLOG_INFO, "Layer %d: %s\n", layerCnt, LayerParser::getParamName(params->paramsID));
                for (int i = 0; i < MAX_INPUT_TENSORS; i++)
                    if (addrs->inputs_[i] != nullptr) {
                        snprintf(fname, sizeof(fname), "input-%d-%d-dump.bin", layerCnt, i);
                        cache::flush(fname);
                        cache::invalidate(addrs->inputs_[i], params->inputs[i].getDataSize());
                        saveMemoryToFile((uint32_t)addrs->inputs_[i], params->inputs[i].getDataSize(), fname);
                    }

                // Incremented in output saving
                // layerCnt++;
                break;
            }
            case RtDbgState::SaveOutputs: {
                AbsoluteAddresses *addrs = (AbsoluteAddresses *)dbgValue;
                SoftParams *params = (SoftParams *)dbgValue2;
                // These have already been invalidated during SaveInputs
                for (int i = 0; i < MAX_OUTPUT_TENSORS; i++)
                    if (addrs->outputs_[i] != nullptr) {
                        snprintf(fname, sizeof(fname), "output-%d-%d-dump.bin", layerCnt, i);
                        cache::flush(fname);
                        cache::invalidate(addrs->outputs_[i], params->outputs[i].getDataSize());
                        saveMemoryToFile((uint32_t)addrs->outputs_[i], params->outputs[i].getDataSize(), fname);
                    }

                layerCnt++;
                break;
            }
            default:
                nnLog(MVLOG_ERROR, "SVU NN RT Tracing error (Sh = %d dbgState = %d, dbgValue = %d) ", shNum,
                      (uint32_t)dbgState, dbgValue);
                break;
            }

#else
            nnLog(MVLOG_ERROR, "SVU NN RT Tracing should not be enabled in shave bianary");
#endif
            break;
        }
        default: nnLog(MVLOG_ERROR, "SVN NN RT error: invalid tag sent from Shave %d (%x)", shNum, tag); break;
        }

        svuRtInit.rtState.lrtInterruptServiced = true;
        OsDrvMutexUnlock(&upa->mtxHandle);
    }
}

void UPALayerRunner::setupShave(const ShaveResource &res, const SoftKernel &kernel) {
    DmaAlLeon dma;
#ifdef DEBUG_NN_SVU_RUNTIME
    nnLog(MVLOG_INFO, "SVU NN RT ShaveID: %d Kernel: %p", res.shaveID, &kernel);
    nnLog(MVLOG_INFO, "SVU NN RT Code: %p size: %ld", kernel.codeBaseAddress, kernel.codeSize);
    nnLog(MVLOG_INFO, "SVU NN RT Data: %p size: %ld", kernel.dataBaseAddress, kernel.dataSize);
    nnLog(MVLOG_INFO, "SVU NN RT Entry: %p", kernel.kernelEntry);
#endif

    assert(SHAVE_LIB_STACK_MAX_SIZE + kernel.dataSize <= CMX_SLICE_SIZE && "Exceeded SVU NN RT CMX slice total size");

    // Copy data area to CMX
    dma.start(kernel.dataBaseAddress, (void *)res.cmxSliceAddr(), kernel.dataSize);
    dma.wait();

    DrvSvuSetShaveWindow(res.shaveID, e_WINDOW_A_NO, res.cmxSliceAddr());
    DrvSvuSetShaveWindow(res.shaveID, e_WINDOW_B_NO, (uint32_t)kernel.codeBaseAddress);

    // Align the stack ptr to be 16 byte aligned. This is so DMA descriptors allocated
    // on the stack are also 16 byte aligned, since the compiler does not adjust the stack
    // pointer on entry to a function which allocates DMA descriptors (through the abstraction)
    // on the stack.  This alignment is maintained across function calls within the shave
    shv_ExecutionContext_t *shvContext = &upaShaveHandle.shaveAllocations[res.shaveID].shaveContext;

    uint32_t stackAddr = (uint32_t)shvContext->stack_address;
    shvContext->stack_address = (void *)(stackAddr & ~0xFu);
    shvContext->stack_size = CMX_SLICE_SIZE - kernel.dataSize - (stackAddr - (uint32_t)shvContext->stack_address);

    // By default Non-Windowed Data accesses go to the bypass partition
    // All of our tensors are not windowed, so pick the data L2C partition (WIN_A) from the WIN_CPC register
    // and use it for non-windowed as well.
    uint32_t win_cpc = GET_REG_WORD_VAL((uint32_t)SHAVE_BASE_ADDR(res.shaveID) + SLC_OFFSET_TOP + SLC_TOP_WIN_CPC);
    uint32_t dataPart = win_cpc & DRV_SHAVE_WIN_CPC_WIN_L2C_PART_ID;        // WIN_A
    uint32_t instPart = (win_cpc >> 5) & DRV_SHAVE_WIN_CPC_WIN_L2C_PART_ID; // WIN_B
    DrvShaveSetNonWinL2CachePart(res.shaveID, dataPart, DRV_SHAVE_CACHE_DATA);

    // Custom layers (OCL, eDSL, etc) have code in WIN_C and data in WIN_D (backward from normal convention)
    // Swap the C and D WIN partitions (in normal config C is data and D is code)
    DrvShaveSetWinL2CachePart(res.shaveID, e_WINDOW_C_NO, instPart);
    DrvShaveSetWinL2CachePart(res.shaveID, e_WINDOW_D_NO, dataPart);

    shvContext->heap_address = svuRtInit.heapAddress;
    shvContext->heap_size = svuRtInit.heapSize;

    rtems_status_code sc = svuRegisterCustomVoidIrqHook(&upaShaveHandle, res.shaveID, voidIrqHandlerExitHook);
    UNUSED(sc);
    // TODO: Check status code here...
}

bool UPALayerRunner::hasResources() const {
    return resources.size() > 0;

    // TODO: Retry allocation somehow. There is a thread safety issue between multiple
    // inferences executing in parallel from MvNCI side
    // Look at putting check in Executor so we don't even try to process and fail earlier
    // May have problems with instantiation order (IRS and USDispatcher see comments on PR 1268)
}

void UPALayerRunner::flushShaveL2DataCache(void) {
    for (auto res : resources) {
        svuFlushInvalidateShaveL1L2CacheData(&upaShaveHandle, OS_DRV_SHAVE_L2C_FLUSH_INV, res.shaveID);
    }
}

void UPALayerRunner::flushShaveL2InstructionCache(void) {
    for (auto res : resources) {
        svuInvalidateShaveL1L2CacheInstr(&upaShaveHandle, res.shaveID);
    }
}

uint32_t UPALayerRunner::getControllerShaveID() const {
    return resources.empty() ? INVALID_SHAVE_ID : resources.front().shaveID;
}

void UPALayerRunner::allocateResources() {
    rtems_status_code sc;

    nnLog(MVLOG_INFO, "Allocating SHAVE NN Resources");

    memset(&upaShaveHandle, 0, sizeof(upaShaveHandle));
    memset(&svuRtInit, 0, sizeof(svuRtInit));

    // Allocation of a free mutex identified by info: 500.
    // Info can be any uint32_t number.
    // The actual HW mutex id will be available in hndl->id
    // after a successful allocation
    sc = OsDrvMutexAllocate(500, &mtxHandle);
    if (sc != RTEMS_SUCCESSFUL) {
        assert(0 && "Failed to allocate mutex");
    }

    sc = svuInit(&upaShaveHandle, NULL);
    if (sc != RTEMS_SUCCESSFUL) {
        assert(0 && "Failed to allocate sDrvSvuHandler");
    }

#ifdef CONFIG_NN_L2C_PAGE_TABLE
    // Configure L2C_PAGE_TABLE for dKMB when virtual addressing is enabled
    uint8_t dramPageTablePrefix = CONFIG_NN_L2C_PAGE_TABLE;
    sc = svuCfgCustomL2cDdrPageTablePrefix(&upaShaveHandle, dramPageTablePrefix);
    if (sc == RTEMS_SUCCESSFUL)
        nnLog(MVLOG_INFO, "Using L2C_PAGE_TABLE value 0x%x", dramPageTablePrefix);
    else
        nnLog(MVLOG_ERROR, "Cannot configure L2C_PAGE_TABLE, error %u", sc);
#endif

    // Initalize the CMX FIFO so that UPA Shaves know
    // what fifo they should monitor. See CmxFifo.h for
    // more information
    mvReturn cmx_status = CmxFifoInitialize();
    if (cmx_status != MV_RET_SUCCESS && cmx_status != MV_RET_ALREADY_INITIALIZED) {
        nnLog(MVLOG_ERROR, "CmxFifoInitialize=%d", cmx_status);
        // the cmx fifo has not been initalized and so shaves do not know
        // what fifos to monitor
        assert(0 && "CMX FIFOs could not be initalized");
    }

    allocateShaveResources();

    // This case will get checked when trying to execute an inference as well
    if (resources.size() == 0)
        nnLog(MVLOG_ERROR, "No Shave resources available!");

    std::fill(std::begin(svuRtInit.rtState.totResources), std::end(svuRtInit.rtState.totResources), ShaveResource());
    std::fill(std::begin(svuRtInit.rtState.preResources), std::end(svuRtInit.rtState.preResources), ShaveResource());
    std::fill(std::begin(svuRtInit.rtState.newTotResources), std::end(svuRtInit.rtState.newTotResources), ShaveResource());

    {
        svuRtInit.rtState.resCount = resources.size();

        for (unsigned int i = 0; i < resources.size(); ++i)
        {
            svuRtInit.rtState.totResources[i].shaveID = resources[i].shaveID;
            svuRtInit.rtState.preResources[i].shaveID = svuRtInit.rtState.totResources[i].shaveID;
        }

        svuRtInit.svuMutexId = mtxHandle.id;

        svuRtInit.heapAddress = nullptr;
        svuRtInit.heapSize = 0;
    }

    cache::flush(svuRtInit);
}

rtems_status_code UPALayerRunner::allocateShaveResource() {
    ResMgrRequest req;
    ResMgrHandler handle;
    req.id.index = RESMGR_REQ_INDEX_ANY;
    req.type = RESMGR_UPA_SHAVE;
    req.wait = true;

    auto ret = ResMgrAllocate(&handle, &req, NN_ALLOCATION_TIMEOUT_MS);

    if (ret == RTEMS_SUCCESSFUL) {
        ShaveResource res;
        res.shaveID = handle.index;

        if(res.shaveID > MAX_SHAVE_ID)
            return RTEMS_UNSATISFIED;

        // Workaround for issue in some Jenkins tests where a shave is still powered on when we call svuOpenShave
        // svuOpenShave fails trying to power it on again.  Can be removed when the driver does not treat this
        // condition as fatal
        {
            bool isEnabled = false;
            OsDrvCprDeviceConfig shvConfig;

            shvConfig.device =
                static_cast<OsDrvCprDevice>(static_cast<uint32_t>(OS_DRV_CPR_DEV_UPA_SHAVE_0) + res.shaveID);
            shvConfig.action = OS_DRV_CPR_DEV_DISABLE;

            OsDrvCprIsDeviceEnabled(shvConfig.device, &isEnabled);

            if (isEnabled) {
                nnLog(MVLOG_INFO, "Setting UPA shave %d to DISABLED state", res.shaveID);
                OsDrvCprSysDeviceAction(&shvConfig);
            }
        }

#ifndef CONFIG_OS_DRV_SVU_ENABLE_CLOCK_CONTROL
        OsDrvCprDeviceConfig shvConfig;
        shvConfig.device = static_cast<OsDrvCprDevice>(static_cast<uint32_t>(OS_DRV_CPR_DEV_UPA_SHAVE_0) + res.shaveID);
        shvConfig.action = OS_DRV_CPR_DEV_ENABLE;
        OsDrvCprSysDeviceAction(&shvConfig);
#endif

        rtems_status_code st = svuOpenShave(&upaShaveHandle, res.shaveID, RTEMS_NO_WAIT, RTEMS_NO_TIMEOUT);
        if (st == RTEMS_SUCCESSFUL) {
            DrvSvuEnablePerformanceCounter(res.shaveID, 0, PC_EX_IN_EN);
            DrvSvuEnablePerformanceCounter(res.shaveID, 1, PC_CLK_CYC_EN);
            DrvSvuEnablePerformanceCounter(res.shaveID, 2, PC_DEFAULT);
            DrvSvuEnablePerformanceCounter(res.shaveID, 3, PC_BR_TAKEN_EN);

            resources.push_back(res);
            resourcesHandles.push_back(handle);
        } else { // if error:
            nnLog(MVLOG_ERROR, "Error opening SHAVE %d st=%d", res.shaveID, st);
            return ret;
        }
    } else {
        return ret;
    }

    return ret;
}

void UPALayerRunner::allocateShaveResources() {
    assert(resourcesHandles.size() == 0 && "Internal Error: Reallocation of SVUNNRT resources");
    resources.reserve(TOTAL_NUM_SHAVES);
    auto ret = allocateShaveResource();

    if (ret != RTEMS_SUCCESSFUL) {
        nnLog(MVLOG_ERROR, "Could not allocate SHAVE nr. 0; ret=%d", ret);
    }

}

bool UPALayerRunner::resizeShavePool(unsigned int total_shaves) {
    nnLog(MVLOG_INFO, "resources.size: %u, total_shaves: %u, TOTAL_NUM_SHAVES: %u", resources.size(), total_shaves, resources.capacity());
    if(resources.size() >= total_shaves){
        nnLog(MVLOG_INFO, "%u SHAVEs requested when %u is already allocated. No extra allocation needed.", total_shaves, resources.size());
        return true;
    }
    //We store number of SHAVEs before resizing bcs we need to start it later.
    unsigned int preresize_shaves = resources.size();
    unsigned int extra_shaves = total_shaves - resources.size();

    if(preresize_shaves + extra_shaves > resources.capacity()){
        extra_shaves = TOTAL_NUM_SHAVES - preresize_shaves;
        nnLog(MVLOG_WARN, "Not enough SHAVE resources, reducing to max available %u", extra_shaves);
    }

    while(svuRtInit.rtState.updateTotResources)
        cache::invalidate(svuRtInit.rtState);

    bool ret{ true };

    for (unsigned int i = preresize_shaves; i < (preresize_shaves + extra_shaves); i++) {
        auto rtret = allocateShaveResource();

        if (rtret != RTEMS_SUCCESSFUL) {
            nnLog(MVLOG_WARN, "Could not add SHAVE nr. %d out of %d; ret=%d", i, (preresize_shaves + extra_shaves), rtret);
            ret = false;
        }
    }

    for (unsigned int i = 0; i < resources.capacity(); ++i)
        svuRtInit.rtState.newTotResources[i].shaveID = i < resources.size() ? resources[i].shaveID : INVALID_SHAVE_ID;

    svuRtInit.rtState.newResCount = resources.size();
    svuRtInit.rtState.updateTotResources = true;
    cache::flush(svuRtInit.rtState);

    setupAndStartResources(preresize_shaves);

    nnLog(MVLOG_INFO, "State after appending %u SHAVE resources:", extra_shaves);

    for (uint32_t i = 0; i < svuRtInit.rtState.newResCount; i++){
        nnLog(MVLOG_INFO, "\tshaveID - %u, cmxSliceAddr - %u", svuRtInit.rtState.newTotResources[i].shaveID, svuRtInit.rtState.newTotResources[i].cmxSliceAddr());
    }

    return ret;
}

void UPALayerRunner::setupAndStartResources(unsigned int shaveFrom){
    const auto &kernel = LayerLoader::instance().builtinKernels();

#ifdef DEBUG_NN_SVU_RUNTIME
    nnLog(MVLOG_WARN, "Recording enabled for SVU NN Runtime debug state to CMX address %p ",
          &(svuRtInit.rtState->dbgState));
#endif
    nnLog(MVLOG_INFO, "Starting UPA shaves for NN at entry %p with init %p ",
          reinterpret_cast<unsigned int>(kernel.kernelEntry), &svuRtInit);
    for (unsigned int i = shaveFrom; i < resources.size(); i++) {
        ShaveResource &res = resources[i];
        setupShave(res, kernel);

        nnLog(MVLOG_DEBUG, "Launching shave %d", res.shaveID);
        svuStartShaveCC(&upaShaveHandle, res.shaveID, SVU_NN_KERNEL_ENTRY, RTEMS_NO_WAIT, RTEMS_NO_TIMEOUT, "i",
                        &svuRtInit);
    }
}

void UPALayerRunner::freeResources() {
    nnLog(MVLOG_INFO, "Deallocating SHAVE NN Resources");

    freeShaveResources();
    svuDeInit(&upaShaveHandle);
    OsDrvMutexDeallocate(&mtxHandle);

    nnLog(MVLOG_INFO, "UPA SHAVE Driver De-initialized");
}

void UPALayerRunner::freeShaveResources() {
    assert(resources.size() == resourcesHandles.size());

    if (upaShaveHandle.owner != rtems_task_self()) {
        nnLog(MVLOG_WARN, "Shave ownership changed from %u, to task_self %u", upaShaveHandle.owner, rtems_task_self());
        upaShaveHandle.owner = rtems_task_self();
    }

    for (uint32_t i = 0; i < resources.size(); ++i) {
        uint32_t shvId = resources[i].shaveID;

        auto st = RTEMS_SUCCESSFUL;
#ifndef CONFIG_VALIDATION_APP_ENABLED
        st = svuCloseShave(&upaShaveHandle, shvId); /* close shave will turn power of shave off.*/
#endif

        if (st == RTEMS_SUCCESSFUL) {
#ifndef CONFIG_OS_DRV_SVU_ENABLE_CLOCK_CONTROL
            OsDrvCprDeviceConfig shvConfig;
            shvConfig.device = OS_DRV_CPR_DEV_UPA_SHAVE_0 + shvId;
            shvConfig.action = OS_DRV_CPR_DEV_DISABLE;
            OsDrvCprSysDeviceAction(&shvConfig);
#endif
            nnLog(MVLOG_DEBUG, "Closed UPA SHAVE %d", shvId);

            auto ret = ResMgrRelease(&resourcesHandles[i]);

            if (ret == RTEMS_SUCCESSFUL) {
                nnLog(MVLOG_DEBUG, "Released UPA SHAVE %d", shvId);
            } else
                nnLog(MVLOG_ERROR, "Error releasing SHAVE %d ret=%d", shvId, ret);
        } else
            nnLog(MVLOG_ERROR, "Error closing SHAVE %d st=%d", shvId, st);
    }
}
} // namespace shave_lib
} // namespace nn
