#include <DrvRegUtils.h>
#include <registersVpu.h>
#include <VcsHooksApi.h>
#include <stdio.h>
//#include <swcLeonUtils.h>
//#include <vpuipCapsApi.h>
#include <DrvCprDefines3600.h>
#include <DrvCpr.h>
#include <DrvTimer.h>
#include <DrvSvu.h>
#include <swcShaveLoader.h>
#include <DrvNCEShaveL2Cache.h>

//#include "../runtime/nn_inference.h"
//#include "../runtime/barrier.h"
#include "common_functions.h"
#include "dma.h"
#include "descriptor.h"

#define TARGET_SHAVE (ACT_SHAVE_0)
#ifdef CONFIG_TARGET_SOC_3720
__attribute__((aligned(1024)))
#include "sk.nnActEntry.3010xx.text.xdat"
#endif
extern void*  (shvNN0_act_shave_runtime_shaveMain);

typedef struct {
    uint64_t cmx_control_data_size : 22;
    uint64_t active_barrier_count : 8;
    uint64_t dma_task_count : 15;
    uint64_t dpu_task_count : 15;
    uint64_t using_dynamic_barriers : 1;
    uint64_t unused : 3;
    uint64_t shv_task_count : 15;
    uint64_t terminal_barrier_id : 8;
    uint64_t unused1 : 41;
    uint64_t control_base_address;
    uint64_t dpu_descriptor_start_address;
    uint64_t dma_descriptor_start_address;
    uint64_t shv_descriptor_start_address;
}management_cmx_control;

char printBuffer[256];
void vcsFastPrintf(const char * format, ...)
{
    va_list args;
    va_start(args, format);
    vsnprintf(printBuffer, sizeof(printBuffer), format, args);
    vcsFastPuts(printBuffer);
    va_end(args);
}

void move_descriptors_to_cmx(
        uint8_t* dram_control_ptr,
        uint32_t* dpu_descriptor_offset,
        uint32_t* dma_descriptor_offset,
        uint32_t* shv_descriptor_offset){
    management_cmx_control* ctrl_ptr = (management_cmx_control*)dram_control_ptr;

    const uint32_t dpu_descriptor_size = DPU4_DESCRIPTOR_SIZE * ctrl_ptr->dpu_task_count;
    const uint32_t dma_descriptor_size
            = sizeof(dma_descriptor_common_t) * ctrl_ptr->dma_task_count;

    const uint32_t shv_descriptor_size
            = sizeof(shv_job_header) * ctrl_ptr->shv_task_count;

    *dpu_descriptor_offset = 0U;
    *dma_descriptor_offset = dpu_descriptor_size;
    *shv_descriptor_offset = *dma_descriptor_offset + dma_descriptor_size;

    uint64_t dpu_desc_src_addr = ctrl_ptr->dpu_descriptor_start_address;
    uint64_t dpu_desc_dst_addr = ctrl_ptr->control_base_address;

    retarget_address(&dpu_desc_src_addr);
    retarget_address(&dpu_desc_dst_addr);
    uint8_t* src_ptr = (uint8_t*)dpu_desc_src_addr;
    uint8_t* dst_ptr = (uint8_t*)dpu_desc_dst_addr;
//    dpu_job_descriptor* first_dpu_desc_ptr = (dpu_job_descriptor*)src_ptr;

    printf("INFO: cpu_memcpy 1 %ld \n", dpu_descriptor_size);
    //cpu_memcpy(src_ptr, dst_ptr, dpu_descriptor_size);

    uint64_t dma_desc_src_addr = ctrl_ptr->dma_descriptor_start_address;
    uint64_t dma_desc_dst_addr = ctrl_ptr->control_base_address;
    dma_desc_dst_addr += dpu_descriptor_size;

    vcsFastPuts((char*)"INFO: retarget_address 1 ");
    retarget_address(&dma_desc_src_addr);
    vcsFastPuts((char*)"INFO: retarget_address 2 ");
    retarget_address(&dma_desc_dst_addr);

    src_ptr = (uint8_t*)dma_desc_src_addr;
    dst_ptr = (uint8_t*)dma_desc_dst_addr;



//    dma_descriptor_common_t* first_dma_desc_ptr = (dma_descriptor_common_t*)src_ptr;
    vcsFastPuts((char*)"INFO: cpu_memcpy 2 ");
    //cpu_memcpy(src_ptr, dst_ptr, dma_descriptor_size);

    uint64_t shv_desc_src_addr = ctrl_ptr->shv_descriptor_start_address;
    uint64_t shv_desc_dst_addr = dma_desc_dst_addr;
    shv_desc_dst_addr += dma_descriptor_size;

    vcsFastPuts((char*)"INFO: retarget_address 3 ");
    retarget_address(&shv_desc_src_addr);
    vcsFastPuts((char*)"INFO: retarget_address 4 ");
    retarget_address(&shv_desc_dst_addr);

    src_ptr = (uint8_t*)shv_desc_src_addr;
    dst_ptr = (uint8_t*)shv_desc_dst_addr;
//    shv_job_header* first_shv_desc_ptr = (shv_job_header*)src_ptr;
    vcsFastPrintf("INFO: cpu_memcpy shave_descriptor %d ", shv_descriptor_size);
    cpu_memcpy(src_ptr, dst_ptr, shv_descriptor_size);
}

void swcSetStackPointer(u32 shaveNumber,u32 stackPointer,u32 stackSize)
{
    // make the stack size a multiple of 64 bytes
    stackSize = ((stackSize + 63) >> 6) << 6;

    u32 lastDataAddr = stackPointer - stackSize;
    assert(shaveNumber < TOTAL_NUM_SHAVES);

    DrvSvutIrfWrite(shaveNumber, 19, stackPointer);
    DrvSvutIrfWrite(shaveNumber, 20, lastDataAddr);
    DrvSvutIrfWrite(shaveNumber, 21, 0);
    return;
}

int startActShave(void)
{
#if 0
    if (IS_PLATFORM_FPGA) {
        vpuipCapsNCEShaveInfo_t nceShaveInfo;
        VpuipCapsGetNceShaveInfo(&nceShaveInfo);

        if (!(VpuipCapsIsNceShaveId(&nceShaveInfo, TARGET_SHAVE))) {
            vcsFastPuts((char*)"Warning: SoC not capable for test");
            // Not a FAIL condition, report PASS.
            return 0;
        }
    }
#endif
//    vcsFastPuts((char*)"INFO: DrvCprAllOn started");
    DrvCprAllOn(CPR_NCE);

//    vcsFastPuts((char*)"INFO: DrvNCEShaveL2CacheSetModeBM");
    DrvNCEShaveL2CacheSetModeBM(SHAVEL2C_MODE_BYPASS);
//    vcsFastPuts((char*)"INFO: DrvNCEShaveL2CacheSetPage");
    DrvNCEShaveL2CacheSetPage(1);

//    vcsFastPuts((char*)"INFO: leonFlushDcache() ");
    leonFlushDcache();
//    vcsFastPuts((char*)"INFO: swcResetShave() ");
    swcResetShave(TARGET_SHAVE);

//    vcsFastPuts((char*)"INFO: swcSetStackPointer() ");
    const u32 stack_pointer_address = 0x81005000;
    const u32 stack_size = 0x1000;
    swcSetStackPointer(TARGET_SHAVE,stack_pointer_address, stack_size);

//    vcsFastPuts((char*)"INFO: swcStartShave(TARGET_SHAVE) ");

    // Push starting SHAVE job to Act FIFO
    SET_REG_WORD(NNCMX_NCE_FIFO_3_MONITOR_0_ADR,0x0);
    SET_REG_WORD(NNCMX_NCE_FIFO_3_0_ADR, 0xFFFFFFFF);

    leonFlushDcache();

//    swcStartShave(TARGET_SHAVE,(u32)sk_nnActEntry_3010xx_text);
    swcStartShave(TARGET_SHAVE,(u32)&shvNN0_act_shave_runtime_shaveMain);

    return 0;
}

void stopActShave(void) {
    // Stop the activation SHAVE
    SET_REG_WORD(NNCMX_NCE_FIFO_3_0_ADR, 0);
//    vcsFastPuts((char*)"INFO: swcWaitShave(TARGET_SHAVE) ");
    swcWaitShave(TARGET_SHAVE);

    shv_job_header** hPtr = (shv_job_header**)0x2e1100A0;
    vcsFastPrintf("INFO: act-shave jobid = %p", *hPtr);

    // Flush the cache before we check the result
//    vcsFastPuts((char*)"INFO: swcLeonDataCacheFlush(TARGET_SHAVE) ");
    swcLeonDataCacheFlush();

//    vcsFastPuts((char*)"INFO: SW Layer inference complete");
#if defined (CONFIG_OYB_SKU)
    DrvCprAllOff(CPR_NCE);
#endif // CONFIG_OYB_SKU

}

shv_job_header __attribute__((section(".nncmx0.shared.data"))) gJob;
uint32_t __attribute__((section(".nncmx0.shared.data"))) gCompleted;

bool enqueueShaveJob(shv_job_header * job) {
    vcsFastPrintf("INFO: enqueueShaveJob() ptr = %p", reinterpret_cast<void*>(job->shv_kernel_address));

    job->job_completed_pointer = reinterpret_cast<uint64_t>(&gCompleted);
    gCompleted = false;
    cpu_memcpy((uint8_t*)job, (uint8_t*)&gJob, sizeof(shv_job_header));

    // Push starting SHAVE job to Act FIFO
    SET_REG_WORD(NNCMX_NCE_FIFO_3_MONITOR_0_ADR, 0x0);
    // vcsFastPrintf("INFO: act-shave submit desc start addrd : %p", job);
    SET_REG_WORD(NNCMX_NCE_FIFO_3_0_ADR, &gJob);
    leonFlushDcache();

    return true;
}




//void startActShaves(const uint8_t tile, const ActKernelRuntimeConfigs &cfgs) {
//    static_assert(AS_PER_TILE == SUPPORTED_ACT_SHV_PER_TILE_NB, "Only 2 ActShvs per tile is supported");
//
//    // Check that we are operating on a supported tile ID
//    if (!(tile < MAX_TILES)) {
//        nnLog(MVLOG_ERROR, "Invalid Shave type selected");
//        return;
//    }
//
//    // Shave IDs, depending on the tile
//    const uint32_t startShvId = tile * AS_PER_TILE;
//    const uint32_t maxShvId = startShvId + AS_PER_TILE;
//
//    // Set stack location, set the stack size, then start the Shave
//    for (uint32_t i = startShvId; i < maxShvId; i++) {
////        nnLog(MVLOG_INFO, "ACTSHV %d stack addr @ %p", i, actShvStacks[i]);
//        auto rc = ShaveCtrlSetStackAddr(actShvHnd[i], actShvStacks[i]);
//        if (rc != SHAVE_CTRL_SUCCESS) {
////            nnLog(MVLOG_ERROR, "ActShaveCtrlSetStackAddr: %d", (int)rc);
//        }
//
//        nnLog(MVLOG_INFO, "ACTSHV %d stack size = 0x%x", i, cfgs.stackSize_);
//        rc = ShaveCtrlSetStackSize(actShvHnd[i], cfgs.stackSize_);
//        if (rc != SHAVE_CTRL_SUCCESS) {
//            nnLog(MVLOG_ERROR, "ActShaveCtrlSetStackSize: %d", (int)rc);
//        }
//
//        nnLog(MVLOG_INFO, "ACTSHV %d WIN_%d = %p", i, mapWindowAddrMaskToName(ACT_RT_CODE_WINDOW),
//              reinterpret_cast<uint32_t>(actShvTextsBuffers[tile]));
//        rc = ShaveCtrlSetWindowAddr(actShvHnd[i], mapWindowAddrMaskToName(ACT_RT_CODE_WINDOW),
//                                    reinterpret_cast<uint32_t>(actShvTextsBuffers[tile]));
//        if (rc != SHAVE_CTRL_SUCCESS) {
//            nnLog(MVLOG_ERROR, "ShaveCtrlSetWindowAddr (for RT code buffer): 0x%x", ACT_RT_CODE_WINDOW);
//        }
//
//        nnLog(MVLOG_INFO, "!!!!!!! ACTSHV %d WIN_%d = %p", i, mapWindowAddrMaskToName(ACT_CMX_WINDOW),
//              cmxMapping.workareas_[tile].addr32());
//        rc = ShaveCtrlSetWindowAddr(actShvHnd[i], mapWindowAddrMaskToName(ACT_CMX_WINDOW),
//                                    cmxMapping.workareas_[tile].addr32());
//        if (rc != SHAVE_CTRL_SUCCESS) {
//            nnLog(MVLOG_ERROR, "ShaveCtrlSetWindowAddr (for window into CMX): ox%x", ACT_CMX_WINDOW);
//        }
//
//        nnLog(MVLOG_INFO, "Starting ACTSHV %d from %p windowed to A", i, actShvEntries[tile]);
//        auto fifoCfg = acts_cfgs[i];
//        printFifoConfig(unpackSHVConfig(fifoCfg));
//        rc = ShaveCtrlStart(actShvHnd[i], reinterpret_cast<void *>(actShvEntries[tile]), "i", fifoCfg);
//        if (rc != SHAVE_CTRL_SUCCESS) {
//            nnLog(MVLOG_ERROR, "ActShaveCtrlStart: %d", (int)rc);
//        }
//    }
//}
