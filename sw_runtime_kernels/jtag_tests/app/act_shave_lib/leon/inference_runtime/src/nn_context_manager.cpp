/*
 * {% copyright %}
 *
 */

#include "nn_context_manager.h"

#include <assert.h>
#include <mv_types.h>
#include <nn_fifo.h>
#include <nn_fifo_configs.h>
#include <stdint.h>
#include <Dma.h>
#include "nn_cache.h"
#include "nn_fifo_manager.h"

#ifdef NN_ENABLE_CONTEXT_DEBUGGING
#define dbgPrint(...) nnLog(MVLOG_INFO, __VA_ARGS__)
#else
#define dbgPrint(...)
#endif

#define DDR_SECTION __attribute__((section(".ddr.data"), aligned(64)))
#define DDR_WIPE_PATTERN_SIZE 128

static uint8_t DDR_SECTION ddr_dma_wipe_pattern[DDR_WIPE_PATTERN_SIZE] = {0};

using namespace nn::common_runtime::fifo;

namespace nn {
namespace inference_runtime {
namespace context {
namespace {

void init_cmx_wipe_buffer() {
    memset((void *)ddr_dma_wipe_pattern, (uint8_t)0, DDR_WIPE_PATTERN_SIZE);
    nn::cache::flush(ddr_dma_wipe_pattern, DDR_WIPE_PATTERN_SIZE);
}
} // namespace

DmaStatus wipe_cmx(char tile, uint32_t dest, uint32_t size) {
    DmaStatus status{DMA_SUCCESS};
    DmaDescriptor dma_desc_cmx;
    DmaJobHandle dma_handle_cmx;
    DmaDescriptorConfig dma_desc_config_cmx;
    DmaJobConfig dma_job_config;
    const uint32_t cid = get_context_id(tile);

    // create dma job handle
    status = DmaJobCreate(&dma_handle_cmx);
    if (status != DMA_SUCCESS) {
        nnLog(MVLOG_ERROR, "Failed to create DMA job handle.");
        return status;
    }

    // setup job config
    dma_job_config.link_agent = 0;
    dma_job_config.wait = true;
    dma_job_config.callback = nullptr;
    dma_job_config.context = nullptr;
    dma_job_config.context_id = (uint8_t)cid;

    // setup dma descriptor config for dma clear to cmx
    dma_desc_config_cmx.src = static_cast<uint64_t>(reinterpret_cast<uint32_t>(ddr_dma_wipe_pattern));
    dma_desc_config_cmx.dst = static_cast<uint64_t>(reinterpret_cast<uint32_t>(dest));
    dma_desc_config_cmx.size = size;
    dma_desc_config_cmx.src_width = DDR_WIPE_PATTERN_SIZE;
    dma_desc_config_cmx.dst_width = size;
    dma_desc_config_cmx.src_stride = 0;
    dma_desc_config_cmx.dst_stride = 0;
    dma_desc_config_cmx.num_planes = 1;
    dma_desc_config_cmx.src_plane_stride = DMA_DEFAULT_PLANE_STRIDE;
    dma_desc_config_cmx.dst_plane_stride = DMA_DEFAULT_PLANE_STRIDE;
    dma_desc_config_cmx.burst_size = 64;
    dma_desc_config_cmx.feature_flags = HGL_DMA_FLAG_CRITICAL_TASK;
    dma_desc_config_cmx.barrier_cfg = nullptr;

    // add descriptor to job handle
    status = DmaJobAddDescriptor(&dma_handle_cmx, &dma_desc_cmx, &dma_desc_config_cmx);
    if (status != DMA_SUCCESS) {
        nnLog(MVLOG_ERROR, "Failed to add descriptor to job.");
        return status;
    }

    status = DmaJobStart(&dma_handle_cmx, HGL_DMA_TYPE_NCE, HGL_DMA_ENGINE_0, DMA_MODE_NORMAL, &dma_job_config);
    if (status != DMA_SUCCESS) {
        nnLog(MVLOG_ERROR, "CMX wipe DmaJobStart with DmaStatus: %d", status);
        return status;
    }
    return DMA_SUCCESS;
}

void configure_nce_shave_l2_for_single_user_context() {
    configureShaveL2ForSingleUserContext();
}

void configure_nce_shave_l2_for_user_context_per_tile() {
    configureShaveL2ForUserContextPerTile();
}

void init_context_default_state(StaticMapping &sm) {
    initDefaultState();
    init_cmx_wipe_buffer();
    configureShaveL2ForSingleUserContext();
    flushNceShaveL2GlobalContextPart();

    context_violation_irq_enable(UINT32_MAX);

#ifdef NN_ENABLE_CONTEXT_DEBUGGING
    ViolationState state;
    // debug_print_L2_config();
    if (check_and_record_context_violation(state)) {
        context_violation_irq_clear(state.cid_viol_bitfield);
        print_context_violation(state);
    }
#endif

    flush_tiles_of_context(sm);
    prepare_tiles_for_context(DEFAULT_USER_CONTEXT_ID, sm);
    configureSNNIPCFifosForGlobalContext(0);

#ifdef NN_ENABLE_CONTEXT_DEBUGGING
    if (check_and_record_context_violation(state)) {
        context_violation_irq_clear(state.cid_viol_bitfield);
        print_context_violation(state);
    }
#endif
    // TODO: do other one-time init stuff
}

bool configured_for_single_context() {
    return shaveL2ConfiguredForSingleContext();
}

uint8_t get_context_id(uint8_t tile) {
    return getContextId(tile);
}

uint32_t get_host_ID_mapping(uint32_t context_id) {
    UNUSED(context_id);
    // FIXME: implement HSSID mapping
    nnLog(MVLOG_WARN, "Host Substream ID mapping not implemented!");

    return 0;
}

ResourceMask get_bound_resources(uint32_t context_id) {
    return getBoundResources(context_id);
}

bool flush_tile_of_context(StaticMapping &sm, uint8_t tile) {
    // https://docs.intel.com/documents/MovidiusExternal/vpu27/MTL/HW/VPU_HAS.html#context-switchretirement-flow

    if (isInvalidTile(tile)) {
        nnLog(MVLOG_ERROR, "Incorrect tile number %d requested for flush. Single context: %d\n", tile,
              shaveL2ConfiguredForSingleContext());
        return false;
    }

    cleanFifoResources(AS_WORK_FIFO_NUM[tile], asWorkFifoIndexMask[tile]);
    cleanFifoResources(CTRL_FIFO_NUM[tile], ctrlFifoIndexMask[tile]);
    cleanBarrierResources(tile);
    flushActShvAndCaches(tile);
    recordAndResetPerfCounters(tile);
    flushNceShaveL2UserContextPartForTile(tile);

    // Wipe buffer is in LRT memory (global context), so we need to switch after flushing
    // The correct context ID will be configured during the next prepare_tiles_for_context
    configureComputeSliceForContext(tile, GLOBAL_CONTEXT_ID);
    configureStolenWindow(tile, sm.globalData_[tile].addr32(), sm.globalData_[tile].size());
    wipe_cmx(tile, sm.workareas_[tile].addr32(), sm.workareas_[tile].size());

    // Wipe-Out NCE CMX Slice Memories allocated to retiring context.
    // Note: CMX Wipe Out is expected to take ~32μs.

    return true;
}

bool prepare_tile_for_context(uint8_t tile, uint32_t context_id, StaticMapping &sm) {
    // Allocate static resources to new context:
    //   NCE Compute Slice Context Allocation - NCE_CNTX_ID_NCE_SLICE(N),
    //   SHAVE L2 Cache Partition Context Allocation - NCE_CNTX_ID_L2C_PARTIT(N)
    //   and NCE_SHAVE_L2C_PART_ID_ASSIGNx_ADDR, ACT-SHAVE FIFOs (IPC/Work)
    //   Context Allocation - NCE_CNTX_ID_ACT_SHV_FIFO(N) &
    //   NCE_CNTX_ID_IPC_SHV_FIFO(N), HW Barrier Context Allocation -
    //   NCE_CNTX_ID_BARRIER(N)

    // NCE_CNTX_ID_NCE_SLICE0_ADR
    // NCE_CNTX_ID_NCE_SLICE1ADDR

    if (isInvalidTile(tile)) {
        nnLog(MVLOG_ERROR, "Incorrect tile number %d requested for prepare. Single context: %d\n", tile,
              shaveL2ConfiguredForSingleContext());
        return false;
    }

    if ((0b11111 & context_id) != context_id) {
        nnLog(MVLOG_ERROR, "Context ID %x exceeds 5 bits", context_id);
        return false;
    }

    const uint16_t fifo3Mask = asWorkFifoIndexMask[tile] | asPerfFifoIndexMask[tile];

    nnLog(MVLOG_DEBUG, "Tile %d", tile);
    nnLog(MVLOG_DEBUG, "Work FIFO mask: %x", fifo3Mask);
    nnLog(MVLOG_DEBUG, "IPC  FIFO mask: %x", ctrlFifoIndexMask[tile]);

    configureNceShaveL2ForContext(tile, context_id);
    configureComputeSliceForContext(tile, context_id);
    configureStolenWindow(tile, sm.globalData_[tile].addr32(), sm.globalData_[tile].size());
    configureBarriersForContext(tile, context_id);
    configureFifosForContext(AS_WORK_FIFO_NUM[tile], fifo3Mask, context_id);
    configureFifosForContext(CTRL_FIFO_NUM[tile], ctrlFifoIndexMask[tile], context_id);

    // FIXME: the act-shaves probably should not be started here
    // resumeActShv(tile);

    return true;
}

void flush_tiles_of_context(StaticMapping &sm) {
    flush_tile_of_context(sm, 0);

    if (!shaveL2ConfiguredForSingleContext()) {
        flush_tile_of_context(sm, 1);
    }
}

bool prepare_tiles_for_context(uint32_t context_id, StaticMapping &sm) {
    if (shaveL2ConfiguredForSingleContext()) {
        return prepare_tile_for_context(0, context_id, sm);
    }

    auto ret = prepare_tile_for_context(0, context_id, sm);
    ret &= prepare_tile_for_context(1, context_id, sm);

    return ret;
}

void context_violation_irq_enable(uint32_t cid_mask) {
    contextViolationIrqEnable(cid_mask);
}

void context_violation_irq_clear(uint32_t cid_mask) {
    contextViolationIrqClear(cid_mask);
}

bool check_and_record_context_violation(ViolationState &state) {
    return checkAndRecordContextViolation(state);
}

void print_context_violation(ViolationState &state) {
    if (state.cid_viol_bitfield) {
        switch (state.target) {
            case NNCMX_FIREWALLS:
                nnLog(MVLOG_WARN, "Context violation in NN-CMX Firewalls from context_id %d accessing %p (dump: 0x%x)",
                      state.cid, state.cid_viol_addr, state.cid_viol_target);
                break;
            case BARRIER_ACCESS:
                nnLog(MVLOG_WARN,
                      "Context violation during Barrier Access Check from context_id %d accessing %p (dump: 0x%x)",
                      state.cid, state.cid_viol_addr, state.cid_viol_target);
                break;
            case DPU_WORK_FIFO:
                nnLog(MVLOG_WARN,
                      "Context violation during DPU Work FIFO Check from context_id %d accessing %p (dump: 0x%x)",
                      state.cid, state.cid_viol_addr, state.cid_viol_target);
                break;
            case MEMBLOCK_FIFO:
                nnLog(MVLOG_WARN,
                      "Context violation during MemBlock FIFO Check from context_id %d accessing %p (dump: 0x%x)",
                      state.cid, state.cid_viol_addr, state.cid_viol_target);
                break;
            case ACTSHAVE_WORK_FIFO:
                // If violation is triggered for invalid access to global configuration registers (registers which can
                // not associated with barriers i.e. NCE_FIFO_2_AFULL_IRQ_EN) value of 0x4F will be returned
                nnLog(MVLOG_WARN,
                      "Context violation during ACT-SHAVE Work FIFO Check from context_id %d to FIFO3_%d accessing %p "
                      "(dump: 0x%x)",
                      state.cid, state.target_val, state.cid_viol_addr, state.cid_viol_target);

                break;
            case IPC_FIFO:
                nnLog(MVLOG_WARN, "Context violation during IPC FIFO Check from context_id %d accessing %p (dump: 0x%x)",
                      state.cid, state.cid_viol_addr, state.cid_viol_target);
                break;
            case DMA_CONFIGURATION_ACCESS:
                nnLog(MVLOG_WARN,
                      "Context violation during DMA Configuration Access Check from context_id %d accessing %p (dump: "
                      "0x%x)",
                      state.cid, state.cid_viol_addr, state.cid_viol_target);
                break;
            case COMPRESSION_CONFIGURATION_ACCESS:
                nnLog(MVLOG_WARN,
                      "Context violation during Compression Configuration Access Check from context_id %d accessing %p "
                      "(dump: 0x%x)",
                      state.cid, state.cid_viol_addr, state.cid_viol_target);
                break;
            case UNKNOWN_VIOLATION:
                nnLog(MVLOG_ERROR,
                      "Internal Error: Unknown Context violation detected from context_id %d accessing %p "
                      "(dump: 0x%x)",
                      state.cid, state.cid_viol_addr, state.cid_viol_target);
                break;
            default:
                nnLog(MVLOG_ERROR, "Internal Error: Undefined context state!");
                break;
        }

#ifdef NN_ENABLE_CONTEXT_DEBUGGING
        {
            int cid{-1};
            uint32_t tmp{state.cid_viol_bitfield};
            do {
                cid++;
                tmp = tmp & 1 ? 0 : tmp >> 1;
            } while (tmp);

            if (state.cid != cid) {
                nnLog(MVLOG_ERROR, "Internal State Error: CID!=Bitfield mismatch: %d!=%d", state.cid, cid);
            }
        }
#endif
    }
}
} // namespace context
} // namespace inference_runtime
} // namespace nn
