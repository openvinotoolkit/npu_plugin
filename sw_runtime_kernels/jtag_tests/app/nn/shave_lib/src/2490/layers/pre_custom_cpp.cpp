/*
* {% copyright %}
*/
#include <pre_custom_cpp.h>
#include <param_custom_cpp.h>
#include <sw_layer.h>
#include <sw_shave_res_manager.h>

#include "dma_shave_params.h"

# include <nn_log.h>

#if defined(__leon_rt__) || defined(__leon__)
#    include <nn_cache.h>
#endif

namespace nn {
namespace shave_lib {

void execCleanupCustomLayerCpp(const LayerParams *params, ShaveResourceManager *resMgr) {
    UNUSED(params);
    UNUSED(resMgr);
#ifdef ENABLE_CUSTOM_KERNEL_PERF_COUNTERS
    unsigned int nShaves = 1; // Filled in by getAssignedShaves
    auto res = resMgr->getAssignedShaves(nShaves);

    for (size_t i = 0; i < nShaves; i++) {
        auto *cmxParams = resMgr->getParams<CustomLayerCppParams>(res[i]);

#    if defined(__leon_rt__) || defined(__leon__)
        nn::cache::invalidate(*cmxParams->perf);
#    endif
        nnLog(MVLOG_PERF, "ShaveRes %d: Instructions: %llu    Cycles: %llu    Branches: %llu    Stalls: %llu", i,
              cmxParams->perf->instrs, cmxParams->perf->cycles, cmxParams->perf->branches, cmxParams->perf->stalls);
    }
#endif // ENABLE_CUSTOM_KERNEL_PERF_COUNTERS
}

void preCustomLayerCpp(const LayerParams *params, ShaveResourceManager *resMgr) {
    // DMA to copy layer params from DDR
    CustomLayerCppParams local_params;
    dmaShaveParams(local_params, params);
    const CustomLayerCppParams *cfg = & local_params;

    {
        // Would be better to move this buffer to UPA, allocated with getRawExecContext
        uint32_t *data = (uint32_t *)cfg->argBuffer;
        uint32_t i = 0;
        for (; i < cfg->inputsSize; ++i) {
            data[i] = (uint32_t)resMgr->getAbsoluteInputAddr(data[i]);
        }
        for (; i < cfg->inputsSize + cfg->outputsSize; ++i) {
            data[i] = (uint32_t)resMgr->getAbsoluteOutputAddr(data[i] - cfg->inputsSize);
        }
    }

// Setup arguments' buffers
#if defined(__leon_rt__) || defined(__leon__)
    nn::cache::flush(cfg->argBuffer, cfg->argBufferSize);
#endif

    // unsigned int numShaves = resMgr->getMaximumShaves();
    unsigned int numShaves = 1; // TODO: make sure that kernel works fine with multiple shaves
    auto res = resMgr->requestShaves(numShaves);

    // Need to flush instruction cache for any custom kernel.
    resMgr->invalidateL1L2InstCacheForAssignedShaves();

    // Flush L1/L2 data cache. This is needed after we patch the global arguments above, so that the
    // worker Shaves can have the same copy of the written data.
    resMgr->flushL1L2DataCacheForAssignedShaves();

    for (decltype(numShaves) shave = 0; shave < numShaves; shave++) {
        auto &sh = res[shave];
        CustomLayerCppParams *cmxParams = resMgr->getParams<CustomLayerCppParams>(sh);

        // Copy scheduler data section area to CMX, make a call before filling in `cmxParams`
        resMgr->setupShaveForKernel(sh);

        // Copy DDR Params into CMX Params
        *cmxParams = *cfg;
        cmxParams->scheduleInfo.shaveId = shave;
        cmxParams->scheduleInfo.nShaves = numShaves;

        // ToDO: patch arg buffer with locals from cmx if any on leon side

        // The following will properly set the CMX Data address and size
        resMgr->updateLayerParams(sh, cmxParams);

#ifdef ENABLE_CUSTOM_KERNEL_PERF_COUNTERS
        cmxParams->perf = (MvPerfStruct *)resMgr->getRawExecContext(sizeof(*cmxParams->perf));

        cmxParams->perf->stalls = 0;
        cmxParams->perf->branches = 0;
        cmxParams->perf->instrs = 0;
        cmxParams->perf->cycles = 0;

#    if defined(__leon_rt__) || defined(__leon__)
        nn::cache::flush(*cmxParams->perf);
#    endif
#endif // ENABLE_CUSTOM_KERNEL_PERF_COUNTERS

// flush into DDR for LEON-SHAVE memory coherency
#if defined(__leon_rt__) || defined(__leon__)
        nn::cache::flush(cmxParams, sizeof(CustomLayerCppParams));
#endif
    }
}
} // namespace shave_lib
} // namespace nn
