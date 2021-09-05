/*
* {% copyright %}
*/
#include "layers/pre_permute.h"
#include "layers/param_permute.h"
#include "sw_layer.h"
#include "sw_shave_res_manager.h"

#include <algorithm>
#include <mv_types.h> // For uint32_t and friends FIXME: uint32_t no longer defined for NFS
#include <string.h>
#include <nn_log.h>

#define DIVIDE_UP(dividend, divisor) (((dividend) + (divisor) -1) / (divisor))
#define MAX(a,b)         ((a) > (b) ? (a) : (b))
#define MIN(a,b)         ((a) < (b) ? (a) : (b))

#if defined(__leon_rt__) || defined(__leon__)
#include "dma_leon.h"
#else
#include "dma_shave.h"
#endif

namespace nn {
namespace shave_lib {

void prePermute1D(const LayerParams *layerParams, ShaveResourceManager *srm) {
    const PermuteParams1D *pp = static_cast<const PermuteParams1D *>(layerParams);
    auto in  = (u8 *)srm->getAbsoluteInputAddr(0);
    auto out = (u8 *)srm->getAbsoluteOutputAddr(0);

#if defined(__leon_rt__) || defined(__leon__)
    DmaAlLeon task;
#else
    DmaAlShave task;
#endif

    if(pp->inWidthStride == pp->outWidthStride)
    {
        task.start((u8*)in, (u8*)out, pp->inWidth * pp->inWidthStride);
    }
    else
    {
        task.start((u8*)in, (u8*)out,
                pp->inWidth * pp->bpp,
                pp->bpp,
                pp->bpp,
                pp->inWidthStride,
                pp->outWidthStride);
    }
    task.wait();
}

void prePermute(const LayerParams *layerParams, ShaveResourceManager *srm) {
    unsigned int numShaves = srm->getMaximumShaves();
    const PermuteParams *pp = static_cast<const PermuteParams *>(layerParams);
    const s32 first_shave = 0;
    auto shavesRes = srm->requestShaves(numShaves);
    auto in  = (u8 *)srm->getAbsoluteInputAddr(0);
    auto out = (u8 *)srm->getAbsoluteOutputAddr(0);
    const s32 last_shave = first_shave + numShaves - 1;

    int n_of_slices = pp->n_of_slices;
    int sliceDivider = 1;

    int max = pp->maxInnerDims;
    if(n_of_slices / numShaves < 2 && max > 255)
    {
        sliceDivider = DIVIDE_UP(2 * numShaves, n_of_slices);
        if(sliceDivider > max)
        {
            sliceDivider = DIVIDE_UP(1 * numShaves, n_of_slices);
        }
        int step = DIVIDE_UP(max, sliceDivider);
        sliceDivider = DIVIDE_UP(max, step);

        sliceDivider = (sliceDivider < max)
                       ? sliceDivider : 1;
        n_of_slices *= sliceDivider;
    }

    nnLog(MVLOG_DEBUG, "n_of_slices %d, sliceDivider %d\n", n_of_slices, sliceDivider);
    int slices_step = DIVIDE_UP(n_of_slices, numShaves);
    int c_slice = 0;

    for(int shave_idx = first_shave; shave_idx <= last_shave; shave_idx++, c_slice += slices_step)
    {
        auto shaveRes = shavesRes[shave_idx];

        mvPermuteParams *cmxParams = srm->getParams<mvPermuteParams>(shaveRes);
        srm->setupShaveForKernel(shaveRes);
        srm->updateLayerParams(shaveRes, cmxParams);

        t_PermParam *p_permuteParams = &cmxParams->mvPermuteUnion.mvPermParam;

        cmxParams->is_shave_enabled = c_slice < n_of_slices;

        if(cmxParams->is_shave_enabled){
            p_permuteParams->bpp           = pp->bpp;
            p_permuteParams->input         = in;
            p_permuteParams->output        = out;
            p_permuteParams->slice = c_slice;
            p_permuteParams->n_slices = MIN(slices_step, n_of_slices - c_slice);
            p_permuteParams->sliceDivider = sliceDivider;
            p_permuteParams->parsedPerm = pp->parsedPerm;
            p_permuteParams->cmxData = (u8*)srm->getDataAddr(shaveRes);
            p_permuteParams->cmxSize = cmxParams->availableCmxBytes;
            cmxParams->run_mv_transpose = pp->run_mv_transpose;
        }
#if defined(__leon_rt__) || defined(__leon__)
        rtems_cache_flush_multiple_data_lines(cmxParams, sizeof(mvPermuteParams));
#endif
    }
}

} // namespace shave_lib
} // namespace nn
