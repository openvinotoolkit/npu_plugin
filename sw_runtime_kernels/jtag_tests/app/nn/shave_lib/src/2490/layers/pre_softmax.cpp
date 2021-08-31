// {% copyright %}

#include "sw_layer.h"

#include "sw_shave_res_manager.h"
#include <nn_log.h>
#include <mvSubspaces.h>
#include <param_softmax.h>

#define SOFTMAX_SET_LENGTH 200

using namespace subspace;

// int32_t softmax_ret_status __attribute__((section(".cmx_direct.data")));

namespace nn {
namespace shave_lib {

extern "C" {
void preSoftmax(const LayerParams *params, ShaveResourceManager *resMgr) {

    auto lp = static_cast<const t_MvSoftMaxParamNClasses *>(params);

    int to_process = getTotal(lp->in_dims, lp->ndims);
    unsigned int shaves_no = std::min<int>(to_process, resMgr->getMaximumShaves());
    const nn::shave_lib::ShaveResource *shaves = resMgr->requestShaves(shaves_no);
    int32_t firstShave = 0;
    int32_t lastShave = firstShave + static_cast<int>(shaves_no) - 1;
    nnLog(MVLOG_DEBUG, "preSoftmax: run on %d SHAVEs\n", shaves_no);
    //    softmax_ret_status = 0;

#if 0 // 2classes and Inner_1x1xN variants temporary swithed off
    int shaves_per_set = shaves_no / to_process;
    if (inputData.ndims == 2 && axisDim == 2 && axis == 0 &&
            inputData.dims[0] * inputData.strides[0] < STUB_HEAP_DATA_SIZE / 2 &&
            axisIStride == static_cast<int>(axisDim * sizeof(fp16)) &&
            axisOStride == static_cast<int>(axisDim * sizeof(fp16)))
    {
        MVT_DPRINTF("softMaxParam2Classes %d\n", __LINE__);
        const s32 shaves_no = (last_shave - first_shave + 1);

        // Divide heights to shaves_no
        s32 remain_height = inputData.dims[1];
        const s32 height_per_shave = (remain_height + (shaves_no - 1)) / shaves_no;

        s32 real_last_shave = last_shave;
        u32 total_heights = 0;

        for (int i = first_shave; i <= last_shave; i++)
        {
            t_MvSoftMaxParam2Classes *softMaxParam2Classes = useShaveParam<t_MvSoftMaxParam2Classes>(i);
            u32 height_for_this_shave = 0;
            if (remain_height < height_per_shave)
            {
                height_for_this_shave = remain_height;  // Last shave
            }
            else
            {
                height_for_this_shave = height_per_shave;
            }
            remain_height -= height_for_this_shave;

            //  Doesn't call shaves any more if thire are no remained data
            if (height_for_this_shave == 0)
            {
                real_last_shave = i;
                break;
            }

            u32 in_offset  = (i - first_shave) * height_per_shave * inputData.strides[1];
            u32 out_offset = (i - first_shave) * height_per_shave * outputData.strides[1];

            total_heights += height_for_this_shave;

            //initialize SoftMax SHAVE param
            MVT_DPRINTF("softMaxParam2Classes %d\n", i);
            softMaxParam2Classes->input = (half*)(((u8*)ibuffer) + in_offset);
            softMaxParam2Classes->output = (half*)(((u8*)obuffer) + out_offset);
            softMaxParam2Classes->cmxslice = getCMXSliceDataSection(i);
            softMaxParam2Classes->dimX = inputData.dims[0];
            softMaxParam2Classes->dimY = height_for_this_shave; // Divided height
            softMaxParam2Classes->inDimXStride = inputData.strides[0];
            softMaxParam2Classes->inDimYStride = inputData.strides[1];
            softMaxParam2Classes->outDimXStride = outputData.strides[0];
            softMaxParam2Classes->outDimYStride = outputData.strides[1];
            softMaxParam2Classes->ret_status = &softmax_ret_status;

//            startShave(i, (u32) & MODULE_ENTRY(mvSoftMax2Classes), (u32)softMaxParam2Classes);
        }

//        waitShaves(first_shave, real_last_shave);
    }
    else
    if(shaves_per_set >= 2 && axis == 0 && (axisDim / shaves_per_set >= SOFTMAX_SET_LENGTH))
    {
        MVT_DPRINTF("softMaxParamNClasses %d\n", __LINE__);
        if(shaves_per_set > 5)
        {
            shaves_per_set = 5;
            shaves_no = to_process * shaves_per_set;
        }
        int shaves_per_set_rem = shaves_no % to_process;
        int shave_i = first_shave;
        for(int grp_i = 0; grp_i < to_process; grp_i++)
        {
            int grp_shave_no = shaves_per_set + ((shaves_per_set_rem-- > 0) ? 1 : 0);
            int elems_per_shave = axisDim / grp_shave_no;
            int elems_per_shave_rem = axisDim % grp_shave_no;

            int shift_in_dim2 = 0;

            for(int sh_in_grp = 0, first_grp_shave = shave_i; sh_in_grp < grp_shave_no; sh_in_grp++, shave_i++)
            {
                t_MvSoftMaxParamNClasses *softMaxParamNClasses = useShaveParam<t_MvSoftMaxParamNClasses>(shave_i);
                int real_elems_per_shave = elems_per_shave + ((elems_per_shave_rem-- > 0) ? 1 : 0);
                int col = grp_i % inputData.dims[0];
                int str = grp_i / inputData.dims[0];

                softMaxParamNClasses->input = ((half*)ibuffer + inputData.strides[0] * str + inputData.strides[0] * col);
                softMaxParamNClasses->output = ((half*)obuffer + outputData.strides[0] * str + outputData.strides[0] * col);
                softMaxParamNClasses->cmxslice = getCMXSliceDataSection(first_grp_shave);
                softMaxParamNClasses->auxcmxslice = getCMXSliceDataSection(first_grp_shave+1);
                softMaxParamNClasses->axis = axis;
                softMaxParamNClasses->axisDim = axisDim;
                softMaxParamNClasses->axisIStride = axisIStride;
                softMaxParamNClasses->axisOStride = axisOStride;
                softMaxParamNClasses->grpLeaderShave = first_grp_shave;
                softMaxParamNClasses->grpShavesNo = grp_shave_no;
                softMaxParamNClasses->start = shift_in_dim2;
                softMaxParamNClasses->toProcess = real_elems_per_shave;
                softMaxParamNClasses->this_shave = shave_i;
//                softMaxParamNClasses->dmaLinkAgent = dmaLinkAgent;

//                startShave(shave_i, (u32) & MODULE_ENTRY(mvSoftMaxInner_1x1xN), (u32)softMaxParamNClasses);

                shift_in_dim2 += real_elems_per_shave;
            }
        }
//        waitShaves(first_shave, first_shave + shaves_no - 1);
    }
    else
#endif
    {
        nnLog(MVLOG_DEBUG, "softMaxParamNClasses %d\n", __LINE__);
        // one or many softmax sets on one shave
        int step_size = to_process / shaves_no;
        int step_size_rem = to_process % shaves_no;

        nnLog(MVLOG_DEBUG, "axis %d, step_size %d, to_process %d, shaves_no %d\n", lp->axis,
                    step_size, to_process, shaves_no);

        int i = firstShave;
        int processed = 0;

        for (; i <= lastShave/* && processed < to_process*/; i++) {
            resMgr->setupShaveForKernel(shaves[i]);
            t_MvSoftMaxParamNClasses *softMaxParamNClasses =
            resMgr->getParams<t_MvSoftMaxParamNClasses>(shaves[i]);
            resMgr->updateLayerParams(shaves[i], softMaxParamNClasses);
            int to_process_on_shave = step_size + ((step_size_rem-- > 0) ? 1 : 0);
            nnLog(MVLOG_DEBUG, "i %d, to_process_on_shave %d lines, started from %d\n", i, to_process_on_shave, processed);

            softMaxParamNClasses->input = reinterpret_cast<const half *>(resMgr->getAbsoluteInputAddr(0));
            softMaxParamNClasses->output = reinterpret_cast<half *>(resMgr->getAbsoluteOutputAddr(0));

            softMaxParamNClasses->cmxslice = softMaxParamNClasses->cmxData;

            softMaxParamNClasses->ndims = lp->ndims;
            for (int32_t i = 0; i < MAX_ND_DIMS; i++) {
                softMaxParamNClasses->in_dims[i] = lp->in_dims[i];
                softMaxParamNClasses->in_strides[i] = lp->in_strides[i];
                softMaxParamNClasses->out_strides[i] = lp->out_strides[i];
            }
            softMaxParamNClasses->axis = lp->axis;
            softMaxParamNClasses->axisDim = lp->axisDim;
            softMaxParamNClasses->axisIStride = lp->axisIStride;
            softMaxParamNClasses->axisOStride = lp->axisOStride;
            softMaxParamNClasses->start = processed;
            softMaxParamNClasses->toProcess = to_process_on_shave;

            resMgr->updateLayerParams(shaves[i], softMaxParamNClasses);

#if defined(__leon_rt__) || defined(__leon__)
            rtems_cache_flush_multiple_data_lines(softMaxParamNClasses, sizeof(*softMaxParamNClasses));
#endif
            processed += to_process_on_shave;
        }
    }
}

}
} // namespace shave_lib
} // namespace nn
