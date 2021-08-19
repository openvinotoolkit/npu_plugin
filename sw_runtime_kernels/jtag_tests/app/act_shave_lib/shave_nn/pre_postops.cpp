// {% copyright %}

#include "pre_postops.h"
#include "param_postops.h"
#include "sw_layer.h"
#include "sw_shave_res_manager.h"

#include <stdio.h>

namespace nn {
namespace shave_lib {

void prePostOpsHWC(const LayerParams *params, ShaveResourceManager *resMgr) {

    const PostOpsParams *ddrParams = static_cast<const PostOpsParams *>(params);

    unsigned int numShaves = resMgr->getMaximumShaves();

    auto input_data = reinterpret_cast<const half*>(resMgr->getAbsoluteInputAddr(0));
    const half* weights_data = nullptr;
    const half* biases_data = nullptr;

    int input_idx = 1;
    if (ddrParams->has_weights) {
        weights_data = reinterpret_cast<const half*>(resMgr->getAbsoluteInputAddr(input_idx));
        input_idx++;
    }
    if (ddrParams->has_biases) {
        biases_data = reinterpret_cast<const half*>(resMgr->getAbsoluteInputAddr(input_idx));
        input_idx++;
    }
    auto output_data = reinterpret_cast<half*>(resMgr->getAbsoluteOutputAddr(0));

    auto res = resMgr->requestShaves(numShaves);

    int first_shave = 0;
    int last_shave = numShaves - 1;

    u32 no_shaves = last_shave - first_shave + 1;
    u32 h_per_shave = ddrParams->height / no_shaves;
    u32 h_per_shave_remainder = ddrParams->height % no_shaves;
    s32 in_offset = 0;
    s32 out_offset = 0;

    for(int shave_i = first_shave; shave_i <= last_shave; ++shave_i)
    {
        const ShaveResource &sh = res[shave_i];
        resMgr->setupShaveForKernel(sh);
        t_HWCPostOps3DParams *postOpsParams = resMgr->getParams<t_HWCPostOps3DParams>(sh);

        *postOpsParams = *static_cast<const t_HWCPostOps3DParams*>(ddrParams);

        postOpsParams->input        = (input_data  + in_offset);
        postOpsParams->output       = (output_data + out_offset);
        postOpsParams->weights      = weights_data;
        postOpsParams->bias         = biases_data;
        postOpsParams->height       = h_per_shave;

        // Distribute one line of width to the first h_per_shave_remainder shave.
        in_offset += h_per_shave * ddrParams->in_step;
        out_offset += h_per_shave * ddrParams->out_step;

        if(h_per_shave_remainder != 0)
        {
            postOpsParams->height += 1;
            in_offset += ddrParams->in_step;
            out_offset += ddrParams->out_step;
            --h_per_shave_remainder;
        }

        resMgr->updateLayerParams(sh, postOpsParams);

#if defined(__leon_rt__) || defined(__leon__)
        rtems_cache_flush_multiple_data_lines(postOpsParams, sizeof(t_HWCPostOps3DParams));
#endif
    }
}
#if 0
void prePostOpsCHW(const LayerParams *params, ShaveResourceManager *resMgr) {
    const PostOpsParams *ddrParams = static_cast<const PostOpsParams *>(params->layerParams);
    unsigned int numShaves = resMgr->getMaximumShaves();

    auto input_data = reinterpret_cast<const half*>(resMgr->getAbsoluteInputAddr(0));
    const half* weights_data = nullptr;
    const half* biases_data = nullptr;

    int input_idx = 1;
    if (ddrParams->has_weights) {
        weights_data = reinterpret_cast<const half*>(resMgr->getAbsoluteInputAddr(input_idx));
        input_idx++;
    }
    if (ddrParams->has_biases) {
        biases_data = reinterpret_cast<const half*>(resMgr->getAbsoluteInputAddr(input_idx));
        input_idx++;
    }

    auto output_data = reinterpret_cast<half*>(resMgr->getAbsoluteOutputAddr(0));

    auto res = resMgr->requestShaves(numShaves);

    int first_shave = 0;
    int last_shave = numShaves - 1;

    u32 channels = ddrParams->channels;

    u32 used_shaves = std::min(static_cast<u32>(last_shave - first_shave + 1), channels);
    last_shave = first_shave + used_shaves - 1;

    u32 channels_per_shave = channels / used_shaves;
    u32 channels_per_shave_remainder = channels % used_shaves;
    s32 in_offset = 0;
    s32 out_offset = 0;
    s32 channels_offset = 0;

    for (int shave_i = first_shave; shave_i <= last_shave; ++shave_i)
    {
        const ShaveResource &sh = res[shave_i];
        resMgr->setupShaveForKernel(sh);
        t_CHWPostOps3DParams *postOpsParams = resMgr->getParams<t_CHWPostOps3DParams>(sh);
        *postOpsParams = *static_cast<const t_CHWPostOps3DParams*>(ddrParams);

        postOpsParams->input        = (input_data  + in_offset);
        postOpsParams->output       = (output_data + out_offset);
        postOpsParams->weights      = weights_data ? (weights_data + channels_offset) : weights_data;
        postOpsParams->bias         = biases_data ? (biases_data + channels_offset) : biases_data;

        u32 channels_extra = (channels_per_shave_remainder != 0) ? 1 : 0;
        u32 channels_used = channels_per_shave + channels_extra;

        postOpsParams->channels     = channels_used;

        in_offset += ddrParams->in_step * ddrParams->height * channels_used;
        out_offset += ddrParams->out_step * ddrParams->height * channels_used;
        channels_offset += channels_used;
        channels_per_shave_remainder -= channels_extra;

        resMgr->updateLayerParams(sh, postOpsParams);
#if defined(__leon_rt__) || defined(__leon__)
        rtems_cache_flush_multiple_data_lines(postOpsParams, sizeof(t_CHWPostOps3DParams));
#endif
    }
}

void prePostOpsHCW(const LayerParams *params, ShaveResourceManager *resMgr) {
    const PostOpsParams *ddrParams = static_cast<const PostOpsParams *>(params->layerParams);
    unsigned int numShaves = 1;
    //TODO: enable multi shaves, now it works incorrect for Relu family - S#48990
    //unsigned int numShaves = resMgr->getMaximumShaves();

    auto input_data = reinterpret_cast<const half*>(resMgr->getAbsoluteInputAddr(0));
    const half* weights_data = nullptr;
    const half* biases_data = nullptr;

    int input_idx = 1;
    if (ddrParams->has_weights) {
        weights_data = reinterpret_cast<const half*>(resMgr->getAbsoluteInputAddr(input_idx));
        input_idx++;
    }
    if (ddrParams->has_biases) {
        biases_data = reinterpret_cast<const half*>(resMgr->getAbsoluteInputAddr(input_idx));
        input_idx++;
    }

    auto output_data = reinterpret_cast<half*>(resMgr->getAbsoluteOutputAddr(0));

    auto res = resMgr->requestShaves(numShaves);

    int first_shave = 0;
    int last_shave = numShaves - 1;

    u32 no_shaves =  last_shave - first_shave + 1;
    u32 h_per_shave = ddrParams->height / no_shaves;
    u32 h_per_shave_remainder = ddrParams->height % no_shaves;
    s32 in_out_offset = 0, start_line = 0;

    for(int shave_i = first_shave; shave_i <= last_shave; ++shave_i)
    {
        const ShaveResource &sh = res[shave_i];
        resMgr->setupShaveForKernel(sh);
        t_HCWPostOps3DParams *postOpsParams = resMgr->getParams<t_HCWPostOps3DParams>(sh);

        *postOpsParams = *static_cast<const t_HCWPostOps3DParams*>(ddrParams);

        postOpsParams->input        = (input_data  + in_out_offset);
        postOpsParams->output       = (output_data + in_out_offset);
        postOpsParams->offset       = in_out_offset;
        postOpsParams->start_line   = start_line;
        postOpsParams->weights      = weights_data;
        postOpsParams->bias         = biases_data;
        postOpsParams->height       = h_per_shave;

        // Distribute one line of width to the first h_per_shave_remainder shave.
        in_out_offset += h_per_shave * ddrParams->out_step;

        if(h_per_shave_remainder != 0)
        {
            postOpsParams->height += 1;
            in_out_offset += ddrParams->out_step;
            --h_per_shave_remainder;
        }
        start_line += postOpsParams->height;

        resMgr->updateLayerParams(sh, postOpsParams);

#if defined(__leon_rt__) || defined(__leon__)
        rtems_cache_flush_multiple_data_lines(postOpsParams, sizeof(t_HCWPostOps3DParams));
#endif
    }
}
#endif
} // namespace shave_lib
} // namespace nn
