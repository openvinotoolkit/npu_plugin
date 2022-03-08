//
// Copyright 2022 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#ifdef CONFIG_HAS_LRT_SRCS
#include <nn_log.h>
#else
#define nnLog(level, ...)
#endif
#ifdef CONFIG_TARGET_SOC_3720
#include <dma_shave_nn.h>
#else
#include <dma_shave.h>
#endif

#include <param_scatterNDUpdate.h>

using namespace sw_params;

namespace {

}  // namespace

namespace nn {
namespace shave_lib {

extern "C" {

void singleShaveScatterNDUpdate(uint32_t lParamsAddr) {
    printf("Hello from shave :)\n");

    const ScatterNDUpdateParams* layerParams = reinterpret_cast<const ScatterNDUpdateParams*>(lParamsAddr);

    half* p_act_data = (half*)(layerParams->input.dataAddr);
    half* p_indices_data = (half*)(layerParams->indices.dataAddr);
    half* p_updates_data = (half*)(layerParams->updates.dataAddr);
    half* p_act_out = (half*)(layerParams->output.dataAddr);

    for(int i = 0; i < 10; i++)
        printf("%x\n", p_act_data[i]);

    printf("***\n");
    for(int i = 0; i < 10; i++)
        printf("%x\n", p_updates_data[i]);
}

}
} // namespace shave_lib
} // namespace nn
