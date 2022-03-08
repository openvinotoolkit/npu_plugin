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
#include <param_gather.h>

using namespace sw_params; // for GatherParams

extern "C" {
void singleShaveGather(uint32_t lParams) {
    const GatherParams* layerParams = reinterpret_cast<const GatherParams*>(lParams);

    half* pActInput = (half*)(layerParams->input.dataAddr);
    int32_t* pActIndices = (int32_t*)(layerParams->indices.dataAddr);
    int32_t axis = *(int32_t*)(layerParams->axis.dataAddr);
    half* pActOutput = (half*)(layerParams->output.dataAddr);

    // TODO: Check window in kernel
    half* pActWindowFp16 = (half*)(layerParams->windowfp16.dataAddr);
    half* pActWindowFp16Values = pActWindowFp16;
    for (int i = 0; i < 4 * 3 * 2; i++) {
        *pActWindowFp16Values = 0;
        if (i == 0) {
            *pActWindowFp16Values = 1234;
        }
        pActWindowFp16Values++;
    }

    int32_t* pActWindowInt32 = (int32_t*)(layerParams->windowint32.dataAddr);
    int32_t* pActWindowInt32Values = pActWindowInt32;
    for (int i = 0; i < 4 * 3 * 2; i++) {
        *pActWindowInt32Values = 1;
        if (i == 0) {
            *pActWindowInt32Values = 4321;
        }
        pActWindowInt32Values++;
    }
}
}
