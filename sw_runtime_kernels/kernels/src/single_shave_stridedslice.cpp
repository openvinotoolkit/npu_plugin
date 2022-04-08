//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mvSubspaces.h>
#include <numeric>
#include <algorithm>

#include <param_stridedslice.h>

namespace nn {
namespace shave_lib {

extern "C" {
void single_shave_stridedslice(uint32_t lParamsAddr) {
        const StridedSliceParams* layerParams = reinterpret_cast<const StridedSliceParams*>(lParamsAddr);

        MemRefData inputData = layerParams->input;
        MemRefData outputData = layerParams->output;

        half* p_act_data = (half*)(layerParams->input.dataAddr);
        half* p_act_out = (half*)(layerParams->output.dataAddr);

        const int32_t* inputDims = (int32_t *)(inputData.dimsAddr);
        const int32_t* outputDims = (int32_t *)(outputData.dimsAddr);

        int64_t* p_begins = (int64_t*)(layerParams->begins);
        int64_t* p_strides = (int64_t*)(layerParams->strides);

        int32_t ndims = inputData.numDims;
        int32_t outndims = outputData.numDims;

        const int total = subspace::getTotal(outputDims, outndims);

        int32_t setCoords[MAX_ND_DIMS]{};
        int32_t memoryStep[MAX_ND_DIMS]{};
        memoryStep[0] = 1;

        for (int32_t i = 1; i != ndims; i++)
        {
            memoryStep[i] = memoryStep[i-1]*inputDims[i-1];
        }
        subspace::getCoord(0, outputDims, outndims, setCoords);

        for (int32_t i = 0; i != total; i++)
        {
            int32_t offset = 0;
            for (int32_t j = 0; j != ndims; j++)
            {
                int32_t idx = ndims - j - 1;
                offset += (p_begins[idx] + setCoords[j] * p_strides[idx]) * memoryStep[j];
            }
            p_act_out[i] = p_act_data[offset];
            subspace::incrementNCoord(setCoords, outputDims, outndims, 1);
        }
}
}
}
}
