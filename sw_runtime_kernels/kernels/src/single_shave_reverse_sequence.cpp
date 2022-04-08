//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mvSubspaces.h>
#include <param_reverse_sequence.h>

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

void single_shave_reverse_sequence(uint32_t lParamsAddr) {
    const ReverseSequenceParams* lParams = (const ReverseSequenceParams*)lParamsAddr;

    half* p_act_data = (half*)(lParams->input.dataAddr);  // 0x1F000000
    int32_t* seq_lengths_data = (int32_t*)(lParams->seq_lengths.dataAddr);
    half* p_act_out = (half*)(lParams->output.dataAddr);  // 0x1F004000

    int64_t* p_stride = (int64_t*)(lParams->input.stridesAddr);
    int32_t nElements = 1;
    int32_t* pDims = (int32_t*)(lParams->input.dimsAddr);
    int32_t numDims = lParams->input.numDims;
    int32_t batch_axis = lParams->batch_axis;
    int32_t seq_axis = lParams->seq_axis;

    for (int32_t i = 0; i < numDims; ++i) {
        nElements *= pDims[i];
        p_stride[i] /= (8 * sizeof(half));
    }

    int32_t length = nElements * sizeof(half);
    memcpy_s(p_act_out, length, p_act_data, length);

    switch (numDims) {
    case 2: {
        for (int32_t batch = 0; batch < pDims[batch_axis]; batch++) {
            for (int32_t seqAxis = 0; seqAxis < pDims[seq_axis]; seqAxis++) {
                int32_t seqLength = seq_lengths_data[batch];
                if (seqLength - 1 - seqAxis >= 0) {
                    int32_t ind = batch * p_stride[batch_axis] + seqAxis * p_stride[seq_axis];
                    int32_t indChange = (seqLength - 1 - 2 * seqAxis) * p_stride[seq_axis];
                    int32_t newind = ind + indChange;
                    p_act_out[newind] = p_act_data[ind];
                }
            }
        }
        return;
    }
    case 3: {
        int32_t numElement = nElements / (pDims[batch_axis] * pDims[seq_axis]);
        int32_t remStride = p_stride[0] * p_stride[1] * p_stride[2] / (p_stride[batch_axis] * p_stride[seq_axis]);

        for (int32_t batch = 0; batch < pDims[batch_axis]; batch++) {
            for (int32_t seqAxis = 0; seqAxis < pDims[seq_axis]; seqAxis++) {
                int32_t seqLength = seq_lengths_data[batch];
                if (seqLength - 1 - seqAxis >= 0) {
                    int32_t initOffset = batch * p_stride[batch_axis] + seqAxis * p_stride[seq_axis];
                    int32_t indChange = (seqLength - 1 - 2 * seqAxis) * p_stride[seq_axis];
                    for (int32_t i = 0; i < numElement; i++) {
                        int32_t ind = initOffset + i * remStride;
                        int32_t newind = ind + indChange;
                        p_act_out[newind] = p_act_data[ind];
                    }
                }
            }
        }
        return;
    }
    case 4: {
        int32_t remAxisLow = 0;
        for (int32_t i = 0; i < numDims; i++) {
            if (pDims[i] != batch_axis && pDims[i] != seq_axis) {
                remAxisLow = i;
                break;
            }
        }
        int32_t remAxisHigh = 6 - remAxisLow - seq_axis - batch_axis;
        int32_t numElement = pDims[remAxisLow];
        int32_t remStrideLow = p_stride[remAxisLow];

        for (int32_t batch = 0; batch < pDims[batch_axis]; batch++) {
            for (int32_t seqAxis = 0; seqAxis < pDims[seq_axis]; seqAxis++) {
                int32_t seqLength = seq_lengths_data[batch];
                if (seqLength - 1 - seqAxis >= 0) {
                    int32_t initOffset = batch * p_stride[batch_axis] + seqAxis * p_stride[seq_axis];
                    for (int axisHigh = 0; axisHigh < pDims[remAxisHigh]; ++axisHigh) {
                        for (int32_t i = 0; i < numElement; i++) {
                            int32_t ind = initOffset + i * remStrideLow;
                            int32_t indChange = (seqLength - 1 - 2 * seqAxis) * p_stride[seq_axis];
                            int32_t newind = ind + indChange;
                            p_act_out[newind] = p_act_data[ind];
                        }
                        initOffset += p_stride[remAxisHigh];
                    }
                }
            }
        }
        return;
    }

    default: {
        return;
    }
    }
}
}
}  // namespace shave_lib
}  // namespace nn
