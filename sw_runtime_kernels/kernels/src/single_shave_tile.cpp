//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <param_tile.h>
#include "mvSubspaces.h"
#define bpp sizeof(half)

using namespace sw_params;

namespace nn {
namespace shave_lib {
void reorgDimsOrder(int32_t& dim4, int32_t& dim5, int32_t& dim6, int32_t& dim7, int extended_dim[MAX_TILE_DIMS]) {
    dim4 = extended_dim[4];
    dim5 = extended_dim[5];
    dim6 = extended_dim[6];
    dim7 = extended_dim[7];
}

extern "C" {

void single_shave_tile(uint32_t tileParamsAddr) {
    const TileParams* tileParams = reinterpret_cast<const TileParams*>(tileParamsAddr);

    half* input = (half*)(tileParams->input.dataAddr);
    half* output = (half*)(tileParams->output.dataAddr);
    int32_t* repeats = (int32_t*)(tileParams->repeats.dataAddr);

    int32_t* input_dim = (int32_t*)(tileParams->input.dimsAddr);
    int32_t* out_shape = (int32_t*)(tileParams->output.dimsAddr);
    int32_t repeats_length = ((int32_t*)(tileParams->repeats.dimsAddr))[0];

    int32_t in_dim_length = (int32_t)tileParams->input.numDims;
    int32_t out_shape_size = (int32_t)tileParams->output.numDims;

    int extended_dim[MAX_TILE_DIMS];
    int extended_rep[MAX_TILE_DIMS];

    for (int i = 0; i < MAX_TILE_DIMS; i++)
        extended_dim[i] = extended_rep[i] = 1;

    for (int i = 0; i < in_dim_length; i++)
        extended_dim[i + (MAX_TILE_DIMS - in_dim_length)] = input_dim[i];
    for (int i = 0; i < repeats_length; i++)
        extended_rep[i + (MAX_TILE_DIMS - repeats_length)] = repeats[i];

    int32_t output_size = 1;

    for (int i = 0; i < MAX_TILE_DIMS; i++) {
        output_size *= extended_dim[i] * extended_rep[i];
    }

    int32_t inDimW, inDimH, inDimC, inDimN, repDimW, repDimH, repDimC, repDimN;
    if ((tileParams->input.dimsOrder == 0x1234) || (tileParams->input.dimsOrder == 0x123)) {
        // NCHW / CHW
        reorgDimsOrder(inDimW, inDimH, inDimC, inDimN, extended_dim);
        reorgDimsOrder(repDimW, repDimH, repDimC, repDimN, extended_rep);

    } else if ((tileParams->input.dimsOrder == 0x321)) {
        // WHC
        reorgDimsOrder(inDimN, inDimW, inDimH, inDimC, extended_dim);
        reorgDimsOrder(repDimN, repDimC, repDimH, repDimW, extended_rep);

    } else if ((tileParams->input.dimsOrder == 0x21)) {
        // CN
        reorgDimsOrder(inDimC, inDimN, inDimW, inDimH, extended_dim);
        reorgDimsOrder(repDimN, repDimC, repDimH, repDimW, extended_rep);

    } else {
        reorgDimsOrder(inDimW, inDimH, inDimC, inDimN, extended_dim);
        reorgDimsOrder(repDimN, repDimC, repDimH, repDimW, extended_rep);
    }

    for (int rep_inp4 = 1; rep_inp4 <= repDimN; rep_inp4++) {
        for (int inp4 = 1; inp4 <= inDimN; inp4++) {
            for (int rep_inp5 = 1; rep_inp5 <= repDimC; rep_inp5++) {
                for (int inp5 = 1; inp5 <= inDimC; inp5++) {
                    for (int rep_inp6 = 1; rep_inp6 <= repDimH; rep_inp6++) {
                        for (int inp6 = 1; inp6 <= inDimH; inp6++) {
                            for (int rep_inp7 = 1; rep_inp7 <= repDimW; rep_inp7++) {
                                memcpy(output, input, inDimW * bpp);
                                output += inDimW;
                            }
                            input += inDimW;
                        }
                        if (rep_inp6 < repDimH) {
                            input -= inDimH * inDimW;
                        }
                    }
                }
                if (rep_inp5 < repDimC) {
                    input -= inDimC * inDimH * inDimW;
                }
            }
        }
        if (rep_inp4 < repDimN) {
            input -= inDimN * inDimC * inDimH * inDimW;
        }
    }
}
}
}  // namespace shave_lib
}  // namespace nn
