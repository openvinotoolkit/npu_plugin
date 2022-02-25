//
// Copyright Intel Corporation.
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

#include <param_broadcast.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

void broadcast(uint32_t lParamsAddr) {
    const BroadcastParams* lParams = (const BroadcastParams*)lParamsAddr;

    half* input = (half*)(lParams->input.dataAddr);
    half* output = (half*)(lParams->output.dataAddr);
    // std::vector<int32_t> broadcast_axes = (std::vector<int32_t>)lParams->axesMapping;

    const BroadcastParams* layerParams = reinterpret_cast<const BroadcastParams*>(lParams);
    int32_t* pDims_in = (int32_t*)(lParams->input.dimsAddr);
    int32_t* pDims_out = (int32_t*)(lParams->output.dimsAddr);

    int32_t nElements_in = 1;
    int32_t nElements_out = 1;

    for (uint32_t i = 0; i != lParams->input.numDims; i++) {
        nElements_in *= pDims_in[i];
    }
    for (uint32_t i = 0; i != lParams->output.numDims; i++) {
        nElements_out *= pDims_out[i];
    }

    // const auto output_rank = MAX(nElements_in, nElements_out);
    // half* adjusted_in_shape = input;
    // for (const auto& axis : broadcast_axes) {
    //     if (nElements_in < output_rank) {
    //         adjusted_in_shape.insert(adjusted_in_shape[0] + axis, 1);
    //     }
    // }

    // half* adjusted_out_shape = output;
    // adjusted_out_shape.insert(adjusted_out_shape[0], output_rank - nElements_out, 1);
}
}
}  // namespace shave_lib
}  // namespace nn
