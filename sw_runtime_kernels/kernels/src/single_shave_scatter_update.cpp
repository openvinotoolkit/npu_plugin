//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <dma_shave_nn.h>
#include <mvSubspaces.h>
#include <param_scatter_update.h>

using namespace sw_params;
using namespace subspace;

namespace {

void dmaTransfer(const uint8_t* srcData_addr, uint8_t* dstData_addr, int size, DmaAlShave& dmaTask) {
    const int max_transfer_size = DmaAlShave::max_transfer_size;
    for (int i = 0; i < size; i += max_transfer_size) {
        const int tail = size - i;
        const int length = std::min(tail, max_transfer_size);
        dmaTask.start(srcData_addr + i, dstData_addr + i, length);
        dmaTask.wait();
    }
}

u32 getBpp(DataType type) {
    u32 bpp = 0;
    switch (type) {
    case NN_INT16:
    case NN_FP16:
        bpp = 2;
        break;

    case NN_U8:
    case NN_I8:
        bpp = 1;
        break;

    case NN_INT32:
    case NN_FP32:
        bpp = 4;
        break;

    case NN_UNDEFINED:
    default:
        bpp = 0;
        break;
    }
    return bpp;
}

}  // namespace

namespace nn {
namespace shave_lib {

extern "C" {
void single_shave_scatter_update(uint32_t lParamsAddr) {
    const ScatterUpdateParams* lParams = reinterpret_cast<const ScatterUpdateParams*>(lParamsAddr);

    half* p_act_data = (half*)(lParams->data.dataAddr);
    half* p_act_indices = (half*)(lParams->indices.dataAddr);
    half* p_act_updates = (half*)(lParams->updates.dataAddr);
    half* p_act_out = (half*)(lParams->output.dataAddr);

    int64_t axis = lParams->axis;
    MemRefData inputData = lParams->data;

    UNUSED(axis);

    int64_t* pIndicesStrides = (int64_t*)(lParams->indices.stridesAddr);

    int32_t numIndicesDims = (int32_t)(lParams->indices.numDims);
    int32_t* pIndicesDims = (int32_t*)(lParams->indices.dimsAddr);

    int32_t numdataDims = (int32_t)(lParams->data.numDims);
    int32_t* pdataDims = (int32_t*)(lParams->data.dimsAddr);

    int32_t numUpdatesDims = (int32_t)(lParams->updates.numDims);
    int32_t* pUpdatesDims = (int32_t*)(lParams->updates.dimsAddr);

    int32_t numOutputDims = (int32_t)(lParams->output.numDims);
    int32_t* pOutputDims = (int32_t*)(lParams->output.dimsAddr);

    int32_t indicesStrides[SCATTER_UPDATE_MAX_SUPPORTED_DIMS];
    for (int i = 0; i < numIndicesDims; i++) {
        indicesStrides[i] = pIndicesStrides[i] / CHAR_BIT;
    }

    int32_t outputCoords[SCATTER_UPDATE_MAX_SUPPORTED_DIMS];
    getCoord(0, pOutputDims, numOutputDims, outputCoords);

    // prepare a copy of input data to output data
    int bpp = getBpp(static_cast<sw_params::DataType>(inputData.dataType));
    DmaAlShave dmaTask;
    dmaTransfer((uint8_t*)p_act_data, (uint8_t*)p_act_out, bpp * subspace::getTotal(pdataDims, numdataDims), dmaTask);

    // start iteration
    int32_t updatesCoords[numUpdatesDims];
    int32_t* indicesCoords = updatesCoords + numOutputDims - 1;
    getCoord(0, pUpdatesDims, numUpdatesDims, updatesCoords);
    int32_t length_updates = bpp * subspace::getTotal(pdataDims, numdataDims - 1);
    const int indicesTotal = getTotal(pIndicesDims, numIndicesDims);

    for (int i = 0; i < indicesTotal; i++) {
        int indicesOffset = getOffsetU8(indicesCoords, indicesStrides, numIndicesDims);
        const int indices_value = *reinterpret_cast<const int32_t*>((uint8_t*)p_act_indices + indicesOffset);
        outputCoords[numOutputDims - 1] = indices_value;
        dmaTransfer((uint8_t*)p_act_updates + i * length_updates, (uint8_t*)p_act_out + indices_value * length_updates,
                    length_updates, dmaTask);
        subspace::increment1Coord(indicesCoords, pIndicesDims, numIndicesDims);
    }
}
}
}  // namespace shave_lib
}  // namespace nn
