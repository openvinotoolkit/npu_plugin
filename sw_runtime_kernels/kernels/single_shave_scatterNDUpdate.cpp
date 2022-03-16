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

#include <sw_tensor_ref.h>
#include <numeric>

#include <param_scatterNDUpdate.h>

using namespace sw_params;
using namespace nn;

namespace {

void dmaTransfer(const uint8_t *srcData_addr, uint8_t *dstData_addr, int size, DmaAlShave &dmaTask){
    const int max_transfer_size = DmaAlShave::max_transfer_size;
    for (int i = 0; i < size; i += max_transfer_size) {
        const int tail = size - i;
        const int length = std::min(tail, max_transfer_size);
        dmaTask.start(srcData_addr + i, dstData_addr + i, length);
        dmaTask.wait();
    }
}

}  // namespace

namespace nn {
namespace shave_lib {

extern "C" {

void singleShaveScatterNDUpdate(uint32_t lParamsAddr) {
    const ScatterNDUpdateParams* layerParams = reinterpret_cast<const ScatterNDUpdateParams*>(lParamsAddr);

    uint8_t* inputData_addr = reinterpret_cast<uint8_t*>(layerParams->input.dataAddr);
    uint8_t* indicesData_addr = reinterpret_cast<uint8_t*>(layerParams->indices.dataAddr);
    uint8_t* updatesData_addr = reinterpret_cast<uint8_t*>(layerParams->updates.dataAddr);
    uint8_t* outputData_addr = reinterpret_cast<uint8_t*>(layerParams->output.dataAddr);

    MemRefData inputData = layerParams->input;
    MemRefData indicesData = layerParams->indices;
    MemRefData updatesData = layerParams->updates;
    MemRefData outputData = layerParams->output;

    int32_t* inputDims = reinterpret_cast<int32_t*>(inputData.dimsAddr);
    int32_t* indicesDims = reinterpret_cast<int32_t*>(indicesData.dimsAddr);

    int bpp = getBpp(static_cast<sw_params::DataType>(inputData.dataType));
    int last_idx_dim = indicesDims[0];


    DmaAlShave dmaTask;
    int coord[last_idx_dim];
    size_t input_data_dim_pading[inputData.numDims];
    int update_chunk_size = inputData.numDims - last_idx_dim < 0 ? 0 : inputData.numDims - last_idx_dim;;
    const auto update_el_number = subspace::getTotal(&inputDims[0], update_chunk_size);

    dmaTransfer(inputData_addr, outputData_addr, bpp * subspace::getTotal(inputDims, inputData.numDims), dmaTask);

    input_data_dim_pading[inputData.numDims - 1] = 1;
    for (size_t i = inputData.numDims - 1; i != 0; --i) {
        input_data_dim_pading[i - 1] = input_data_dim_pading[i] * inputDims[inputData.numDims - 1 - i];
    };

    const auto num_of_updates = subspace::getTotal(&indicesDims[1], indicesData.numDims - 1);
    for (int i = 0; i != num_of_updates; ++i) {
        const auto indices_coord = reinterpret_cast<const int*>(indicesData_addr) + i * last_idx_dim;
        for (int j = 0; j < last_idx_dim; j++)
            coord[j] = indices_coord[j];
        const auto out_index = std::inner_product(&coord[0], &coord[last_idx_dim], &input_data_dim_pading[0], 0);

        const auto update_data = updatesData_addr + i * update_el_number * bpp;
        const auto update_mem_size = update_el_number * bpp;
        dmaTransfer(update_data, outputData_addr + out_index * bpp, update_mem_size, dmaTask);
    }
}
}
}  // namespace shave_lib
}  // namespace nn
