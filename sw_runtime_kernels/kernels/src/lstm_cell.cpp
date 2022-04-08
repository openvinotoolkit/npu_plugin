//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <math.h>
#include <param_lstm_cell.h>

using namespace sw_params;

#define SUM_VEC_8(_a, _b) __builtin_shave_vau_add_f16_rr((_a), (_b))
#define MUL_VEC_8(_a, _b) __builtin_shave_vau_mul_f16_rr((_a), (_b))
#define MIN(_a, _b) __builtin_shave_cmu_min_i32_rr_int((_a), (_b))

// #define ACCURACY_half

enum : int { n_gates = 4 };

static half logistic(half x) {
#ifdef ACCURACY_FP16
    return 1.0f / (1.0f + __builtin_shave_sau_exp2_f16_l_r(-x));
#else
    return (1.0f / (1.0f + expf((float)(-x))));
#endif
}

static half _tanhf(half x) {
#ifdef ACCURACY_FP16
    return (1 - (2.0f / (__builtin_shave_sau_exp2_f16_l_r(2 * x) + 1)));
#else
    return tanhf((float)(x));
#endif
}

static void LSTMCellBody(int input_size, int state_size,
                         // weights
                         const half* weights_layer, const half* weights_iter_h, const half* bias,
                         // input
                         const half* src_layer, const half* src_iter_h, const half* src_iter_c,

                         int outputs_number,
                         // output
                         half* h_dst, half* c_dst, half* last_h_dst) {
    const int offset_multiplier[] = {0, 1, 3, 2};

    const half* psrc_layer = src_layer;
    const half* phidden_state_src = src_iter_h;
    const half* pcell_state_src = src_iter_c;

    const half* pweights = weights_layer;
    const half* pweights_hidden = weights_iter_h;
    const half* pbias = bias;

    half* phidden_state_dst = h_dst;
    half* pcell_state_dst = c_dst;
    half* plast_hidden_state_dst = last_h_dst;

    int row0 = 0;
    int row_num = state_size;

    int src_layer_buff_size = input_size;
    int weights_buff_size = n_gates * input_size;
    int hidden_state_src_buff_size = state_size;
    int weights_hidden_buff_size = n_gates * state_size;
    int bias_buff_size = n_gates * row_num;

    const half* psrc_layer_buff;
    const half* pweights_buff;
    const half* phidden_state_src_buff;
    const half* pweights_hidden_buff;
    const half* pbias_buff;

    int weights_buff_stride = 0;
    int weights_hidden_buff_stride = 0;
    int bias_buff_offset = 0;
    int bias_buff_stride = 0;
    int dma_rows_max = 1;

    psrc_layer_buff = psrc_layer;
    pweights_buff = pweights;
    phidden_state_src_buff = phidden_state_src;
    pweights_hidden_buff = pweights_hidden;
    pbias_buff = pbias;

    bias_buff_offset = 0;
    bias_buff_stride = state_size;

    weights_buff_stride = state_size * input_size;
    weights_hidden_buff_stride = state_size * state_size;

    int bias_offset_map_0 = 0;
    int bias_offset_map_1 = 0;
    int bias_offset_map_2 = 0;
    int bias_offset_map_3 = 0;

    bias_offset_map_0 = offset_multiplier[0] * bias_buff_stride - bias_buff_offset;
    bias_offset_map_1 = offset_multiplier[1] * bias_buff_stride - bias_buff_offset;
    bias_offset_map_2 = offset_multiplier[2] * bias_buff_stride - bias_buff_offset;
    bias_offset_map_3 = offset_multiplier[3] * bias_buff_stride - bias_buff_offset;

    for (int r = row0; r < row0 + row_num; r += dma_rows_max) {
        int dma_rows_step = MIN(dma_rows_max, row0 + row_num - r);
        for (int step_row = 0; step_row < dma_rows_step; step_row++) {
            half8 sum0 = (half8)0;
            half8 sum1 = (half8)0;
            half8 sum2 = (half8)0;
            half8 sum3 = (half8)0;

            half8 sum4 = (half8)0;
            half8 sum5 = (half8)0;
            half8 sum6 = (half8)0;
            half8 sum7 = (half8)0;

            int weights_offset_map0_0 = (offset_multiplier[0] * dma_rows_step + step_row) * weights_buff_stride;
            int weights_offset_map1_0 = (offset_multiplier[1] * dma_rows_step + step_row) * weights_buff_stride;
            int weights_offset_map2_0 = (offset_multiplier[2] * dma_rows_step + step_row) * weights_buff_stride;
            int weights_offset_map3_0 = (offset_multiplier[3] * dma_rows_step + step_row) * weights_buff_stride;

            int weights_offset_map0_1 = (offset_multiplier[0] * dma_rows_step + step_row) * weights_hidden_buff_stride;
            int weights_offset_map1_1 = (offset_multiplier[1] * dma_rows_step + step_row) * weights_hidden_buff_stride;
            int weights_offset_map2_1 = (offset_multiplier[2] * dma_rows_step + step_row) * weights_hidden_buff_stride;
            int weights_offset_map3_1 = (offset_multiplier[3] * dma_rows_step + step_row) * weights_hidden_buff_stride;

            int k = 0;
            for (; k <= input_size - 8; k += 8) {
                half8 val0_0 = MUL_VEC_8(*((half8*)(pweights_buff + k + weights_offset_map0_0)),
                                         *((half8*)(psrc_layer_buff + k)));
                half8 val0_1 = MUL_VEC_8(*((half8*)(pweights_buff + k + weights_offset_map1_0)),
                                         *((half8*)(psrc_layer_buff + k)));
                half8 val0_2 = MUL_VEC_8(*((half8*)(pweights_buff + k + weights_offset_map2_0)),
                                         *((half8*)(psrc_layer_buff + k)));
                half8 val0_3 = MUL_VEC_8(*((half8*)(pweights_buff + k + weights_offset_map3_0)),
                                         *((half8*)(psrc_layer_buff + k)));

                sum0 = SUM_VEC_8(sum0, val0_0);
                sum1 = SUM_VEC_8(sum1, val0_1);
                sum2 = SUM_VEC_8(sum2, val0_2);
                sum3 = SUM_VEC_8(sum3, val0_3);
            }
            int k1 = 0;
            for (; k1 <= state_size - 8; k1 += 8) {
                half8 val1_0 = MUL_VEC_8(*((half8*)(pweights_hidden_buff + k1 + weights_offset_map0_1)),
                                         *((half8*)(phidden_state_src_buff + k1)));
                half8 val1_1 = MUL_VEC_8(*((half8*)(pweights_hidden_buff + k1 + weights_offset_map1_1)),
                                         *((half8*)(phidden_state_src_buff + k1)));
                half8 val1_2 = MUL_VEC_8(*((half8*)(pweights_hidden_buff + k1 + weights_offset_map2_1)),
                                         *((half8*)(phidden_state_src_buff + k1)));
                half8 val1_3 = MUL_VEC_8(*((half8*)(pweights_hidden_buff + k1 + weights_offset_map3_1)),
                                         *((half8*)(phidden_state_src_buff + k1)));

                sum4 = SUM_VEC_8(sum4, val1_0);
                sum5 = SUM_VEC_8(sum5, val1_1);
                sum6 = SUM_VEC_8(sum6, val1_2);
                sum7 = SUM_VEC_8(sum7, val1_3);
            }
            half sum0_0 = (half)0;
            half sum0_1 = (half)0;
            half sum0_2 = (half)0;
            half sum0_3 = (half)0;

            half sum1_0 = (half)0;
            half sum1_1 = (half)0;
            half sum1_2 = (half)0;
            half sum1_3 = (half)0;

            for (; k < input_size; k++) {
                sum0_0 += pweights_buff[k + weights_offset_map0_0] * psrc_layer_buff[k];
                sum0_1 += pweights_buff[k + weights_offset_map1_0] * psrc_layer_buff[k];
                sum0_2 += pweights_buff[k + weights_offset_map2_0] * psrc_layer_buff[k];
                sum0_3 += pweights_buff[k + weights_offset_map3_0] * psrc_layer_buff[k];
            }
            for (; k1 < state_size; k1++) {
                sum1_0 += pweights_hidden_buff[k1 + weights_offset_map0_1] * phidden_state_src_buff[k1];
                sum1_1 += pweights_hidden_buff[k1 + weights_offset_map1_1] * phidden_state_src_buff[k1];
                sum1_2 += pweights_hidden_buff[k1 + weights_offset_map2_1] * phidden_state_src_buff[k1];
                sum1_3 += pweights_hidden_buff[k1 + weights_offset_map3_1] * phidden_state_src_buff[k1];
            }
            sum0_0 += __builtin_shave_sau_sumx_f16_r(sum0);
            sum0_1 += __builtin_shave_sau_sumx_f16_r(sum1);
            sum0_2 += __builtin_shave_sau_sumx_f16_r(sum2);
            sum0_3 += __builtin_shave_sau_sumx_f16_r(sum3);

            sum1_0 += __builtin_shave_sau_sumx_f16_r(sum4);
            sum1_1 += __builtin_shave_sau_sumx_f16_r(sum5);
            sum1_2 += __builtin_shave_sau_sumx_f16_r(sum6);
            sum1_3 += __builtin_shave_sau_sumx_f16_r(sum7);

            half b_0 = 0;
            half b_1 = 0;
            half b_2 = 0;
            half b_3 = 0;
            // adding bias
            b_0 = pbias_buff[(r + step_row) + bias_offset_map_0];
            b_1 = pbias_buff[(r + step_row) + bias_offset_map_1];
            b_2 = pbias_buff[(r + step_row) + bias_offset_map_2];
            b_3 = pbias_buff[(r + step_row) + bias_offset_map_3];

            half v_0 = sum0_0 + sum1_0 + b_0;
            half v_1 = sum0_1 + sum1_1 + b_1;
            half v_2 = sum0_2 + sum1_2 + b_2;
            half v_3 = sum0_3 + sum1_3 + b_3;

            // activation step
            v_0 = logistic(v_0);
            v_1 = logistic(v_1);
            v_2 = logistic(v_2);
            v_3 = _tanhf(v_3);

            float res = v_0 * pcell_state_src[r + step_row] + v_1 * v_3;

            if (pcell_state_dst)
                pcell_state_dst[r + step_row] = (half)res;

            phidden_state_dst[r + step_row] = v_2 * _tanhf(res);

            if (outputs_number == 3 && plast_hidden_state_dst)
                plast_hidden_state_dst[r + step_row] = phidden_state_dst[r + step_row];
        }
    }
}
static int getTotal(const int32_t subspaceDims[], int nDims) {
    int totalSubspaces = 1;
    for (int i = 0; i < nDims; i++) {
        totalSubspaces *= subspaceDims[i];
    }
    return totalSubspaces;
}
extern "C" {
void lstm_cell(struct LSTMCellParams* lParams) {
    const int n_gates = 4;

    const half* inputData = reinterpret_cast<const half*>(lParams->inputData.dataAddr);
    const half* initialHiddenState = reinterpret_cast<const half*>(lParams->initialHiddenState.dataAddr);
    const half* initialCellState = reinterpret_cast<const half*>(lParams->initialCellState.dataAddr);
    const half* weights = reinterpret_cast<const half*>(lParams->weights.dataAddr);
    const half* biases = reinterpret_cast<const half*>(lParams->biases.dataAddr);
    half* outputHiddenState = reinterpret_cast<half*>(lParams->outputHiddenState.dataAddr);
    half* outputCellState = reinterpret_cast<half*>(lParams->outputCellState.dataAddr);
    half* pOutputLastHiddenState = nullptr;  // This kernel version is for LSTMCell only

    int32_t* inputDataDims = (int32_t*)(lParams->inputData.dimsAddr);
    int32_t* outputHiddenStateDims = (int32_t*)(lParams->outputHiddenState.dimsAddr);
    int32_t* outputCellStateDims = (int32_t*)(lParams->outputCellState.dimsAddr);

    uint32_t inputDataNumDims = lParams->inputData.numDims;
    uint32_t outputHiddenStateNumDims = lParams->outputHiddenState.numDims;
    uint32_t outputCellStateNumDims = lParams->outputCellState.numDims;

    int RNNForward = (int)lParams->RNNForward;
    int nCells = (int)lParams->nCells;
    int nBatches = (int)lParams->nBatches;
    int useCellState = (int)lParams->useCellState;
    int outputsNumber = (int)lParams->outputsNumber;

    int inputDataNumElements = getTotal(inputDataDims, inputDataNumDims);
    int outputHiddenStateNumElements = getTotal(outputHiddenStateDims, outputHiddenStateNumDims);

    const int inputSize = inputDataNumElements / nBatches / nCells;
    const int stateSize = outputHiddenStateNumElements / nBatches / nCells;

    int cellStart = RNNForward ? 0 : (nCells - 1);
    int cellStride = RNNForward ? 1 : (-1);

    const half* weights_hidden = reinterpret_cast<const half*>(lParams->recurrenceWeights.dataAddr);

    for (int b = 0; b < nBatches; b++) {
        const half* pInputHiddenStateTmp = initialHiddenState + b * stateSize;
        const half* pInputCellStateTmp = initialCellState + b * stateSize;
        half* pOutputCellStateTmp = nullptr;

        for (int c = 0; c < nCells; c++) {
            if (c == nCells - 1)
                pOutputCellStateTmp = (outputCellState == nullptr) ? nullptr : (outputCellState + b * stateSize);

            int cellInd = cellStart + cellStride * c;
            LSTMCellBody(inputSize, stateSize, weights, weights_hidden, biases,
                         // inputs
                         inputData + inputSize * cellInd + inputSize * nCells * b, pInputHiddenStateTmp,
                         pInputCellStateTmp,

                         outputsNumber,
                         // outputs
                         outputHiddenState + stateSize * cellInd + stateSize * nCells * b, pOutputCellStateTmp,
                         (c == nCells - 1) ? pOutputLastHiddenState + stateSize * cellInd + stateSize * nCells * b
                                           : nullptr);

            pInputHiddenStateTmp = outputHiddenState + stateSize * cellInd + stateSize * nCells * b;
            if (pOutputCellStateTmp)
                pInputCellStateTmp = pOutputCellStateTmp;
        }
    }
}
}
