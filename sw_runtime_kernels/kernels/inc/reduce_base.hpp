//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <param_reduce.h>
#include <moviVectorConvert.h>
#include <cmath>

using namespace sw_params;
using namespace subspace;

#define F16_MAX 65504.0f
#define REDUCE_MAX_ND_DIMS 4
#define BYTE_SIZE 8

template <class DataType>
class LogicalAnd;

template <class DataType>
class L1;

template <class DataType>
class Sum;

template <class DataType>
class Mean;

template <class DataType>
class Max;

template <class DataType>
class Min;

template <class DataType>
class LogicalOr;

template <class DataType>
class L2;

template <class DataType>
class Prod;

static uint32_t axes2mask(int N, int K, int32_t* axes) {
    uint32_t mask = 0;
    for (int j = 0; j < K; ++j) {
        int32_t i = axes[j];
        if ((i >= 0) && (i < N))
            mask |= (1 << i);
    }
    return mask;
}

static void split(int N, uint32_t mask, int32_t* D, int32_t* DR, int32_t* DI) {
    int jr = 0, ji = 0;
    for (int i = 0; i < N; ++i) {
        if (mask & (1 << i)){
            DR[jr++] = D[i];
        }
        else
            DI[ji++] = D[i];
    }
}

static void merge(int N, uint32_t mask, int32_t* DR, int32_t* DI, int32_t* D) {
    int jr = 0, ji = 0;
    for (int i = 0; i < N; ++i) {
        if (mask & (1 << i))
            D[i] = DR[jr++];
        else
            D[i] = DI[ji++];
    }
}

static void fill(int32_t* Z, int K, int val) {
    for (int j = 0; j < K; ++j)
        Z[j] = val;
}

template<typename DataType, class Op>
static void fullReduce(MemRefData in, MemRefData out, Op op) {
    DataType* inputData  = reinterpret_cast<DataType*>(in.dataAddr);
    DataType* outputData = reinterpret_cast<DataType*>(out.dataAddr);
    op.init();
    for(int i = 0; i < getTotal((const int32_t*)in.dimsAddr, in.numDims); i++){
        op.accumulate(inputData[i]);
    }
    outputData[0] = op.result();
}

void getStridesInBytes(const int64_t* inStrides, int32_t* outStrides, int nDims){
    for(int i = 0; i < nDims; i++){
        outStrides[i] = inStrides[i] / BYTE_SIZE;
    }
}

void reorderDims(int32_t* dims, int nDims){
    for(int i = 0; i < nDims / 2; i++){
        int aux = dims[i];
        dims[i] = dims[nDims - i - 1];
        dims[nDims-i-1] = aux;
    }
}

template<typename DataType, class Op>
static void partReduce(MemRefData in, MemRefData axes, MemRefData out, bool keep_dims, Op op){
    const int N = in.numDims;
    const int K = ((int32_t*)axes.dimsAddr)[0];
    const int O = keep_dims ? N : N-K;

    DataType* inputData  = reinterpret_cast<DataType*>(in.dataAddr);
    DataType* outputData = reinterpret_cast<DataType*>(out.dataAddr);

    int32_t inputDims[REDUCE_MAX_ND_DIMS] = {0};
    for(int i = 0; i < N; i++){
        inputDims[i] = ((int32_t*)in.dimsAddr)[N - i - 1];
    }

    int32_t inputStrides[N];
    int32_t outputStrides[O];
    getStridesInBytes(reinterpret_cast<const int64_t*>(in.stridesAddr),  inputStrides,  N);
    getStridesInBytes(reinterpret_cast<const int64_t*>(out.stridesAddr), outputStrides, O);

    reorderDims(inputStrides, N);
    reorderDims(outputStrides, O);

    unsigned mask = axes2mask(N, K, reinterpret_cast<int32_t*>(axes.dataAddr));
    int32_t DR[REDUCE_MAX_ND_DIMS] = { 0 };
    int32_t DI[REDUCE_MAX_ND_DIMS] = { 0 };
    split(N, mask, inputDims, &DR[0], &DI[0]);
    int32_t ZR[REDUCE_MAX_ND_DIMS] = { 0 };
    fill(ZR, K, 0);

    int32_t di[REDUCE_MAX_ND_DIMS] = { 0 };
    int32_t dr[REDUCE_MAX_ND_DIMS] = { 0 };
    int32_t id[REDUCE_MAX_ND_DIMS] = { 0 };
    int32_t od[REDUCE_MAX_ND_DIMS] = { 0 };

    getCoord(0, DI, N-K, di);
    getCoord(0, DR, K,   dr);

    int DI_total = getTotal(DI, N - K);
    int DR_total = getTotal(DR, K);
    int k=0;
    for(int i = 0; i < DI_total; i++){ //DI
        op.init();
        for(int j = 0; j < DR_total; j++){ //DR
            merge(N, mask, dr, di, id);
            int in_idx = getOffsetU8(id, inputStrides, N) / op.getBpp();
            DataType input_element = inputData[in_idx];
            op.accumulate(input_element);
            increment1Coord(dr, DR, K);
        }
        if(keep_dims){
            merge(N, mask, ZR, di, od);
            int out_idx = getOffsetU8(od, outputStrides, N) / op.getBpp();
            outputData[out_idx] = op.result();
        }
        else{
            int out_idx = getOffsetU8(di, outputStrides, N-K) / op.getBpp();
            outputData[out_idx] = op.result();
        }
        increment1Coord(di, DI, N-K);
    }
}

template<typename DataType, class Op>
static void refReduce(MemRefData in, MemRefData axes, MemRefData out, bool keep_dims, Op op) {
    int N = in.numDims;
    int K = ((int32_t*)axes.dimsAddr)[0];

    if ((K <= 0) || (K >= N)) {
        if (K >= N)
            fullReduce<DataType, Op>(in, out, op);
    } else {
        partReduce<DataType, Op>(in, axes, out, keep_dims, op);
    }
}

template<typename DataType, class Op>
static void reduce(const struct ReduceParams *lParams, Op op) {
    int32_t*  axesData = reinterpret_cast<int32_t*>(lParams->axes.dataAddr);
    DataType* outputData = reinterpret_cast<DataType*>(lParams->output.dataAddr);

    int32_t* inputDims  = (int32_t*)(lParams->input.dimsAddr);
    int32_t* axesDims   = (int32_t*)(lParams->axes.dimsAddr);
    int32_t* outputDims = (int32_t*)(lParams->output.dimsAddr);

    uint32_t inputNumDims  = lParams->input.numDims;
    uint32_t axesNumDims   = lParams->axes.numDims;
    uint32_t outputNumDims = lParams->output.numDims;

    //Initialize output memory
    for(int i = 0; i < getTotal(outputDims, outputNumDims); i++){
        outputData[i] = (DataType)0.0f;
    }

    bool success;
    NDDims indices = orderNDToIndices(lParams->input.dimsOrder, success);

    //Make axes pozitive
    for(int i = 0; i < ((int32_t*)lParams->axes.dimsAddr)[0]; i++){
        axesData[i] = axesData[i] < (int32_t)0 ? axesData[i] + inputNumDims : axesData[i];
        axesData[i] = indices[axesData[i]];
    }

    //Process data
    refReduce<DataType, Op>(lParams->input, lParams->axes, lParams->output, lParams->keep_dims, op);
}
