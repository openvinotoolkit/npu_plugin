// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0

#include <moviVectorTypes.h>
#include <math.h>
#include <param_reorder.h>

#include <mvSubspaces.h>

extern "C"
void reorder_fp16(const struct ReorderParams *lParams) {

    const u8* inData = (const u8*)(lParams->input.dataAddr); // 0x1F000000
    u8* outData = (u8*)(lParams->output.dataAddr); // 0x1F004000

    const int ndims = lParams->input.numDims;

    const int32_t* inDims = (int32_t *)(lParams->input.dimsAddr);
    const int32_t* outDims = (int32_t *)(lParams->output.dimsAddr);

    const int64_t* perm64 = (int64_t *)(lParams->perm);

    const uint64_t* inStrides64 = (uint64_t *)(lParams->input.stridesAddr);
    const uint64_t* outStrides64 = (uint64_t *)(lParams->output.stridesAddr);

    int32_t inStrides[MAX_ND_DIMS] = {};
    int32_t outStrides[MAX_ND_DIMS] = {};
    int32_t perm[MAX_ND_DIMS] = {};

    for (int i = 0; i < ndims; ++i)
    {
        inStrides[i] = int32_t(inStrides64[i] / 8);
        outStrides[i] = int32_t(outStrides64[i] / 8);
        perm[i] = (int32_t)(perm64[i]);
    }

    const int total = subspace::getTotal(inDims, ndims);

    int32_t in[MAX_ND_DIMS] = {};
    subspace::getCoord(0, inDims, ndims, in);

    for (int current = 0; current < total; ++current)
    {
        int32_t out[MAX_ND_DIMS] = {};
        subspace::permuteArray(in, perm, out, ndims);

        unsigned inOfs = subspace::getOffsetU8(in, inStrides, ndims);
        unsigned outOfs = subspace::getOffsetU8(out, outStrides, ndims);

        *(half*)(outData + outOfs) = *(const half*)(inData + inOfs);

        subspace::increment1Coord(in, inDims, ndims);
    }
}
