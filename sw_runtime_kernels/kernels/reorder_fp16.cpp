// {% copyright %}

#include <moviVectorTypes.h>
#include <math.h>
#include <param_reorder.h>

#include <mvSubspaces.h>

//#include <stdio.h>

extern "C"
void reorder_fp16(const struct ReorderParams *lParams) {

    const u8* inData = (const u8*)(lParams->input.dataAddr); // 0x1F000000
    u8* outData = (u8*)(lParams->output.dataAddr); // 0x1F004000

    const int ndims = lParams->input.numDims;

    const int32_t* inDims = (int32_t *)(lParams->input.dimsAddr);
    const int32_t* outDims = (int32_t *)(lParams->output.dimsAddr);

    const int32_t* perm = (int32_t *)(lParams->perm);

    const uint64_t* inStrides64 = (uint64_t *)(lParams->input.stridesAddr);
    const uint64_t* outStrides64 = (uint64_t *)(lParams->output.stridesAddr);

    int32_t inStrides[MAX_ND_DIMS] = {};
    int32_t outStrides[MAX_ND_DIMS] = {};

    for (int i = 0; i < ndims; ++i)
    {
        inStrides[i] = int32_t(inStrides64[i] / 8);
        outStrides[i] = int32_t(outStrides64[i] / 8);
    }

#if 0
    printf("inDims     = %d %d %d\n", inDims[2], inDims[1], inDims[0]);
    printf("inStrides  = %d %d %d\n", inStrides[2], inStrides[1], inStrides[0]);
    printf("outDims    = %d %d %d\n", outDims[2], outDims[1], outDims[0]);
    printf("outStrides = %d %d %d\n", outStrides[2], outStrides[1], outStrides[0]);
    printf("perm       = %d %d %d\n", perm[2], perm[1], perm[0]);
#endif

//    int32_t nElements = 1;
//    int32_t i = 0;
//    half act = 0;

    const int total = subspace::getTotal(inDims, ndims);

#if 0
    for (int i = 0; i < total; ++i)
    {
        printf("# inData = %f\n", float(((const half*)inData)[i]));
    }
#endif

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

#if 0
    for (int i = 0; i < total; ++i)
    {
        ((half*)outData)[i] = ((const half*)inData)[i];
    }
#endif

//            const SingleTest* test = m_currentTest;
//            const int ndims = m_inputTensor.ndims();
//            m_inputTensor.forEach(false, [&](const MemoryDims& in)
//            {
//                MemoryDims out;
//                permuteArray(in.dims, test->customLayerParams.layerParams, out.dims, ndims);
//                m_referenceOutputTensor.at(out) = m_inputTensor.at(in);
//            });

//    for (i = 0; i!= lParams->input.numDims; i++ ) {
//        // TODO: check overflow
//        nElements *=  inDims[i];
//    }
//
//    for (uint32_t e = 0; e < nElements; ++e) {
//        act = *inData++ * -1.0f;
//        act = 1.0f + expf(act);
//        act = 1.0f / act;
//        *outData++ = (half)act;
//    }
}
