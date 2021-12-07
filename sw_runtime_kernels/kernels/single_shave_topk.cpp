// {% copyright %}

#include <moviVectorTypes.h>
#include <mvSubspaces.h>
#include <math.h>
#include <stdio.h>
#include <param_topk.h>

using namespace sw_params;

extern "C" {

void singleShaveTopK(uint32_t lParamsAddr) {

    half* p_act_input = (half*)(reinterpret_cast<TopKParams*>(lParamsAddr)->input.dataAddr);
    int32_t k = *(int32_t*)(reinterpret_cast<TopKParams*>(lParamsAddr)->k.dataAddr);
    half* p_act_value = (half*)(reinterpret_cast<TopKParams*>(lParamsAddr)->value.dataAddr);
    int32_t* p_act_index = (int32_t*)(reinterpret_cast<TopKParams*>(lParamsAddr)->index.dataAddr);

    const TopKParams* lParams = reinterpret_cast<const TopKParams *>(lParamsAddr);

    for (int i = 0; i < 2 * 3 * 4; i++) {
        *p_act_value = *p_act_input;
        if (i == 0) {
            *p_act_value = 1234;
        }
        p_act_input++;
        p_act_value++;
    }

    // Case dimension NCHW 4, 3, 2 (0,0,0)=1, (3,2,1)=24
    int32_t axis = 1;
    int32_t mode = 0; // max: 0, min: 1
    int32_t sort = 0; // value: 0, index: 1

    int32_t numInputDims = (int32_t)lParams->input.numDims;
    int32_t* pInputDims = (int32_t*)(lParams->input.dimsAddr);
    int64_t* pInputStrides = (int64_t*)(lParams->input.stridesAddr);

    int32_t numValueDims = (int32_t)lParams->value.numDims;
    int32_t* pValueDims = (int32_t*)(lParams->value.dimsAddr);
    int64_t* pValueStrides = (int64_t*)(lParams->value.stridesAddr);

    int32_t numIndexDims = (int32_t)lParams->index.numDims;
    int32_t* pIndexDims = (int32_t*)(lParams->index.dimsAddr);
    int64_t* pIndexStrides = (int64_t*)(lParams->index.stridesAddr);

    p_act_index[0] = k;

    // input dim number
    p_act_index[1] = numInputDims; // 3
    // input dims
    p_act_index[2] = pInputDims[0]; // 4
    p_act_index[3] = pInputDims[1]; // 3
    p_act_index[4] = pInputDims[2]; // 2
    p_act_index[5] = pInputDims[3]; // -1
    // input strides
    p_act_index[6] = pInputStrides[0] / CHAR_BIT; // 2
    p_act_index[7] = pInputStrides[1] / CHAR_BIT; // 8
    p_act_index[8] = pInputStrides[2] / CHAR_BIT;; // 24
    p_act_index[9] = pInputStrides[3] / CHAR_BIT;; // -842150451 or 0

//    // value dim number
//    p_act_index[10] = numValueDims; // 3
//    // value dims
//    p_act_index[11] = pValueDims[0]; // 4
//    p_act_index[12] = pValueDims[1]; // 3
//    p_act_index[13] = pValueDims[2]; // 2
//    p_act_index[14] = pValueDims[3]; // -1
//    // value strides
//    p_act_index[15] = pValueStrides[0] / CHAR_BIT; // 2
//    p_act_index[16] = pValueStrides[1] / CHAR_BIT; // 8
//    p_act_index[17] = pValueStrides[2] / CHAR_BIT; // 24
//    p_act_index[18] = pValueStrides[3] / CHAR_BIT; // -842150451 or 0

    // index dim number
    p_act_index[10] = numIndexDims; // 3
    // index dims
    p_act_index[11] = pIndexDims[0]; // 4
    p_act_index[12] = pIndexDims[1]; // 3
    p_act_index[13] = pIndexDims[2]; // 2
    p_act_index[14] = pIndexDims[3]; // -1
    // index strides
    p_act_index[15] = pIndexStrides[0] / CHAR_BIT; // 4
    p_act_index[16] = pIndexStrides[1] / CHAR_BIT; // 16
    p_act_index[17] = pIndexStrides[2] / CHAR_BIT; // 48
    p_act_index[18] = pIndexStrides[3] / CHAR_BIT; // -842150451 or 0

}
}
