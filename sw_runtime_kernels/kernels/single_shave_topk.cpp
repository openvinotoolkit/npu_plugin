// {% copyright %}

#include <moviVectorTypes.h>
#include <math.h>
#include <stdio.h>
#include <param_topk.h>

using namespace sw_params;

extern "C" {

void singleShaveTopK(uint32_t lParamsAddr) {

    half* p_act_input = (half*)(reinterpret_cast<TopKParams*>(lParamsAddr)->input.dataAddr);
    half* p_act_value = (half*)(reinterpret_cast<TopKParams*>(lParamsAddr)->value.dataAddr);

    const TopKParams* lParams = reinterpret_cast<const TopKParams *>(lParamsAddr);

    for (int i = 0; i < 2 * 3 * 4; i++) {
        *p_act_value = *p_act_input;
        if (i == 0) {
            *p_act_value = 1234;
        }
        p_act_input++;
        p_act_value++;
    }

}
}
