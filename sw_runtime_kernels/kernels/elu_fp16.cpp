// {% copyright %}

#ifdef CONFIG_HAS_LRT_SRCS
#include <nn_log.h>
#else
#define nnLog(level, ...)
#endif
#include <math.h>
#include <param_elu.h>
#include <moviVectorTypes.h>
#ifdef CONFIG_TARGET_SOC_3720
#include <dma_shave_nn.h>
#else
#include <dma_shave.h>
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

void elu_fp16(uint32_t lParamsAddr) {

    const EluParams * lParams = (const EluParams*)lParamsAddr;

    half* p_act_data = (half*)(lParams->input.dataAddr); // 0x1F000000
    half* p_act_out = (half*)(lParams->output.dataAddr); // 0x1F004000

    const EluParams * layerParams = reinterpret_cast<const EluParams *>(lParams);
    const half alpha = layerParams->alpha;
    int32_t *pDims = (int32_t *)(lParams->input.dimsAddr);

    int32_t nElements = 1;
    int32_t i = 0;

    for (i = 0; i!= lParams->input.numDims; i++ ) {
        // TODO: where pointers patch should be???
        // TODO: check overflow
        nElements *=  pDims[i];
    }

    for (uint32_t e = 0; e < nElements; ++e) {
        half min = MIN(*p_act_data, 0.0f);
	half max = MAX(*p_act_data, 0.0f);
        *p_act_out++ = max + 0.3f * (exp((double)min) - 1.0f);
	p_act_data++;
    }
}
}
} // namespace shave_lib
} // namespace nn
