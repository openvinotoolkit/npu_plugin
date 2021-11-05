// {% copyright %}

#include <sw_layer.h>

#include <sw_shave_res_manager.h>
#include <nn_log.h>
#include <mvSubspaces.h>
#include <param_softmax.h>

#include <mv_types.h>
#include <math.h>
#include <moviVectorTypes.h>
#include <moviVectorConvert.h>

#include <svuCommonShave.h>

#ifdef CONFIG_TARGET_SOC_3720
#include <dma_shave_params_nn.h>
#else
#include <dma_shave_params.h>
#endif

#include "param_mvn.h"

using namespace nn;
using namespace shave_lib;
using namespace sw_params;

namespace {

using namespace subspace;

}  // namespace


using namespace subspace;

namespace nn {
namespace shave_lib {

extern "C" {
void mvn(uint32_t lParams) {
    const MvnParams * layerParams = reinterpret_cast<const MvnParams *>(lParams);
}
}

} // namespace shave_lib
} // namespace nn
