/*
* {% copyright %}
*/
#include "sw_shave_res_manager.h"
#include <OsDrvSvu.h>

namespace nn {
namespace shave_lib {

uint32_t ShaveResource::cmxSliceAddr() const { return CMX_SLICE_0_BASE_ADR + (CMX_SLICE_SIZE * shaveID); }

} // namespace shave_lib
} // namespace nn
