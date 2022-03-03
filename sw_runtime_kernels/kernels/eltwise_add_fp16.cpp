//
// Copyright Intel Corporation.
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

#include <math.h>
#include <param_eltwise.h>

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

#define ELTWISE_VEC_OP __builtin_shave_vau_add_f16_rr
#define ELTWISE_FN(a,b) (a+b)
#include <eltwise_base.h>

ELTWISE_BINARY_OP(add_fp16);

}
}  // namespace shave_lib
}  // namespace nn
