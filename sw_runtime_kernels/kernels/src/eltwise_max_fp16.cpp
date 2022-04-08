//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

namespace {

#define ELTWISE_VEC_OP

#include <eltwise_base.hpp>

inline half eltwise_scl_fp16(half a, half b){
    return __builtin_shave_cmu_max_f16_rr_half(a, b);
}

inline half8 eltwise_vec_fp16(half8 a, half8 b){
    return __builtin_shave_cmu_max_f16_rr_half8(a, b);
}

} // namespace

namespace nn {
namespace shave_lib {

extern "C" {

void eltwise_max_fp16(const struct EltwiseParams *lParams) {
    eltwise_fp16(lParams);
}

}
}  // namespace shave_lib
}  // namespace nn
