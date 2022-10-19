//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

namespace {

#define ELTWISE_VEC_OP

#include <eltwise_base.hpp>

inline half eltwise_scl_fp16(half a, half b){
    return a * b;
}

inline half8 eltwise_vec_fp16(half8 a, half8 b){
    return __builtin_shave_vau_mul_f16_rr(a, b);
}

} // namespace

namespace nn {
namespace shave_lib {

extern "C" {

void eltwise_mul_fp16(const struct EltwiseParams *lParams) {
    eltwise_fp16(lParams);
}

}
}  // namespace shave_lib
}  // namespace nn
