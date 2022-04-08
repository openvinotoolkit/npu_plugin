//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

namespace {

#include <eltwise_base.hpp>
#include <cmath>

inline half eltwise_scl_fp16(half a, half b){
    half div = static_cast<half>(std::floor((float)(a / b)));
    return static_cast<half>(a - b * div);
}

} // namespace

namespace nn {
namespace shave_lib {

extern "C" {

void eltwise_floor_mod_fp16(const struct EltwiseParams *lParams) {
    eltwise_fp16(lParams);
}

}
}  // namespace shave_lib
}  // namespace nn
