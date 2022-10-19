//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

namespace {

#include <eltwise_base.hpp>

inline half eltwise_scl_fp16(half a, half b){
    return a / b;
}

} // namespace

namespace nn {
namespace shave_lib {

extern "C" {

void eltwise_div_fp16(const struct EltwiseParams *lParams) {
    eltwise_fp16(lParams);
}

}
}  // namespace shave_lib
}  // namespace nn
