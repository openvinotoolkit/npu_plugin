//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

namespace {

#include <eltwise_base.hpp>

inline half eltwise_scl_fp16(half a, half b){
    static const half hOne = static_cast<half>( 1.0 );
    static const half hZero = static_cast<half>( 0.0 );
    return static_cast<half>(((a!= hZero) ^ (b!= hZero)) ? hOne : hZero); 
}

} // namespace

namespace nn {
namespace shave_lib {

extern "C" {

void eltwise_logical_xor_fp16(const struct EltwiseParams *lParams) {
    eltwise_fp16(lParams);
}

}
}  // namespace shave_lib
}  // namespace nn
