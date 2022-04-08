//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

namespace {

#include <trigonometric_base.hpp>

inline half trigonometric_scl_fp16(half a) {
    return __builtin_sinf(a);
}

}  // namespace

namespace nn {
namespace shave_lib {

extern "C" {

void sin_fp16(const struct TrigonometricParams* lParams) {
    trigonometric_fp16(lParams);
}
}
}  // namespace shave_lib
}  // namespace nn
