//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/tensor/dtype/dtype_registry.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"


namespace mv
{

    // Float16 is actually treated as Integer type
    MV_REGISTER_DTYPE(BFloat16)
    .setIsDoubleType(false)
    .setSizeInBits(16);
}
