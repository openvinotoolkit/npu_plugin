//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/tensor/dtype/dtype_registry.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"

namespace mv
{
    MV_REGISTER_DTYPE(Int8)
    .setIsDoubleType(false)
    .setSizeInBits(8);
}
