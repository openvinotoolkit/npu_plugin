//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#ifndef MV_TENSOR_RESOURCES_H_
#define MV_TENSOR_RESOURCES_H_

#include "mvTensorOutputStream.h"

namespace mv
{
    namespace tensor
    {
        struct Resources
        {
            OutputStream &debug;
        };
    }
}

#endif // MV_TENSOR_RESOURCES_H_
