//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "mvTensor.h"
#include "mvTensor_cpp.h"

#include <stdio.h>
#include <mvTensorDebug.h>

#include "mvTensorUtil.h"
#include "mvTensorOutputStream.h"

typedef const char* (* SubOpNames)(int);

namespace
{
    const char empty[] = "UNKNOWN";
}

const char* getOpName(t_MvTensorOpType op)
{
    if ((int)op < (int)MVCNN::SoftwareLayerParams_MIN) {
        return empty;
    } else {
        return MVCNN::EnumNamesSoftwareLayerParams()[op];
    }
}
