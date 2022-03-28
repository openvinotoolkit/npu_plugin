//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

namespace vpux {

#define CONTEXT_ERROR_str std::string("[INVALID REMOTE CONTEXT] ")

///  Context
#define FAILED_CAST_CONTEXT               \
    std::string("Context is incorrect.\n" \
                "Please make sure you are using VPUXRemoteContext.")

///  Infer request
#define NO_EXECUTOR_FOR_INFERENCE               \
    std::string("Can't create infer request!\n" \
                "Please make sure that the device is available. Only exports can be made.")

}  //  namespace vpux
