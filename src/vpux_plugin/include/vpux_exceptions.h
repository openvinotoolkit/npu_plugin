//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include <cpp_interfaces/exception2status.hpp>

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
