//
// Copyright 2019 Intel Corporation.
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

#include <thread>
#include <chrono>
#include <sstream>
#include <signal.h>

#include <functional>

namespace TestsTimeout
{

enum RunStatus {
    OK = 0,
    TIMEOUT = 1,
    UEXPECTED = 10,
    UNKNOWN = 100
};

void cleanPendingSignals(sigset_t& sigset, siginfo_t& info, int millySecCleanPendingTimeout = 100);

int runWithTimeout (
        const std::function<void(int&)>& runFunc,
        std::string& statusMessage,
        int secRunTimeout, int millySecKillSignalWait = 2000, int millySecCleanPendingTimeout = 100);

} // namespace TestsTimeout
