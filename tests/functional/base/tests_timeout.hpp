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

#include <signal.h>

#include <chrono>
#include <functional>
#include <sstream>
#include <thread>

namespace TestsTimeout {

enum RunStatus {
    OK = 0,
    TIMEOUT = -1001,
    FORK_FAILURE = -1002,
    EXECUTION_FAILURE = -1003,
    UNEXPECTED = -1004,
    UNKNOWN = -1005
};

/*
 * Uses parent/child processes interaction and signals
 * and supposes that signals are not used by the testing (target) function
 * Supposes that tests are not executed in parallel (several tests in different threads)
 */
int runWithTimeout(const std::function<void(int& childExitStatus)>&
                       runFunc,  // Lambda wrapper of target (tested with timeout) function
    // Places in childExitStatus one of the negative values from RunStatus enum if corresponding reason occurs
    // or some another (positive) value that can be interpreted by particular test itself.
    // See the example KmbNoRegressionCompilationOnly test in tests/functional/kmb_tests/kmb_regression_target.cpp
    std::string& statusMessage,
    // Timeout in seconds. Run without timeout if dSecRunTimeout == 0,
    double dSecRunTimeout);

}  // namespace TestsTimeout
