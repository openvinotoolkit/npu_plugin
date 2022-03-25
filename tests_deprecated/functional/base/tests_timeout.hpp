//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
