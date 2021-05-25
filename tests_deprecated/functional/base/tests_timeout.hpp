//
// Copyright 2019 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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
