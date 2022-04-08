//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <signal.h>
#include <functional_test_utils/summary/op_summary.hpp>
#include <iostream>
#include <sstream>
#include "gtest/gtest.h"
#include "kmb_test_report.hpp"

// Headers below are just for Yocto.
#ifdef __aarch64__
#include <stdlib.h>
#include <unistd.h>
#include <exception>
#endif

// Handler for signal SIGINT. Just for running on Yocto.
#ifdef __aarch64__
void sigint_handler(int num) {
    std::cerr << "\nSIGINT signal (Ctrl+C) has been gotten.\n";
    std::cerr << "Exit from program.\n";
    std::cerr << "You may check open/close channels in XLinkUtils.\n";
    exit(1);
}
#endif

void sigsegv_handler(int errCode) {
    auto& s = ov::test::utils::OpSummary::getInstance();
    s.saveReport();
    std::cerr << "Unexpected application crash with code: " << errCode << std::endl;
    std::abort();
}

int main(int argc, char** argv, char** envp) {
// Register handler for signal SIGINT. Just for running on Yocto.
#ifdef __aarch64__
    struct sigaction act;
    sigemptyset(&act.sa_mask);
    act.sa_handler = &sigint_handler;
    act.sa_flags = 0;

    if (sigaction(SIGINT, &act, NULL) == -1) {
        std::cerr << "sigaction() error - can't register handler for SIGINT.\n";
    }
#endif

    // register crashHandler for SIGSEGV signal
    signal(SIGSEGV, sigsegv_handler);

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new LayerTestsUtils::KmbTestReportEnvironment());

    std::ostringstream oss("Environment variables: ", std::ios_base::ate);
    for (char** env = envp; *env != 0; env++) {
        oss << *env << "; ";
    }
    std::cout << oss.str() << std::endl;

    return RUN_ALL_TESTS();
}
