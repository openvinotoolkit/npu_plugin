// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include <iostream>
#include "kmb_layer_test.hpp"

// Headers below are just for Yocto.
#ifdef __aarch64__
#include <signal.h>
#include <exception>
#include <unistd.h>
#include <stdlib.h>
#endif

// Handler for signal SIGINT. Just for running on Yocto.
#ifdef __aarch64__
void sig_handler(int num) {
   std::cerr << "\nSIGINT signal (Ctrl+C) has been gotten.\n";
   std::cerr << "Exit from program.\n";
   std::cerr << "You may check open/close channels in XLinkUtils.\n";
   exit(1);
}
#endif

int main(int argc, char **argv) {

// Register handler for signal SIGINT. Just for running on Yocto.
#ifdef __aarch64__
    struct sigaction act;
    sigemptyset(&act.sa_mask);
    act.sa_handler = &sig_handler;
    act.sa_flags = 0;

    if (sigaction (SIGINT, &act, NULL) == -1) {
        std::cerr << "sigaction() error - can't register handler for SIGINT.\n";
    }
#endif

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
