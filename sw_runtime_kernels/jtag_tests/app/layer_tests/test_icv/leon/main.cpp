//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "icv_test_suite.h"
#include <nn_log.h>

#include <stdlib.h>

using namespace icv_tests;
volatile int status __attribute__((section(".nncmx0.shared.data"))) = 0;
void beforeReturn() {
    puts("");
};

int main(int, char **)
{
    GlobalData::init();

    volatile int localStatus = EXIT_SUCCESS;
    status = -EXIT_FAILURE;

    mvLogLevelSet(NNLOG_DEFAULT_LEVEL);
    mvLogDefaultLevelSet(NNLOG_DEFAULT_LEVEL);

    printf("\n");

    SuiteRegistry::forEach([&localStatus] (SuiteRunner* suite)
    {
        int res = suite->run();
        if ((GlobalData::runMode == RunMode::Run) && (res != 0)) {
            localStatus += res;
        }
    });

    status = localStatus;
    printf("\n");
    printf("status %d\n", status);
    printf("\n");

    // beforeReturn() function is used in debug script to set breakpoint on
    // in validation/validationApps/system/nn/mvTensor/layer_tests/test_icv/build/generated_debug_script.scr:75
    beforeReturn();
    return status;
}
