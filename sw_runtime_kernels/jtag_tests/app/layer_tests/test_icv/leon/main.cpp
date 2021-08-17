// {% copyright %}

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

#if 0
    printf("#********************************************************************************\n");
    printf("# main()\n");
    printf("#********************************************************************************\n");
#endif

    printf("\n");

    if (GlobalData::runMode == RunMode::Run) {
        mvTensorInit(0, MVTENSOR_MAX_SHAVES, MVTENSOR_MAX_SHAVES);
    }

    SuiteRegistry::forEach([&localStatus] (SuiteRunner* suite)
    {
        int res = suite->run();
        if ((GlobalData::runMode == RunMode::Run) && (res != 0)) {
            localStatus += res;
        }
    });

    status = localStatus;
    if (GlobalData::runMode == RunMode::Run)
        mvTensorClose();
    printf("\n");
    printf("status %d\n", status);
    printf("\n");

    // beforeReturn() function is used in debug script to set breakpoint on
    // in validation/validationApps/system/nn/mvTensor/layer_tests/test_icv/build/generated_debug_script.scr:75
    beforeReturn();
    return status;
}
