/*
 * {% copyright %}
 */
#include "nn_inference_runtime.h"
#include <DrvRegUtils.h>
#include <nn_log.h>

extern "C" {
int nn_main(int, char **) {
#if NN_LOG_VERBOSITY > 0
    mvLogLevelSet(NNLOG_DEFAULT_LEVEL);
    mvLogDefaultLevelSet(NNLOG_DEFAULT_LEVEL);
#endif

    nn::inference_runtime::Service::instance().runtime().run();

    return 0;
}
}
