/*
* {% copyright %}
*/

#include "nn_log.h"

// Workaround for creating attributes for shave/shave_nn as mvLog does not create them by itself - limited to debug builds only.
#ifdef CONFIG_VALIDATION_APP_ENABLED
#if NN_LOG_VERBOSITY >= 1 && (defined (__shave__) || defined(__shave_nn__))
unsigned int __attribute__((weak)) MVLOGLEVEL(MVLOG_UNIT_NAME);
unsigned int __attribute__((weak)) MVLOGLEVEL(default);
#endif
#endif
