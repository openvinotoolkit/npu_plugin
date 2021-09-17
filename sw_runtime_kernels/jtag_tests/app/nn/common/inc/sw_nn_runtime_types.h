/*
* {% copyright %}
*/
#ifndef SW_NN_RUNTIME_TYPES_H__
#define SW_NN_RUNTIME_TYPES_H__

#if defined(CONFIG_TARGET_SOC_MA2490) || defined(CONFIG_TARGET_SOC_MA2490_B0) || defined(CONFIG_TARGET_SOC_3100)
#   include "sw_nn_runtime_types_2490.h"
#elif (defined(CONFIG_TARGET_SOC_3600) || defined(CONFIG_TARGET_SOC_3710) || defined(CONFIG_TARGET_SOC_3720))
#   include "sw_nn_runtime_types_3600.h"
#else
#   error "Unsupported target"
#endif

#endif //SW_NN_RUNTIME_TYPES_H__
