/*
* {% copyright %}
*/
#ifndef NN_RUNTIME_TYPES_H__
#define NN_RUNTIME_TYPES_H__

#if defined(CONFIG_TARGET_SOC_MA2490) || defined(CONFIG_TARGET_SOC_MA2490_B0) || defined(CONFIG_TARGET_SOC_3100)
#   include "nn_runtime_types_2490.h"
#endif // CONFIG_TARGET_SOC_MA2490 || CONFIG_TARGET_SOC_MA2490_B0 || CONFIG_TARGET_SOC_3100

#if (defined(CONFIG_TARGET_SOC_3600) || defined(CONFIG_TARGET_SOC_3710) || defined(CONFIG_TARGET_SOC_3720))
#   include "nn_runtime_types_3600.h"
#endif // CONFIG_TARGET_SOC_3600 || CONFIG_TARGET_SOC_3710 || CONFIG_TARGET_SOC_3720

#endif //NN_RUNTIME_TYPES_H__
