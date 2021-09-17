/*
* {% copyright %}
*/
#pragma once

#if defined(CONFIG_TARGET_SOC_MA2490) || defined(CONFIG_TARGET_SOC_MA2490_B0) || defined(CONFIG_TARGET_SOC_3100)
#   include "sw_shave_dispatcher_2490.h"
#endif // CONFIG_TARGET_SOC_MA2490 || CONFIG_TARGET_SOC_MA2490_B0 || CONFIG_TARGET_SOC_3100

#if (defined(CONFIG_TARGET_SOC_3600) || defined(CONFIG_TARGET_SOC_3710) || defined(CONFIG_TARGET_SOC_3720))
#   include "sw_shave_dispatcher_3600.h"
#endif // CONFIG_TARGET_SOC_3600 || CONFIG_TARGET_SOC_3710 || CONFIG_TARGET_SOC_3720
