/*
* {% copyright %}
*/
#pragma once

// These arrays should not be directly accessed by SHAVE kernel code!
// Only access through the pointers passed in through LayerParams
// (usually the passed LayerParams * == sParam)
unsigned char __attribute__((section(".data"), aligned(64))) sParam[SHAVE_LIB_PARAM_SIZE];
unsigned char __attribute__((section(".data"), aligned(64))) sData[SHAVE_LIB_DATA_SIZE];
