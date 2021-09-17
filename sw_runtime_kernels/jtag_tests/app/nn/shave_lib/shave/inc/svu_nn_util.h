/*
* {% copyright %}
*/
#pragma once
#include <sw_nn_runtime_types.h>
#include <svuCommonShave.h>

#define mscroxstr(x) str(x)
#define str(x) #x

// clang-format off

/// @brief Shave interrupt and continue instruction
#define SHAVE_SWI_CNT(x) __asm volatile ( \
    "NOP"                       "\n\t" \
    "BRU.SWIC " mscroxstr(x)    "\n\t" \
    "NOP 6"                     "\n\t" \
    ::: "memory")

// clang-format on

inline bool isControllerShave(const nn::shave_lib::svuNNRtInit *init) {
    return init->rtState.totResources[0].shaveID == scGetShaveNumber();
}

namespace nn {
namespace shave_lib {} // namespace shave_lib
} // namespace nn
