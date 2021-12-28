/*
 * {% copyright %}
 */
#pragma once

#include <DrvRegUtils.h>
#include <DrvSvuL1Cache.h>
#include <nn_runtime_types.h>

#define TO_STR(X) #X
#define VAL_TO_STR(X) TO_STR(X)

using namespace nn::act_runtime;
using namespace nn::util;

inline void waitBarrier(const BarrierUserConfig &bar, const BarrierGpioConfig &gpio, unsigned int shave_index) {
    // TODO: enable GPIO monitor when shave_index is confirmed working
    if (false && gpio.group_ > 0) {
        HglBarrierMonitorSelect(shave_index, gpio.group_ - 1);
        waitBarrierGpio(gpio.mask_);
    } else
        HglBarrierWait(bar.wait_mask_);
}

// Set the window address by writing to the local address space of the current SHAVE
inline void setShaveWindow(uint32_t windowNumber, void *targetWindowBaseAddr) {
    switch (windowNumber) {
        case 0:
            asm volatile("lsu0.sta.32 %[addr], SHAVE_LOCAL, " VAL_TO_STR(OFFSET_WIN_A_OFFSET) "\n\t" ::[addr] "r"(
                targetWindowBaseAddr));
            break;
        case 1:
            asm volatile("lsu0.sta.32 %[addr], SHAVE_LOCAL, " VAL_TO_STR(OFFSET_WIN_B_OFFSET) "\n\t" ::[addr] "r"(
                targetWindowBaseAddr));
            break;
        case 2:
            asm volatile("lsu0.sta.32 %[addr], SHAVE_LOCAL, " VAL_TO_STR(OFFSET_WIN_C_OFFSET) "\n\t" ::[addr] "r"(
                targetWindowBaseAddr));
            break;
        case 3:
            asm volatile("lsu0.sta.32 %[addr], SHAVE_LOCAL, " VAL_TO_STR(OFFSET_WIN_D_OFFSET) "\n\t" ::[addr] "r"(
                targetWindowBaseAddr));
            break;
    }
}

// inline void setFPRound(const unsigned int actShvID) {
inline void setFPRound(uint32_t f2IntRnd) {
    // Set FP round to x86 compatibility mode for verification

    uint32_t reg;
    asm volatile("lsu0.lda.32 %[regval], SHAVE_LOCAL, " VAL_TO_STR(P_CFG_OFFSET) "\n\t"
                 : [regval] "=r"(reg)
                 :
                 : "memory");
    reg = reg & f2IntRnd; // F2INTRND; 0x0 - round to nearest even
    asm volatile("lsu0.sta.32 %[fprd], SHAVE_LOCAL, " VAL_TO_STR(P_CFG_OFFSET) "\n\t" ::[fprd] "r"(reg));
}
