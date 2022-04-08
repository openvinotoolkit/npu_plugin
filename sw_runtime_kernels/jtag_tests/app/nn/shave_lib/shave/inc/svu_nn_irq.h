/*
* {% copyright %}
*/
#pragma once

#include <ShDrvMutex.h>
#include <svu_nn_util.h>
#include <sw_nn_runtime_types.h>

inline void lrtMessageSend(uint32_t tag, uint32_t svuMutex, volatile bool *intFlag) {
    *intFlag = false;
    ShDrvMutexRelease(svuMutex);

    // Send SWIC to LRT with tag value
    switch (tag) {
    case SVU_NN_TAG_FIELD_L2C_DATA_FLUSH: SHAVE_SWI_CNT(SVU_NN_TAG_FIELD_L2C_DATA_FLUSH); break;
    case SVU_NN_TAG_FIELD_L2C_INSTR_FLUSH: SHAVE_SWI_CNT(SVU_NN_TAG_FIELD_L2C_INSTR_FLUSH); break;
    case SVU_NN_TAG_FIELD_PRINT_RT_TRACE: SHAVE_SWI_CNT(SVU_NN_TAG_FIELD_PRINT_RT_TRACE); break;
    default: break;
    }

    // Wait for request to be serviced
    while (!(*(intFlag)))
        ;
    ShDrvMutexRequest(svuMutex);
}
