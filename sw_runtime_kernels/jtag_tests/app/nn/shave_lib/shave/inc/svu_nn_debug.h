/*
* {% copyright %}
*/
#pragma once

#ifdef DEBUG_NN_SVU_RUNTIME
#    include <svu_nn_irq.h>
#    include <svu_nn_util.h>
#    include <sw_nn_runtime_types.h>
#    include <stdio.h>

#    if defined(__cplusplus)
extern "C" {
#    endif

#    define dbgState_1(state) dbgStateImpl(state, 0xDEADBEEF)
#    define dbgState_2(state, value) dbgStateImpl(state, reinterpret_cast<uint32_t>(value))
#    define dbgState_3(state, value, val3) dbgStateImpl(state, reinterpret_cast<uint32_t>(value), reinterpret_cast<uint32_t>(val3))
#    define dbgState_4(state, val, val2, tmp, ...) tmp
#    define dbgState_sel(...) dbgState_4(__VA_ARGS__, dbgState_3, dbgState_2, dbgState_1)
#    define dbgState(...) dbgState_sel(__VA_ARGS__)(__VA_ARGS__)

// Below static vars are not initialized properly currently...
static nn::shave_lib::RtDbgState *staPtr;
static uint32_t *valPtr;
static uint32_t *valPtr2;
static volatile bool *intFlg;
static uint32_t *irq_tx;
static const uint32_t *svuMutexId;

void dbgStateInit(const nn::shave_lib::svuNNRtInit *init);
void dbgStateImpl(nn::shave_lib::RtDbgState state, uint32_t value, uint32_t value2 = 0);

inline void dbgStateInit(const nn::shave_lib::svuNNRtInit *init) {
    if (!isControllerShave(init))
    {
        staPtr = nullptr;
        return;
    }

    staPtr = &init->rtState->dbgState;
    valPtr = &init->rtState->dbgValue;
    valPtr2 = &init->rtState->dbgValue2;
    intFlg = &init->rtState->lrtInterruptServiced;
    svuMutexId = &init->svuMutexId;
    irq_tx = &init->rtState->irq_tx;
}

inline void dbgStateImpl(nn::shave_lib::RtDbgState state, uint32_t value, uint32_t value2) {
    // Debug not enabled for this shave
    if (staPtr == nullptr)
        return;

    *staPtr = state;
    *valPtr = value;
    *valPtr2 = value2;
    (*irq_tx)++;

    lrtMessageSend(SVU_NN_TAG_FIELD_PRINT_RT_TRACE, *svuMutexId, intFlg);
}

#    if defined(__cplusplus)
}
#    endif

#else
//   sizeof will be compiled out but supports arg lists
#    define dbgStateInit sizeof
#    define dbgState sizeof
#endif
