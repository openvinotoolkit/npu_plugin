/*
* {% copyright %}
*/
#include <ShDrvMutex.h>
#include <ShDrvCmxDma.h>
#include <svuCommonShave.h>
#include <nn_log.h>
#include <svu_nn_debug.h>
#include <svu_nn_runtime.h>
#include <sw_nn_runtime_types.h>
#include <sw_shave_lib_common.h>
#include <svu_nn_util.h>
#include <sys/shave_system.h>

extern "C" {
#include <CmxFifo.h>
}

namespace nn {
namespace shave_lib {

extern "C"  __attribute__((dllexport)) void svuNNEntry(svuNNRtInit *init)
{
    __setheap(nullptr, 0);

    if (isControllerShave(init))
        ShDrvMutexRequest(init->svuMutexId);

    dbgStateInit(init);

    {
        // Initialize the CMX FIFO local to the SHAVE for monitoring.
        auto rc = CmxFifoInitialize();
        if (rc != MV_RET_SUCCESS)
            nnLog(MVLOG_ERROR, "Error initializing CMX FIFO (%d)\n", rc);
    }

    {
        auto rc = ShDrvCmxDmaInitialize(nullptr);
        if (rc != MYR_DRV_SUCCESS)
            nnLog(MVLOG_ERROR, "Error initializing CMX DMA (%d)\n", rc);
    }

    {
        dbgState(RtDbgState::RuntimeInitStarting, init);
        SVUNNRuntime slr{ init };
        dbgState(RtDbgState::RuntimeInitComplete, (uint32_t) slr.isController);

        dbgState(RtDbgState::RuntimeStarting);

        slr.runRT();

        dbgState(RtDbgState::RuntimeComplete);
    }

    if (isControllerShave(init))
        ShDrvMutexRelease(init->svuMutexId);
}
} // namespace shave_lib
} // namespace nn
