/*
* {% copyright %}
*/
#include "nn_lnn.h"
#include <ResMgr.h>
#include <LeonNN.h>
#include <nn_log.h>
#include <mvReturnRtems.h>

#ifndef CONFIG_VALIDATION_APP_ENABLED
    extern uint32_t __lrt_bootLoder_startAddr;
    /* following LNN boot loader entrypoint is calculated from the start of the LRT_BOOT memory section + 4k trap table alignment */
    #ifdef CONFIG_DEFAULT_LNN_START_ADDRESS
    #warning "CONFIG_DEFAULT_LNN_START_ADDRESS was redefined to allow for the linker to provide the correct entry point for LNN bootloader used for std FW app LNN core booting"
    #undef CONFIG_DEFAULT_LNN_START_ADDRESS
    #endif
    #define CONFIG_DEFAULT_LNN_START_ADDRESS   (((uint32_t)&__lrt_bootLoder_startAddr) + 0x1000)
#else
    extern uint32_t lnn_text_start;
    #ifndef CONFIG_DEFAULT_LNN_START_ADDRESS
        #define CONFIG_DEFAULT_LNN_START_ADDRESS (((uint32_t)&lnn_text_start) + 0x1000)
    #endif
#endif

namespace
{
#ifdef NN_ACTIVE_TILE_MANAGEMENT
    ResMgrResource res_request[3] = {
        RESMGR_NCE_FIFO_BARRIER,
        RESMGR_NCE_L2_CACHE,
        RESMGR_NCE_LEON
    };
#else
    ResMgrResource res_request[5] = {
        RESMGR_NCE_TILE_0, RESMGR_NCE_TILE_1, RESMGR_NCE_FIFO_BARRIER, RESMGR_NCE_L2_CACHE, RESMGR_NCE_LEON
    };
#endif /* NN_ACTIVE_TILE_MANAGEMENT */

    void allocateNceResources(bool enable)
    {
        mvReturn sc = MV_RET_SUCCESS;

        if (enable) {
            // allocate resources
            for (unsigned int i = 0, n = sizeof(res_request) / sizeof(res_request[0]); i < n; ++i) {
                nnLog(MVLOG_INFO, "Allocate resource type: %d", res_request[i]);
                sc = ResMgrAllocate(res_request[i]);
                if (sc != MV_RET_SUCCESS)
                    nnLog(MVLOG_ERROR, "Failed to allocate resource type: %d. ec %d", res_request[i], sc);
            }
        } else {
            // release resources
            for (unsigned int i = 0, n = sizeof(res_request) / sizeof(res_request[0]); i < n; ++i) {
                sc = ResMgrRelease(res_request[i]);
                if (sc != MV_RET_SUCCESS)
                    nnLog(MVLOG_ERROR, "Failed to deallocate resource type: %d. ec %d", res_request[i], sc);
            }
        }
    }
}

namespace nn
{
    void lnnStart()
    {
        // allocate all resources
        allocateNceResources(true);

        nnLog(MVLOG_INFO, "NCE - LNN start address 0x%X\n", CONFIG_DEFAULT_LNN_START_ADDRESS);
        auto rc = LeonNNStartup(CONFIG_DEFAULT_LNN_START_ADDRESS);
        if (rc != RTEMS_SUCCESSFUL)
            nnLog(MVLOG_ERROR, "Failed to start LNN rc=%d", rc);
    }

    void lnnStop()
    {
        auto rc = LeonNNMarkStopped();
        if (rc != RTEMS_SUCCESSFUL)
            nnLog(MVLOG_ERROR, "Failed to mark LNN stopped rc=%d", rc);

        // There is no graceful shutdown mechanism in place for LNN, so nothing
        // else to do than acting on clocks to stop it.
        // allocate all resources
        allocateNceResources(false);
    }
}
