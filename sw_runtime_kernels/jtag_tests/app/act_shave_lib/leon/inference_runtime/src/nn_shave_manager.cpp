//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#define CONFIG_NN_LOG_VERBOSITY_LRT_INFO

#include "nn_shave_manager.h"
#include <mv_types.h>
#include <nn_log.h>
#include <nn_fifo.h>
#include <nn_fifo_configs.h>
#include <nn_hw_resources.h>
#include <nn_runtime_configs.h>
#include <nn_fifo_manager.h>
#include <OsDrvBootShave.h>
#include <ShaveL2Cache.h>

namespace {
#define MAX_SUPPORTED_NN_SHV_PER_TILE 2
#define SUPPORTED_ACT_SHV_PER_TILE_NB 2

#define TILE_0 0
#define TILE_1 1

inline constexpr ShaveWindow mapWindowAddrMaskToName(uint32_t windowAddrMask) {
    switch (windowAddrMask) {
        case 0x1C000000:
            return ShaveWindow::HGL_SHAVE_WINDOW_A;
        case 0x1D000000:
            return ShaveWindow::HGL_SHAVE_WINDOW_B;
        case 0x1E000000:
            return ShaveWindow::HGL_SHAVE_WINDOW_C;
        case 0x1F000000:
            return ShaveWindow::HGL_SHAVE_WINDOW_D;
        default:
            return ShaveWindow::HGL_SHAVE_WINDOW_NB;
    }
}
} // namespace

namespace nn {
namespace inference_runtime {
namespace shaves {

using namespace act_runtime;
using namespace common_runtime;
using namespace common_runtime::fifo;

ShaveManager::ShaveManager(const StaticMapping &sMapping)
    : cmxMapping(sMapping) {
    // ShaveNN uses SHVNN0 for Tile 0, SHVNN2 for Tile 1.
    const uint32_t nnShaveId[] = {0, 2};

    // One-time init (semaphores, handles, ...)
    auto rc = ShCtrlInit();
    if (rc != HGL_SHAVE_CTRL_SUCCESS) {
        nnLog(MVLOG_ERROR, "ShaveCtrlInit: rc = %x", (int)rc);
    }

    // ACTSHV handle init
    for (unsigned int shave = 0; shave < AS_TOTAL; ++shave) {
        nnLog(MVLOG_DEBUG, "ShaveCtrlOpen: ACTSHV %d", shave);
        auto rc = ShCtrlOpen(SHAVE_ACT, shave, &actShvHnd[shave]);
        if (rc != HGL_SHAVE_CTRL_SUCCESS) {
            nnLog(MVLOG_ERROR, "ShaveCtrlOpen: rc = %x", (int)rc);
        }

        // Set the monitor bit for each ACTSHV
        nn::util::fifoMonitorDynamicSelect(shave, unpackSHVConfig(acts_cfgs[shave]).work.fifo,
                                           unpackSHVConfig(acts_cfgs[shave]).work.index);
    }

    // For STD FW apps: L2C_PAGE_TABLE is set in OsDrvInitShave.c
#if (!(defined(CONFIG_STD_FW_APP_CLIENT)) && defined(CONFIG_NN_NCE_L2C_PAGE_TABLE))
    // Apply L2C page table setting
    uint32_t l2c_page_table = CONFIG_NN_NCE_L2C_PAGE_TABLE;
    nnLog(MVLOG_INFO, "Setting L2C_PAGE_TABLE = %x", l2c_page_table);
    auto rc3 = ShaveL2CacheSetPage(l2c_page_table);
    if (rc3 != SHAVE_L2_SUCCESS)
        nnLog(MVLOG_ERROR, "ShaveL2CacheSetPage: rc = %x", (int)rc3);
#endif
}

ShaveManager::~ShaveManager(void) {
    // ACTSHV handle de-init
    for (unsigned int shave = 0; shave < AS_TOTAL; ++shave) {
        auto rc = ShCtrlClose(&actShvHnd[shave]);
        if (rc != HGL_SHAVE_CTRL_SUCCESS) {
            nnLog(MVLOG_WARN, "ShaveCtrlClose: rc = %x", (int)rc);
        }
    }
}

bool ShaveManager::processConfigChanges(const uint8_t tile, const ActKernelRuntimeConfigs &cfgs) {
    bool foundChanges{false};

    if (cfgs.actRtWindowBase_ != nullptr) {
        // lazy assert: we could figure this out from cfgs.RtInvos_.size() <- is the compiled num actShaves
        static_assert(MAX_TILES <= 2,
                      "ShaveManager cannot infer ActRTConfig stack locations for greater than two tiles");

        for (uint32_t i = tile * AS_PER_TILE; i < tile * AS_PER_TILE + AS_PER_TILE; i++) {
            foundChanges |= actShvStacks[i] != cfgs.stackFrames_[i];
            actShvStacks[i] = cfgs.stackFrames_[i];
        }
    } else {
        nnLog(MVLOG_INFO, "Invalid ActKernelRuntimeConfigs found in inference request");
        return true;
    }

    foundChanges |= actShvTxtBuffSizes[tile] != cfgs.codeWindowBufferSize_;
    actShvTxtBuffSizes[tile] = cfgs.codeWindowBufferSize_;

    if (actShvTextsBuffers[tile] != cfgs.actRtWindowBase_) {
        foundChanges = true;

        actShvTextsBuffers[tile] = cfgs.actRtWindowBase_;

        if (reinterpret_cast<uint32_t>(actShvTextsBuffers[tile]) % 1024) {
            nnLog(MVLOG_ERROR, "ActRT .text window base is not aligned to 1KB: 0x%x", actShvTextsBuffers[tile]);
        }
    }

    if (cfgs.useScheduleEmbeddedRt_) {
        foundChanges |= actShvEntries[tile] != cfgs.runtimeEntry_;
        actShvEntries[tile] = cfgs.runtimeEntry_;
    }

    for (uint32_t i = tile * AS_PER_TILE; i < tile * AS_PER_TILE + AS_PER_TILE; i++) {
        foundChanges = actShvStacks[i] != cfgs.stackFrames_[i];
        actShvStacks[i] = cfgs.stackFrames_[i];
    }

    return foundChanges;
}

void ShaveManager::initActRtCodeBuffer(const uint8_t tile) {
    nnLog(MVLOG_WARN, "Non-userspace resident act-shave instructions are unsupported in std_FW_img with MMU context "
                      "isolation enabled. Act kernels will not work!");
}

#ifdef CONFIG_VALIDATION_APP_ENABLED
void ShaveManager::initActRtStacksAndDatas(const uint8_t tile, const ActKernelRuntimeConfigs &cfgs) {
    // TODO: Implement this when the compiler is ready to embed the ActRT into the blob
    nnLog(MVLOG_DEBUG, "Validation App Mode: Using embedded Act stacks");

    // Workaround for E#23330 when running VPUX softmax software layer.
    // The DDR stacks for the activation SHAVEs is causing instruction corruption.
    // Might be related to how the SHAVE L2 is configured. Placing actSHAVE stacks
    // back in NN CMX
    auto nnActShvStack = cmxMapping.actShvStack_[0].addr32();
    auto stackSize = cmxMapping.actShvStack_[0].size();

    actShvStacks[0] = nnActShvStack + (stackSize >> 1);
    actShvStacks[1] = nnActShvStack + stackSize;

    nnActShvStack = cmxMapping.actShvStack_[1].addr32();
    stackSize = cmxMapping.actShvStack_[1].size();
    actShvStacks[2] = nnActShvStack + (stackSize >> 1);
    actShvStacks[3] = nnActShvStack + stackSize;

    // FIXME: for now, this hack is in place to make things work
    // for (uint32_t i = tile * AS_PER_TILE; i < tile * AS_PER_TILE + AS_PER_TILE; i++)
    //    actShvStacks[i] = cfgs.stackFrames_[i];
}
#endif

void ShaveManager::startActShaves(const uint8_t tile, const ActKernelRuntimeConfigs &cfgs) {
    static_assert(AS_PER_TILE == SUPPORTED_ACT_SHV_PER_TILE_NB, "Only 2 ActShvs per tile is supported");

    // Check that we are operating on a supported tile ID
    if (!(tile < MAX_TILES)) {
        nnLog(MVLOG_ERROR, "Invalid Shave type selected");
        return;
    }

    // Shave IDs, depending on the tile
    const uint32_t startShvId = tile * AS_PER_TILE;
    const uint32_t maxShvId = startShvId + AS_PER_TILE;

    // Set stack location, set the stack size, then start the Shave
    for (uint32_t i = startShvId; i < startShvId + 1; i++) {
        nnLog(MVLOG_DEBUG, "ACTSHV %d stack addr @ %p", i, actShvStacks[i]);
        auto rc = ShCtrlSetStackAddr(actShvHnd[i], actShvStacks[i]);
        if (rc != HGL_SHAVE_CTRL_SUCCESS) {
            nnLog(MVLOG_ERROR, "ActShaveCtrlSetStackAddr: %d", (int)rc);
        }

        nnLog(MVLOG_DEBUG, "ACTSHV %d stack size = 0x%x", i, cfgs.stackSize_);
        rc = ShCtrlSetStackSize(actShvHnd[i], cfgs.stackSize_);
        if (rc != HGL_SHAVE_CTRL_SUCCESS) {
            nnLog(MVLOG_ERROR, "ActShaveCtrlSetStackSize: %d", (int)rc);
        }

        nnLog(MVLOG_DEBUG, "ACTSHV %d WIN_%d = %p", i, mapWindowAddrMaskToName(ACT_RT_CODE_WINDOW),
              reinterpret_cast<uint32_t>(actShvTextsBuffers[tile]));
        rc = ShCtrlSetWindowAddr(actShvHnd[i], mapWindowAddrMaskToName(ACT_RT_CODE_WINDOW),
                                    reinterpret_cast<uint32_t>(actShvTextsBuffers[tile]));
        if (rc != HGL_SHAVE_CTRL_SUCCESS) {
            nnLog(MVLOG_ERROR, "ShaveCtrlSetWindowAddr (for RT code buffer): 0x%x", ACT_RT_CODE_WINDOW);
        }

        nnLog(MVLOG_DEBUG, "ACTSHV %d WIN_%d = %p", i, mapWindowAddrMaskToName(ACT_CMX_WINDOW),
              cmxMapping.workareas_[tile].addr32());
        rc = ShCtrlSetWindowAddr(actShvHnd[i], mapWindowAddrMaskToName(ACT_CMX_WINDOW),
                                    cmxMapping.workareas_[tile].addr32());
        if (rc != HGL_SHAVE_CTRL_SUCCESS) {
            nnLog(MVLOG_ERROR, "ShaveCtrlSetWindowAddr (for window into CMX): ox%x", ACT_CMX_WINDOW);
        }

        nnLog(MVLOG_DEBUG, "Starting ACTSHV %d from %p windowed to A", i, actShvEntries[tile]);
        auto fifoCfg = acts_cfgs[i];
        printFifoConfig(unpackSHVConfig(fifoCfg));
        rc = ShCtrlStart(actShvHnd[i], reinterpret_cast<void *>(actShvEntries[tile]), "i", fifoCfg);
        if (rc != HGL_SHAVE_CTRL_SUCCESS) {
            nnLog(MVLOG_ERROR, "ActShaveCtrlStart: %d", (int)rc);
        }
    }
}

void ShaveManager::startActShavesForTile(const uint32_t tile, const ActKernelRuntimeConfigs &cfgs, bool forceRestart) {
    forceRestart |= processConfigChanges(tile, cfgs);

    if (forceRestart) {
        stopActShavesForTile(tile);

#ifdef CONFIG_VALIDATION_APP_ENABLED
        initActRtStacksAndDatas(tile, cfgs);
#endif

        if (!cfgs.useScheduleEmbeddedRt_) {
            initActRtCodeBuffer(tile);
        }
    }

    startActShaves(tile, cfgs);
}

void ShaveManager::startActShavesForTiles(const ActKernelRuntimeConfigs &cfgs, bool forceRestart) {
    startActShavesForTile(0, cfgs, forceRestart);
    startActShavesForTile(1, cfgs, forceRestart);
}

void ShaveManager::stopActShavesForTile(uint32_t tile) {
    const unsigned int startAct = tile ? AS0_TILE1_GLOBAL_SHAVE_INDEX : AS0_TILE0_GLOBAL_SHAVE_INDEX;
    const unsigned int finalAct = startAct + AS_PER_TILE;

    for (unsigned int i = startAct; i < startAct + 1; i++) {
        nnLog(MVLOG_DEBUG, "Stopping Act Shave");
        auto rc = ShCtrlStop(actShvHnd[i]);
        if (rc != HGL_SHAVE_CTRL_SUCCESS)
            nnLog(MVLOG_ERROR, "ShaveCtrlStop: rc = %x", (int)rc);
    }
}

void ShaveManager::stopActShavesForTiles() {
    static_assert(MAX_TILES == HGL_NCE_TILE_NB, "Supports up to 2 tiles only");
    stopActShavesForTile(TILE_0);
    stopActShavesForTile(TILE_1);
}

} // namespace shaves
} // namespace inference_runtime
} // namespace nn
