//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include <mv_types.h>
#include <nn_resource_locator.h>

#include <ShCtrl.h>

using namespace nn::common_runtime;

namespace nn {
namespace inference_runtime {
namespace shaves {

using namespace act_runtime;

class ShaveManager {
public:
    ShaveManager(const common_runtime::StaticMapping &sMapping);
    ~ShaveManager();

    void startNNShavesForTile(const uint32_t tile);
    void startNNShavesForTiles();
    void stopNNShavesForTileMask(uint32_t mask);
    void startNNShavesForTileMask(uint32_t mask);

    /// These functions reduce to a noop if the shaves are already running, they match the requested configuration, and
    /// forceRestart = false
    /// @param tile description
    void startActShavesForTile(const uint32_t tile, const NNShaveRuntimeConfigs &cfgs, bool forceRestart = false);
//    void startActShavesForTile(const uint32_t tile, const ActKernelRuntimeConfigs &cfgs, bool forceRestart = false);
    void startActShavesForTiles(const NNShaveRuntimeConfigs &cfgs, bool forceRestart);
//    void startActShavesForTiles(const ActKernelRuntimeConfigs &cfgs, bool forceRestart = false);

    void stopNNShavesForTile(const uint32_t tile);
    void stopNNShavesForTiles();

    void stopActShavesForTile(const uint32_t tile);
    void stopActShavesForTiles();

    void updateSNNConfigsForTile(const uint32_t tile, const common_runtime::NNShaveRuntimeConfigs &cfgs);
    void updateSNNConfigsForTiles(const common_runtime::NNShaveRuntimeConfigs &cfgs);

private:
    void startActShaves(const uint8_t tile, const common_runtime::NNShaveRuntimeConfigs &cfgs);
//    void startActShaves(const uint8_t tile, const ActKernelRuntimeConfigs &cfgs);
    bool processConfigChanges(const uint8_t tile, const common_runtime::NNShaveRuntimeConfigs &cfgs);
//    bool processConfigChanges(const uint8_t tile, const ActKernelRuntimeConfigs &cfgs);
    void initActRtCodeBuffer(const uint8_t tile);

#ifdef CONFIG_VALIDATION_APP_ENABLED
//    void initActRtStacksAndDatas(const uint8_t tile, const ActKernelRuntimeConfigs &cfgs);
    void initActRtStacksAndDatas(const uint8_t tile, const common_runtime::NNShaveRuntimeConfigs &cfgs);
#endif

    const StaticMapping &cmxMapping;

    ShHandle *actShvHnd[AS_TOTAL];
    ShHandle *nnShvHnd[SNN_TOTAL];

    uint32_t actShvStacks[AS_TOTAL]{0};

    actRuntimeEntry actShvEntries[MAX_TILES]{0};
    uint8_t *actShvTextsBuffers[MAX_TILES]{nullptr};
    uint32_t actShvTxtBuffSizes[MAX_TILES]{0};
    uint32_t perfMasks[common_runtime::MAX_TILES]{0};
};

} // namespace shaves
} // namespace inference_runtime
} // namespace nn
