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

/**
 * @brief These values represent physical HW configurations. This file _does not_ define how this HW is configured for
 * use by the runtime. See specific *_config.h headers for that.
 *
 */

namespace nn {
namespace common_runtime {
enum : unsigned int{

    // base resources
    MAX_TILES = 2,
    DPU_PER_TILE = 1,

    // FIXME: find direct shvID defines. These correlate with DrvSvuNNControl.c's NCE_SVU_CTRL_ADDR[]
    // FIXME: these indexes may be incorrect if they are constant regardless of SNN_PER_TILE
    SNN_PER_TILE = DPU_PER_TILE,
    SNN_TOTAL = SNN_PER_TILE * MAX_TILES,
    SNN0_TILE0_GLOBAL_SHAVE_INDEX = 0,
    SNN0_TILE1_GLOBAL_SHAVE_INDEX = 2,

    AS_PER_TILE = 2,
    AS_TOTAL = AS_PER_TILE * MAX_TILES,
    AS0_TILE0_GLOBAL_SHAVE_INDEX = 0,
    AS0_TILE1_GLOBAL_SHAVE_INDEX = 2,

    // barrier defines
    MAX_BARRIERS = 64,
    BARRIERS_PER_GROUP = 32,

    // fifo defines
    MAX_FIFOS = 4,

    MAX_SLICES = 2,
    MAX_CLUSTERS = MAX_TILES,
    MAX_DPUS = 2,
    MAX_DPUS_FIFOS = 2,

    SLICE_LENGTH = 2 * 1024 * 1024,
    STOLEN_WINDOW_MIN_LEN = 512,

    ACT_SHAVES_IN_TILE = AS_PER_TILE,
    ACT_SHAVE_0_INDEX = 4,
    WORK_FIFO_COUNT = 16,

    // TODO: Enable second DMA engine on MTL (VPUNND-3752)
    // DMA_ENGINES = 2,
    DMA_ENGINES = 1,

    MAX_DMA_ENGINES = 2,

    DPUS_IN_CLUSTER = DPU_PER_TILE * MAX_TILES,

    TOTAL_PHYSICAL_BARRIERS = 64,
    TOTAL_USED_BARRIERS = BARRIERS_PER_GROUP * MAX_CLUSTERS,

    FIFO_COUNT = MAX_CLUSTERS,
    FIFO_LENGTH = 1024 / FIFO_COUNT,

};

#include <HglResources.h>
static_assert(HGL_NCE_TILE_NB == MAX_TILES, "Desync between HW const declarations!");

} // namespace common_runtime
} // namespace nn
