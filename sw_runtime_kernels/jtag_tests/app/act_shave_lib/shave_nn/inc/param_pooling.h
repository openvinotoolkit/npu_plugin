// {% copyright %}

#pragma once

#include "sw_layer_params.h"
#include <cstdint>
#include <mv_types.h>

#define VECTOR_SIZE (8)
#define CMX_ALIGN (32)
#define DIVR(_val, _size) (((_val) + ((_size)-1)) / (_size))
#define ALIGN_TO_MULTIPLE(_size, _val) (DIVR((_val), (_size)) * (_size))
#define INPUT_BPP (2)

namespace nn {
namespace act_shave_lib {
enum TilingMode {
    TilingUnknown,
    TilingH,
    TilingHW,
    TilingHC, // HWC
    TilingC,
    TilingCH,
    TilingCW // CHW
};

enum AvgPoolingMode { HWC_AvgPoolMxN, CHW_AvgPoolMxN, HWC_AvgPool3x3 };

#pragma pack(push, 1)
struct PoolParams // : LayerParams
        {
    const half *input;
    half *output;
    u8 *cmxslice;

    uint32_t channels;
    uint32_t inHeight;
    uint32_t inWidth;
    uint32_t outHeight;
    uint32_t outWidth;
    uint32_t inputStride; // CHW only
    uint32_t outputStride;
    uint32_t radixX;
    uint32_t radixY;
    uint32_t strideX;
    uint32_t strideY;
    uint32_t padX;
    uint32_t padY;
    uint32_t rpadX;
    uint32_t bpadY;

    uint32_t heightTiles;
    uint32_t widthTiles;
    uint32_t channelTiles;

    uint32_t inTileHeight;
    uint32_t inTileWidth;
    uint32_t outTileHeight;
    uint32_t outTileWidth;
    uint32_t tileChannels;
    uint32_t sliceChannels;

    uint32_t firstTile;
    uint32_t numTiles;

    uint32_t inPlaneSize;  // CHW only
    uint32_t outPlaneSize; // CHW only
    uint32_t tmpPlaneSize; // CHW only

    uint32_t sliceGroups;       // 3x3 only
    uint32_t channelsPerGroup;  // 3x3 only
    uint32_t maxGroupsPerShave; // 3x3 only

    uint16_t tilingMode;
    uint16_t rowsFirst;
    uint16_t fillInAvg;

    bool excludePad;

    NDOrder inOrder;
    AvgPoolingMode mode;

    uint32_t batch_dim;
    uint32_t input_batch_step;
    uint32_t output_batch_step;
};
#pragma pack(pop)
} // namespace shave_lib
} // namespace nn
