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

#include "vpux/compiler/dialect/VPUIP/dpu_tiler.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"

using namespace vpux;

bool vpux::VPUIP::DpuTiler::generateSplitNumberPool(uint8_t numDPU, uint32_t maxSplits,
                                                    SmallVector<uint32_t> validZTiles) {
    if (validZTiles.size() == 0){
        for (size_t i = 4; i<8; ++i){
            validZTiles.push_back(std::pow(2, i));
            validZTiles.push_back(validZTiles.back() + 16);
        }
    }
    
    SmallVector<uint32_t> max_splits_in_z;
    SmallVector<uint32_t> max_splits_in_xy;
    SmallVector<std::pair<uint8_t, uint8_t>> modes = getModes();
    for (auto z_tile : validZTiles){
        max_splits_in_z.push_back(std::ceil(outShape[Dims4D::Act::C] / (double)z_tile));
    }
    for (auto mode : modes){
        max_splits_in_xy.push_back(std::ceil(outShape[Dims4D::Act::H] / (double)mode.first) * std::ceil(outShape[Dims4D::Act::W] / (double)mode.second));
    }
    
    
}

bool vpux::VPUIP::DpuTiler::tileOverZ(uint32_t splitNumber, SmallVector<uint32_t> validZTiles, bool sparse, bool has_se) {
}

SmallVector<std::pair<uint8_t, uint8_t>> vpux::VPUIP::DpuTiler::getModes(){
    SmallVector<std::pair<uint8_t, uint8_t>> modes;
    for (auto mode : mpeModeList){
        switch (mode) {
            case VPUIP::MPEMode::MATRIX:
                modes.push_back({4,4});
                break;
            case VPUIP::MPEMode::VECTOR:
                modes.push_back({1,16});
                break;
            case VPUIP::MPEMode::VECTOR_FP16:
                modes.push_back({1,4});
                break;
            default:
                break;
        }
    }
    return modes;
}

SmallVector<VPUIP::DpuTile> vpux::VPUIP::DpuTiler::tileOverH(int64_t numDPU, ShapeRef outShape, int64_t padLeft,
                                                             int64_t padRight, int64_t padTop, int64_t padBottom) {
    // FIXME: find the optimal number of tiles
    const int64_t minTileSize = 1;

    const int64_t minTilesCount = 1;
    const int64_t maxTilesCount = numDPU;

    int64_t tilesCount = outShape[Dims4D::Act::H] / minTileSize;
    tilesCount = std::min(std::max(tilesCount, minTilesCount), maxTilesCount);

    Shape nTilesOnDim(outShape.size(), minTilesCount);
    nTilesOnDim[Dims4D::Act::H] = tilesCount;

    const auto outTiles = fillDividedTiles(nTilesOnDim, outShape);
    SmallVector<DpuTile> dpuTiles;
    dpuTiles.reserve(outTiles.size());

    for (const auto& outTile : outTiles) {
        const auto padsTileConf = backInferPadsTile(outTile, outShape, PadInfo(padLeft, padRight, padTop, padBottom));

        SmallVector<int64_t> start{outTile.offsets[Dims4D::Act::W], outTile.offsets[Dims4D::Act::H],
                                   outTile.offsets[Dims4D::Act::C]};
        SmallVector<int64_t> end{outTile.offsets[Dims4D::Act::W] + outTile.shape[Dims4D::Act::W] - 1,
                                 outTile.offsets[Dims4D::Act::H] + outTile.shape[Dims4D::Act::H] - 1,
                                 outTile.offsets[Dims4D::Act::C] + outTile.shape[Dims4D::Act::C] - 1};

        dpuTiles.push_back({start, end, padsTileConf.left, padsTileConf.right, padsTileConf.top, padsTileConf.bottom});
    }

    return dpuTiles;
}
