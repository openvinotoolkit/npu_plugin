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

#include "vpux/compiler/core/tiling.hpp"

using namespace vpux;

SmallVector<VPUIP::DpuTile> vpux::VPUIP::DpuTiler::tileOverH(int64_t numDPU, ShapeRef outShape, int64_t padLeft,
                                                             int64_t padRight, int64_t padTop, int64_t padBottom) {
    // FIXME: find the optimal number of tiles
    const int64_t minTileSize = 1;

    const int64_t minTilesCount = 1;
    const int64_t maxTilesCount = numDPU;

    int64_t tilesCount = outShape[IE::Dims4D::Act::H] / minTileSize;
    tilesCount = std::min(std::max(tilesCount, minTilesCount), maxTilesCount);

    Shape nTilesOnDim(outShape.size(), minTilesCount);
    nTilesOnDim[IE::Dims4D::Act::H] = tilesCount;

    const auto outTiles = fillDividedTiles(nTilesOnDim, outShape);
    SmallVector<DpuTile> dpuTiles;
    dpuTiles.reserve(outTiles.size());

    for (const auto& outTile : outTiles) {
        const auto padsTileConf = backInferPadsTile(outTile, outShape, padLeft, padRight, padTop, padBottom);

        SmallVector<int64_t> start{outTile.offsets[IE::Dims4D::Act::W], outTile.offsets[IE::Dims4D::Act::H],
                                   outTile.offsets[IE::Dims4D::Act::C]};
        SmallVector<int64_t> end{outTile.offsets[IE::Dims4D::Act::W] + outTile.shape[IE::Dims4D::Act::W] - 1,
                                 outTile.offsets[IE::Dims4D::Act::H] + outTile.shape[IE::Dims4D::Act::H] - 1,
                                 outTile.offsets[IE::Dims4D::Act::C] + outTile.shape[IE::Dims4D::Act::C] - 1};

        dpuTiles.push_back(
                {start, end, padsTileConf.padLeft, padsTileConf.padRight, padsTileConf.padTop, padsTileConf.padBottom});
    }

    return dpuTiles;
}
