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
#include "vpux/compiler/dialect/VPUIP/tiling.hpp"

using namespace vpux;
using namespace VPUIP;

SmallVector<DpuTile> VPUIP::DpuTiler::tileOverH(uint32_t numDPU, ShapeRef outShape, ArrayRef<int64_t> opPadsBegin,
                                                ArrayRef<int64_t> opPadsEnd) {
    // FIXME: find the optimal number of tiles
    const auto minTileSize = 1;

    const int64_t minTilesCount = 1;
    const auto maxTilesCount = static_cast<int64_t>(numDPU);

    auto tilesCount = outShape[IERT::ConvolutionOp::act_height_dim()] / minTileSize;
    tilesCount = std::min(std::max(tilesCount, minTilesCount), maxTilesCount);

    Shape nTilesOnDim(outShape.size(), minTilesCount);
    nTilesOnDim[IERT::ConvolutionOp::act_height_dim()] = tilesCount;

    const auto outTiles = Tiling::fillDividedTiles(nTilesOnDim, outShape);
    SmallVector<DpuTile> dpuTiles;
    dpuTiles.reserve(outTiles.size());

    const auto C = IERT::ConvolutionOp::act_channel_dim();
    const auto H = IERT::ConvolutionOp::act_height_dim();
    const auto W = IERT::ConvolutionOp::act_width_dim();

    for (const auto& outTile : outTiles) {
        const auto padsTileConf = Tiling::backInferPadsTile(outTile, outShape, opPadsBegin, opPadsEnd);

        SmallVector<int64_t> start{outTile.offsets[W], outTile.offsets[H], outTile.offsets[C]};
        SmallVector<int64_t> end{outTile.offsets[W] + outTile.shape[W] - 1, outTile.offsets[H] + outTile.shape[H] - 1,
                                 outTile.offsets[C] + outTile.shape[C] - 1};

        dpuTiles.push_back({start, end, padsTileConf.begin, padsTileConf.end});
    }

    return dpuTiles;
}
