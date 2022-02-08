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

#include <file_utils.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <mlir/Support/FileUtilities.h>
#include <functional>
#include <numeric>
#include <set>

using namespace vpux;

static const uint32_t DEFAULT_ZTILE_VALUE = 16;
template <typename T>
static T divRoundUp(T x, T m) {
    return (x + m - 1) / m;
}

template <typename T>
static T padRoundUp(T x, T m) {
    return divRoundUp(x, m) * m;
}

static double splitEfficiency(ShapeRef tensorzShape, ShapeRef paddedTensorShape) {
    VPUX_THROW_WHEN(tensorzShape.size() != paddedTensorShape.size(), "Unmatched tensor dims!");
    return static_cast<double>(std::accumulate(tensorzShape.begin(), tensorzShape.end(), static_cast<int64_t>(1),
                                               std::multiplies<int64_t>())) /
           static_cast<double>(std::accumulate(paddedTensorShape.begin(), paddedTensorShape.end(),
                                               static_cast<int64_t>(1), std::multiplies<int64_t>()));
}

static SmallVector<uint32_t> getSplitsFromRange(uint32_t maxSplitRange, uint32_t maxLimit) {
    SmallVector<uint32_t> splits;
    for (uint32_t idx = 0; idx < std::log2(maxSplitRange); idx++) {
        auto powIdx = static_cast<uint32_t>(std::pow(2, idx));
        if (maxSplitRange % powIdx == 0 && maxSplitRange / powIdx <= maxLimit) {
            splits.push_back(maxSplitRange / powIdx);
        }
    }
    return splits;
}

bool vpux::VPUIP::DpuTiler::generateSplitNumberPool(int64_t numDPU, uint32_t maxSplits,
                                                    SmallVector<uint32_t> validZTiles) {
    if (validZTiles.empty()) {
        for (size_t i = 4; i < 8; ++i) {
            validZTiles.push_back(static_cast<uint32_t>(std::pow(2, i)));
            validZTiles.push_back(validZTiles.back() + 16);
        }
    }

    SmallVector<uint32_t> max_splits_in_z;
    SmallVector<uint32_t> max_splits_in_xy;
    SmallVector<std::pair<uint8_t, uint8_t>> modes = getModes();
    for (const auto& z_tile : validZTiles) {
        max_splits_in_z.push_back(static_cast<uint32_t>(std::ceil(_outShape[Dims4D::Act::C] / (double)z_tile)));
    }
    for (const auto& mode : modes) {
        max_splits_in_xy.push_back(static_cast<uint32_t>(std::ceil(_outShape[Dims4D::Act::H] / (double)mode.first) *
                                                         std::ceil(_outShape[Dims4D::Act::W] / (double)mode.second)));
    }
    auto maxSplitInZ = *std::max_element(max_splits_in_z.begin(), max_splits_in_z.end());
    auto maxSplitInXY = *std::max_element(max_splits_in_xy.begin(), max_splits_in_xy.end());
    maxSplits = std::min<uint32_t>(maxSplits, std::max(maxSplitInZ, maxSplitInXY));
    std::set<uint32_t> dpuMulSplits;
    for (auto i = numDPU; i < maxSplits + 1; i = i + numDPU) {
        dpuMulSplits.insert(static_cast<uint32_t>(i));
    }
    for (auto splitsZ : max_splits_in_z) {
        auto zRanges = getSplitsFromRange(splitsZ, maxSplits);
        dpuMulSplits.insert(zRanges.begin(), zRanges.end());
    }
    for (auto splitsXY : max_splits_in_xy) {
        auto xyRanges = getSplitsFromRange(splitsXY, maxSplits);
        dpuMulSplits.insert(xyRanges.begin(), xyRanges.end());
    }
    dpuMulSplits.insert(1);
    _splitNumberPool = SmallVector<uint32_t>(dpuMulSplits.begin(), dpuMulSplits.end());
    return true;
}

bool vpux::VPUIP::DpuTiler::tileOverH(int64_t numDPU) {
    // FIXME: find the optimal number of tiles
    const int64_t minTileSize = 1;

    const int64_t minTilesCount = 1;
    const int64_t maxTilesCount = numDPU;

    int64_t tilesCount = _outShape[Dims4D::Act::H] / minTileSize;
    tilesCount = std::min(std::max(tilesCount, minTilesCount), maxTilesCount);

    Shape nTilesOnDim(_outShape.size(), minTilesCount);
    nTilesOnDim[Dims4D::Act::H] = tilesCount;

    const auto outTiles = fillDividedTiles(nTilesOnDim, _outShape);
    _splitPool.push_back(std::move(outTiles));

    return true;
}

bool vpux::VPUIP::DpuTiler::tileOverZ(uint32_t splitNumber, SmallVector<uint32_t> validZTiles, bool sparse,
                                      bool has_se) {
    VPUX_UNUSED(sparse);
    VPUX_UNUSED(has_se);
    OutputTiling outputTiles;
    if (_outShape.size() < 2 || splitNumber == 0) {
        return false;
    }

    auto W = _outShape[Dims4D::Act::W];
    auto H = _outShape[Dims4D::Act::H];
    auto C = _outShape.size() >= 3 ? _outShape[Dims4D::Act::C] : 0;

    auto maxChannelPerWL = divRoundUp(static_cast<uint32_t>(C), splitNumber);
    std::set<uint32_t> validZTileSet(validZTiles.begin(), validZTiles.end());
    if (!validZTiles.empty()) {
        for (const auto& zTile : validZTileSet) {
            maxChannelPerWL = padRoundUp(maxChannelPerWL, zTile);
            if (validZTileSet.find(maxChannelPerWL) != validZTileSet.end()) {
                break;
            }
        }
    } else {
        maxChannelPerWL = padRoundUp(maxChannelPerWL, DEFAULT_ZTILE_VALUE);
    }
    if (maxChannelPerWL < DEFAULT_ZTILE_VALUE) {
        return false;
    }
    Shape originalShape(4, 1);
    originalShape[Dims4D::Act::W] = W;
    originalShape[Dims4D::Act::H] = H;
    auto bestPadding = selectPadding(originalShape);

    //    SmallVector<DpuTile> dpuTiles;
    //    dpuTiles.reserve(splitNumber);
    uint32_t remainedChannel = static_cast<uint32_t>(C);
    for (uint32_t idx = 0; idx < splitNumber; idx++) {
        TileInfo outTile(_outShape.size());
        outTile.shape[Dims4D::Act::W] = W;
        outTile.shape[Dims4D::Act::H] = H;
        if (remainedChannel > maxChannelPerWL) {
            outTile.shape[Dims4D::Act::C] = maxChannelPerWL;
            remainedChannel -= maxChannelPerWL;
        } else {
            outTile.shape[Dims4D::Act::C] = remainedChannel;
            remainedChannel = 0;
        }
        outTile.offsets[Dims4D::Act::W] = 0;
        outTile.offsets[Dims4D::Act::H] = 0;
        outTile.offsets[Dims4D::Act::C] = idx * maxChannelPerWL;
        if (outTile.shape[Dims4D::Act::C] % DEFAULT_ZTILE_VALUE != 0) {
            return false;
        }
        if (!validZTiles.empty()) {
            if (validZTileSet.find(static_cast<uint32_t>(outTile.shape[Dims4D::Act::C])) == validZTileSet.end()) {
                return false;
            }
        }
        outputTiles.push_back(std::move(outTile));

        if (remainedChannel == 0) {
            break;
        }
    }

    _splitPool.push_back(std::move(outputTiles));
    return true;
}

uint32_t vpux::VPUIP::DpuTiler::simpleCost(const OutputTiling& dpuTiles, const WorkloadCostParams& params) {
    VPUX_THROW_WHEN(params.kernelSize.size() < 2, "kernel array size less than 2");
    VPUX_THROW_WHEN(params.kernelStride.size() < 2, "kernel stride array size less than 2");

    uint32_t max_cost = 0;
    for (const auto& dpuTile : dpuTiles) {
        const auto W = dpuTile.shape[Dims4D::Act::W];
        const auto H = dpuTile.shape[Dims4D::Act::H];
        const auto C = dpuTile.shape[Dims4D::Act::C];
        uint32_t cost = 0;

        if (vpux::VPU::stringifyMPEMode(params.mpeMode) == "VECTOR_FP16") {
            cost = ceil(W / 1.0) * ceil(H / 4.0) * ceil(C / 16.0);
        } else if (vpux::VPU::stringifyMPEMode(params.mpeMode) == "VECTOR") {
            cost = ceil(W / 1.0) * ceil(H / 16.0) * ceil(C / 16.0);
        } else if (vpux::VPU::stringifyMPEMode(params.mpeMode) == "MATRIX") {
            cost = ceil(W / 4.0) * ceil(H / 4.0) * ceil(C / 16.0);
        } else {
            VPUX_THROW("Failed to support mpeMode");
        }

        max_cost = std::max(max_cost, cost);
        // llvm::outs() << "workload split " << i << ":{\n";
        // llvm::outs() << "output:[" << outputTensor.x() << "," << outputTensor.y() << "," << outputTensor.z() << "],";
        // llvm::outs() << "kernel:[" << KX << "," << KY << "],";
        // llvm::outs() << "stride :[" << SX << "," << SY << "],";
        // llvm::outs() << "pad :[" << padsTileConf.top << "," << padsTileConf.bottom << "," << padsTileConf.left << ","
        //              << padsTileConf.right << "],";
        // llvm::outs() << "score:" << workloadCost.back() << "\n";
        // llvm::outs() << "}\n";
    }
    return max_cost;
}

SmallVector<OutputTiling> vpux::VPUIP::DpuTiler::getSplitPool() {
    return _splitPool;
}

SmallVector<uint32_t> vpux::VPUIP::DpuTiler::getSplitNumberPool() {
    return _splitNumberPool;
}

SmallVector<std::pair<uint8_t, uint8_t>> vpux::VPUIP::DpuTiler::getModes() {
    SmallVector<std::pair<uint8_t, uint8_t>> modes;
    for (auto mode : _mpeModeList) {
        switch (mode) {
        case VPU::MPEMode::MATRIX:
            modes.push_back({4, 4});
            break;
        case VPU::MPEMode::VECTOR:
            modes.push_back({1, 16});
            break;
        case VPU::MPEMode::VECTOR_FP16:
            modes.push_back({1, 4});
            break;
        case VPU::MPEMode::CUBOID_16x16:
            modes.push_back({16, 16});
            break;
        default:
            break;
        }
    }
    return modes;
}

Shape VPUIP::DpuTiler::selectPadding(ShapeRef original) {
    auto modeList = getModes();
    VPUX_THROW_WHEN(modeList.empty(), "Mode list empty!");
    double bestEfficiency = 0.0;
    Shape bestMode(original.size(), 1);
    for (auto& mode : modeList) {
        Shape padded(original.size(), 1);
        padded[Dims4D::Act::H] = padRoundUp(original[Dims4D::Act::H], static_cast<int64_t>(mode.first));
        padded[Dims4D::Act::W] = padRoundUp(original[Dims4D::Act::W], static_cast<int64_t>(mode.second));
        auto efficiency = splitEfficiency(original, padded);
        if (bestEfficiency < efficiency) {
            bestEfficiency = efficiency;
            bestMode[Dims4D::Act::H] = mode.first;
            bestMode[Dims4D::Act::W] = mode.second;
        }
    }
    return bestMode;
}
