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
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/PriorityQueue.h>
#include <numeric>

using namespace vpux;

namespace {

constexpr uint32_t DEFAULT_ZTILE_VALUE = 16;
constexpr size_t MIN_VALID_ZTILE_EXPONENT = 4;
constexpr size_t MAX_VALID_ZTILE_EXPONENT = 8;

SmallVector<uint32_t> getSplitsFromRange(uint32_t maxSplitRange, uint32_t maxLimit) {
    SmallVector<uint32_t> splits;
    for (uint32_t idx = 0; idx < std::log2(maxSplitRange); idx++) {
        auto powIdx = static_cast<uint32_t>(std::pow(2, idx));
        auto splitCandidate = maxSplitRange / powIdx;
        if (maxSplitRange % powIdx == 0 && splitCandidate <= maxLimit) {
            splits.push_back(splitCandidate);
        }
    }
    return splits;
}

double totalCostOnDpuCluster(int64_t numDPU, ArrayRef<double> workloadCosts) {
    VPUX_THROW_WHEN(numDPU < 1, "Wrong DPU number: {0}", numDPU);
    VPUX_THROW_WHEN(workloadCosts.empty(), "Workload cost list is empty");

    SmallVector<double> costOnDPU(numDPU, 0);
    llvm::PriorityQueue<double, SmallVector<double>, std::greater<double>> queue(costOnDPU.begin(), costOnDPU.end());
    for (const auto& cost : workloadCosts) {
        auto currentMinCost = queue.top();
        queue.pop();
        queue.push(currentMinCost + cost);
    }
    // get the max cost from queue
    auto queueSize = queue.size();
    for (size_t i = 0; i < queueSize - 1; i++) {
        queue.pop();
    }
    return queue.top();
}

struct Factors {
    int64_t larger = 0;
    int64_t smaller = 0;

    Factors() {
    }
    Factors(int64_t larger, int64_t smaller): larger(larger), smaller(smaller) {
    }
};

SmallVector<Factors> getFactorsList(int64_t n) {
    SmallVector<Factors> factors;
    auto factorMax = static_cast<int64_t>(std::ceil(std::sqrt(n)));
    for (int64_t i = 1; i <= factorMax; i++) {
        if (n % i == 0) {
            factors.emplace_back(n / i, i);  // larger, smaller
        }
    }
    return factors;
}

OutputTiling createDpuTilesOverXY(ShapeRef shape, int64_t widthFactor, int64_t heightFactor,
                                  const std::pair<uint8_t, uint8_t>& mpeMode) {
    auto width = shape[Dims4D::Act::W];
    auto height = shape[Dims4D::Act::H];

    auto maxWidth = divUp(width, widthFactor);
    maxWidth = alignVal(maxWidth, static_cast<int64_t>(mpeMode.second));
    auto actualWidthSplitsNum = divUp(width, maxWidth);
    auto maxHeight = divUp(height, heightFactor);
    maxHeight = alignVal(maxHeight, static_cast<int64_t>(mpeMode.first));
    auto actualHeightSplitsNum = divUp(height, maxHeight);

    OutputTiling outputTiles;
    auto remainedHeight = height;
    for (int64_t idx = 0; idx < actualHeightSplitsNum; idx++) {
        auto currentHeightStep = remainedHeight > maxHeight ? maxHeight : remainedHeight;
        auto remainedWidth = width;
        for (int64_t idy = 0; idy < actualWidthSplitsNum; idy++) {
            TileInfo outTile(shape);
            outTile.shape[Dims4D::Act::C] = shape[Dims4D::Act::C];
            outTile.shape[Dims4D::Act::H] = currentHeightStep;
            outTile.shape[Dims4D::Act::W] = remainedWidth > maxWidth ? maxWidth : remainedWidth;
            remainedWidth -= outTile.shape[Dims4D::Act::W];
            outTile.offsets[Dims4D::Act::H] = idx * maxHeight;
            outTile.offsets[Dims4D::Act::W] = idy * maxWidth;
            outTile.offsets[Dims4D::Act::C] = 0;
            outTile.axis[Dims4D::Act::H] = actualHeightSplitsNum;
            outTile.axis[Dims4D::Act::W] = actualWidthSplitsNum;
            outputTiles.push_back(std::move(outTile));
        }
        remainedHeight -= currentHeightStep;
    }
    return outputTiles;
}
}  // namespace

SmallVector<uint32_t> vpux::VPUIP::DpuTiler::generateSplitNumberPool(int64_t numDPU, uint32_t maxSplits) {
    SmallVector<uint32_t> validZTiles;

    // Note: refer the values from workload number pool implementation at
    // https://github.com/intel-innersource/frameworks.ai.vpu.presilicon.fathom/blob/main/src/Controllers/WorkloadGen.py#L84
    // 2^4 equals to the CMX word size in bytes,  2^8 is an up bound to limit the number of splits
    for (size_t i = MIN_VALID_ZTILE_EXPONENT; i < MAX_VALID_ZTILE_EXPONENT; ++i) {
        validZTiles.push_back(static_cast<uint32_t>(std::pow(2, i)));
        validZTiles.push_back(validZTiles.back() + DEFAULT_ZTILE_VALUE);
    }

    SmallVector<uint32_t> maxSplitsInZ;
    SmallVector<uint32_t> maxSplitsInXY;
    auto mode = getMode();
    for (const auto& zTile : validZTiles) {
        maxSplitsInZ.push_back(
                static_cast<uint32_t>(std::ceil(_outShape[Dims4D::Act::C] / static_cast<double>(zTile))));
    }
    maxSplitsInXY.push_back(
            static_cast<uint32_t>(std::ceil(_outShape[Dims4D::Act::H] / static_cast<double>(mode.first)) *
                                  std::ceil(_outShape[Dims4D::Act::W] / static_cast<double>(mode.second))));

    auto maxZ = *std::max_element(maxSplitsInZ.begin(), maxSplitsInZ.end());
    auto maxXY = *std::max_element(maxSplitsInXY.begin(), maxSplitsInXY.end());
    maxSplits = std::min<uint32_t>(maxSplits, std::max(maxZ, maxXY));
    VPUX_THROW_WHEN(maxSplits == 0, "Invalid max split number: {0}", maxSplits);

    std::set<uint32_t> dpuMulSplits;
    for (auto i = numDPU; i < maxSplits + 1; i = i + numDPU) {
        dpuMulSplits.insert(static_cast<uint32_t>(i));
    }
    for (auto splitsZ : maxSplitsInZ) {
        auto zRanges = getSplitsFromRange(splitsZ, maxSplits);
        dpuMulSplits.insert(zRanges.begin(), zRanges.end());
    }
    for (auto splitsXY : maxSplitsInXY) {
        auto xyRanges = getSplitsFromRange(splitsXY, maxSplits);
        dpuMulSplits.insert(xyRanges.begin(), xyRanges.end());
    }
    dpuMulSplits.insert(1);
    return SmallVector<uint32_t>(dpuMulSplits.begin(), dpuMulSplits.end());
}

void VPUIP::DpuTiler::tileOverH(int64_t numDPU) {
    const int64_t minTileSize = 1;

    const int64_t minTilesCount = 1;
    const int64_t maxTilesCount = numDPU;

    int64_t tilesCount = _outShape[Dims4D::Act::H] / minTileSize;
    tilesCount = std::min(std::max(tilesCount, minTilesCount), maxTilesCount);

    Shape nTilesOnDim(_outShape.size(), minTilesCount);
    nTilesOnDim[Dims4D::Act::H] = tilesCount;

    auto outTiles = fillDividedTiles(nTilesOnDim, _outShape);
    addDpuTilesIntoSplitPool(outTiles);
}

void VPUIP::DpuTiler::tileOverZ(uint32_t splitNumber) {
    VPUX_THROW_WHEN(_outShape.size() < 2, "Invalid output shape size: {0}", _outShape.size());
    VPUX_THROW_WHEN(splitNumber == 0, "Invalid split number: {0}", splitNumber);

    auto C = _outShape.size() >= 3 ? _outShape[Dims4D::Act::C] : 0;
    auto maxChannelPerWL = divUp(static_cast<uint32_t>(C), splitNumber);
    maxChannelPerWL = alignVal(maxChannelPerWL, DEFAULT_ZTILE_VALUE);
    if (maxChannelPerWL < DEFAULT_ZTILE_VALUE) {
        return;
    }
    OutputTiling outputTiles;
    auto actualSplitNumber = divUp(static_cast<uint32_t>(C), maxChannelPerWL);
    auto remainedChannel = static_cast<uint32_t>(C);
    for (uint32_t idx = 0; idx < actualSplitNumber; idx++) {
        TileInfo outTile(_outShape);
        outTile.shape[Dims4D::Act::C] = remainedChannel > maxChannelPerWL ? maxChannelPerWL : remainedChannel;
        remainedChannel -= static_cast<uint32_t>(outTile.shape[Dims4D::Act::C]);
        outTile.offsets[Dims4D::Act::W] = 0;
        outTile.offsets[Dims4D::Act::H] = 0;
        outTile.offsets[Dims4D::Act::C] = idx * maxChannelPerWL;
        outTile.axis[Dims4D::Act::C] = actualSplitNumber;
        if (outTile.shape[Dims4D::Act::C] % DEFAULT_ZTILE_VALUE != 0) {
            return;
        }
        outputTiles.push_back(std::move(outTile));
    }
    addDpuTilesIntoSplitPool(outputTiles);
}

void VPUIP::DpuTiler::tileOverXY(uint32_t splitNumber) {
    VPUX_THROW_WHEN(_outShape.size() < 2, "Invalid output shape size: {0}", _outShape.size());
    VPUX_THROW_WHEN(splitNumber == 0, "Invalid split number: {0}", splitNumber);

    auto width = _outShape[Dims4D::Act::W];
    auto height = _outShape[Dims4D::Act::H];
    auto factorList = getFactorsList(splitNumber);
    auto mpeMode = getMode();
    for (const auto& factor : factorList) {
        // Map factor.larger , factor.smaller -> width, height
        if (factor.larger <= width && factor.smaller <= height) {
            auto outputTiles = createDpuTilesOverXY(_outShape, factor.larger, factor.smaller, mpeMode);
            addDpuTilesIntoSplitPool(outputTiles);
        }
        // Map factor.smaller, factor.larger -> width, height
        if (factor.smaller <= width && factor.larger <= height) {
            auto outputTiles = createDpuTilesOverXY(_outShape, factor.smaller, factor.larger, mpeMode);
            addDpuTilesIntoSplitPool(outputTiles);
        }
    }
}

double vpux::VPUIP::DpuTiler::simpleCost(int64_t numDPU, const OutputTiling& dpuTiles) {
    auto mpeMode = getMode();
    SmallVector<double> workloadCosts;
    for (const auto& dpuTile : dpuTiles) {
        const auto W = dpuTile.shape[Dims4D::Act::W];
        const auto H = dpuTile.shape[Dims4D::Act::H];
        const auto C = dpuTile.shape[Dims4D::Act::C];
        workloadCosts.push_back(ceil(W / mpeMode.second) * ceil(H / mpeMode.first) * ceil(C / 16.0));
    }
    return totalCostOnDpuCluster(numDPU, workloadCosts);
}

SmallVector<OutputTiling> vpux::VPUIP::DpuTiler::getSplitPool() {
    return _splitPool;
}

std::pair<uint8_t, uint8_t> vpux::VPUIP::DpuTiler::getMode() {
    switch (_mpeMode) {
    case VPU::MPEMode::MATRIX:
        return {4, 4};
    case VPU::MPEMode::VECTOR:
        return {1, 16};
    case VPU::MPEMode::VECTOR_FP16:
        return {1, 4};
    case VPU::MPEMode::CUBOID_16x16:
        return {16, 16};
    case VPU::MPEMode::CUBOID_8x16:
        return {8, 16};
    case VPU::MPEMode::CUBOID_4x16:
        return {4, 16};
    default:
        VPUX_THROW("Unsupported MPE mode {0}", _mpeMode);
    }
}

void vpux::VPUIP::DpuTiler::addDpuTilesIntoSplitPool(OutputTiling& outputTiling) {
    auto isSame = [](const OutputTiling& lhs, const OutputTiling& rhs) {
        if (lhs.size() != rhs.size()) {
            return false;
        }
        for (size_t i = 0; i < lhs.size(); i++) {
            if (lhs[i] != rhs[i]) {
                return false;
            }
        }
        return true;
    };
    if (llvm::find_if(_splitPool, [&](const OutputTiling& elem) {
            return isSame(elem, outputTiling);
        }) == _splitPool.end()) {
        _splitPool.push_back(std::move(outputTiling));
    }
}
