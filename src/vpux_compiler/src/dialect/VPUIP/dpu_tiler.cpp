//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/dpu_tiler.hpp"
#include "vpux/compiler/utils/factors.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <numeric>
#include <set>

using namespace vpux;

namespace {

constexpr uint32_t DEFAULT_ZTILE_VALUE = 16;
constexpr size_t MIN_VALID_ZTILE_EXPONENT = 4;
constexpr size_t MAX_VALID_ZTILE_EXPONENT = 8;

// FIXME: POR runtime currently doesn't enable management tasks feature which will leads to a big overhead for large
// number of workloads. Estimate runtime overhead by this value. For further development, please refer comments in
// E#24298
constexpr size_t RUNTIME_OVERHEAD_PER_WORKLOAD = 105;

VPUNN::ExecutionMode getExecutionMode(VPU::MPEMode mpeMode) {
    switch (mpeMode) {
    case VPU::MPEMode::VECTOR:
        return VPUNN::ExecutionMode::VECTOR;
    case VPU::MPEMode::MATRIX:
        return VPUNN::ExecutionMode::MATRIX;
    case VPU::MPEMode::VECTOR_FP16:
        return VPUNN::ExecutionMode::VECTOR_FP16;
    case VPU::MPEMode::CUBOID_16x16:
        return VPUNN::ExecutionMode::CUBOID_16x16;
    case VPU::MPEMode::CUBOID_8x16:
        return VPUNN::ExecutionMode::CUBOID_8x16;
    case VPU::MPEMode::CUBOID_4x16:
        return VPUNN::ExecutionMode::CUBOID_4x16;
    default:
        VPUX_THROW("Unsupported MPE mode type: '{0}'", mpeMode);
    }
}

VPUNN::Operation getOperationType(VPUIP::NCETaskType taskType) {
    switch (taskType) {
    case VPUIP::NCETaskType::CONV:
        return VPUNN::Operation::CONVOLUTION;
    case VPUIP::NCETaskType::DWCONV:
        return VPUNN::Operation::DW_CONVOLUTION;
    case VPUIP::NCETaskType::MAXPOOL:
        return VPUNN::Operation::MAXPOOL;
    case VPUIP::NCETaskType::AVEPOOL:
        return VPUNN::Operation::AVEPOOL;
    case VPUIP::NCETaskType::ELTWISE:
        return VPUNN::Operation::ELTWISE;
    case VPUIP::NCETaskType::CMCONV:
        return VPUNN::Operation::CM_CONVOLUTION;
    // unsupported type for vpunn, use convolution as work around
    case VPUIP::NCETaskType::IDENTITY:
    case VPUIP::NCETaskType::FCL:
        return VPUNN::Operation::CONVOLUTION;
    default:
        VPUX_THROW("Unsupported operation type: '{0}'", taskType);
    }
}

VPUNN::DataType getElementType(mlir::Type type) {
    if (type.isBF16()) {
        return VPUNN::DataType::BFLOAT16;
    } else if (type.isF16()) {
        return VPUNN::DataType::FLOAT16;
    } else if (type.isInteger(CHAR_BIT * sizeof(int8_t))) {
        return VPUNN::DataType::INT8;
    } else if (type.isUnsignedInteger(CHAR_BIT * sizeof(int8_t))) {
        return VPUNN::DataType::UINT8;
    } else if (auto qType = type.dyn_cast<mlir::quant::QuantizedType>()) {
        if (qType.getStorageTypeIntegralWidth() == 8) {
            return qType.isSigned() ? VPUNN::DataType::INT8 : VPUNN::DataType::UINT8;
        }
    }
    VPUX_THROW("Unsupported data type: '{0}'", type);
}

VPUNN::VPUDevice getVPUDeviceType(VPU::ArchKind archKind) {
    switch (archKind) {
    case VPU::ArchKind::VPUX30XX:
    case VPU::ArchKind::VPUX311X:
        return VPUNN::VPUDevice::VPU_2_0;
    case VPU::ArchKind::VPUX37XX:
        return VPUNN::VPUDevice::VPU_2_7;
    default:
        VPUX_THROW("Unsupported VPU arch type: '{0}'", archKind);
    }
}

VPUNN::VPUTensor getVPUTensor(ShapeRef shape, mlir::Type elemType) {
    return VPUNN::VPUTensor(
            {
                    static_cast<unsigned int>(shape[Dims4D::Act::W]),  //
                    static_cast<unsigned int>(shape[Dims4D::Act::H]),  //
                    static_cast<unsigned int>(shape[Dims4D::Act::C]),  //
                    static_cast<unsigned int>(shape[Dims4D::Act::N]),  //
            },
            getElementType(elemType));
}

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

/*
 * Split workloads on both dimension H and W evenly. Each workload will try to make shape value align with the mpe
 * mode on the splitting dimension. For example, output tensor is [1, 16, 12, 12],  mpe mode is 4x4, and we tried to
 * split workloads into 5x5 pieces. On dimension W, max width = 3 = 12/5 (width/splitNumber) doesn't align with mpe mode
 * value(4), so the max workload width will be changed to 4 (alignVal(3,4)), and the split number will be 3 instead of 5
 * then. So does for dimension H.
 *                            --------------------------------
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 *                            --------------------------------
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 *   [0 1 2 ...  11]|  --->   | [0 - 3] | [4 - 7] | [8 - 11]|
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 *                            --------------------------------
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 *   [0 1 2 ...  11]|         | [0 - 3] | [4 - 7] | [8 - 11]|
 */

OutputTiling createDpuTilesOverHW(ShapeRef shape, int64_t widthFactor, int64_t heightFactor,
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

void vpux::VPUIP::DpuTiler::tileOverH(int64_t numDPU) {
    const int64_t minTileSize = 1;

    const int64_t minTilesCount = 1;
    const int64_t maxTilesCount = numDPU;

    int64_t tilesCount = _outShape[Dims4D::Act::H] / minTileSize;
    tilesCount = std::min(std::max(tilesCount, minTilesCount), maxTilesCount);

    Shape nTilesOnDim(_outShape.size(), minTilesCount);
    nTilesOnDim[Dims4D::Act::H] = tilesCount;

    auto outTiles = fillDividedTiles(nTilesOnDim, _outShape);
    _splitPool.insert(std::move(outTiles));
}

void vpux::VPUIP::DpuTiler::tileOverZ(uint32_t splitNumber) {
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
    _splitPool.insert(std::move(outputTiles));
}

void vpux::VPUIP::DpuTiler::tileOverHW(uint32_t splitNumber, const SplitDimension splitDimension) {
    VPUX_THROW_WHEN(_outShape.size() < 2, "Invalid output shape size: {0}", _outShape.size());
    VPUX_THROW_WHEN(splitNumber == 0, "Invalid split number: {0}", splitNumber);
    auto width = _outShape[Dims4D::Act::W];
    auto height = _outShape[Dims4D::Act::H];
    auto mpeMode = getMode();

    switch (splitDimension) {
    case SplitDimension::SPLIT_OVER_H: {
        auto outputTiles = createDpuTilesOverHW(_outShape, 1, splitNumber, mpeMode);
        _splitPool.insert(std::move(outputTiles));
        break;
    }
    case SplitDimension::SPLIT_OVER_W: {
        auto outputTiles = createDpuTilesOverHW(_outShape, splitNumber, 1, mpeMode);
        _splitPool.insert(std::move(outputTiles));
        break;
    }
    case SplitDimension::SPLIT_OVER_HW: {
        auto factorList = getFactorsList(splitNumber);
        for (const auto& factor : factorList) {
            // Map factor.larger , factor.smaller -> width, height
            if (factor.larger <= width && factor.smaller <= height) {
                auto outputTiles = createDpuTilesOverHW(_outShape, factor.larger, factor.smaller, mpeMode);
                _splitPool.insert(std::move(outputTiles));
            }
            // Map factor.smaller, factor.larger -> width, height
            if (factor.smaller <= width && factor.larger <= height) {
                auto outputTiles = createDpuTilesOverHW(_outShape, factor.smaller, factor.larger, mpeMode);
                _splitPool.insert(std::move(outputTiles));
            }
        }
        break;
    }
    default:
        VPUX_THROW("Unsupported split dimension {0}", stringifySplitDimension(splitDimension));
    }
}

uint32_t vpux::VPUIP::DpuTiler::cost(const OutputTiling& dpuTiles, const WorkloadCostParams& params) {
    VPUX_THROW_WHEN(params.kernelSize.size() < 2, "Kernel array size less than 2");
    const auto KY = params.kernelSize[Dims4D::Kernel::Y.ind()];
    const auto KX = params.kernelSize[Dims4D::Kernel::X.ind()];
    VPUX_THROW_WHEN(params.kernelStride.size() < 2, "Kernel stride array size less than 2");
    const auto SY = params.kernelStride[Dims4D::Strides::Y.ind()];
    const auto SX = params.kernelStride[Dims4D::Strides::X.ind()];

    const auto opType = getOperationType(params.nceTaskType);

    std::vector<unsigned int> workloadCost;
    for (const auto& dpuTile : dpuTiles) {
        const auto padsTileConf = backInferPadsTile(dpuTile, params.fullInputShape, params.padInfo,
                                                    makeArrayRef({KY, KX}), makeArrayRef({SY, SX}));

        const auto outputTensor = getVPUTensor(dpuTile.shape, params.dataType);

        const auto IW = (outputTensor.width() - 1) * SX + KX - static_cast<unsigned int>(padsTileConf.left) -
                        static_cast<unsigned int>(padsTileConf.right);
        const auto IH = (outputTensor.height() - 1) * SY + KY - static_cast<unsigned int>(padsTileConf.top) -
                        static_cast<unsigned int>(padsTileConf.bottom);
        const auto IC = params.nceTaskType == VPUIP::NCETaskType::CONV ||
                                        params.nceTaskType == VPUIP::NCETaskType::CMCONV ||
                                        params.nceTaskType == VPUIP::NCETaskType::FCL
                                ? static_cast<unsigned int>(params.inputShape[Dims4D::Act::C])
                                : outputTensor.channels();
        const auto IN = outputTensor.batches();

        const auto inputTensor = getVPUTensor(ShapeRef({IN, IC, IH, IW}), params.dataType);

        auto result = _costModel->DPU(
                {getVPUDeviceType(params.arch),
                 opType,
                 {inputTensor},
                 {outputTensor},
                 {static_cast<unsigned int>(KX), static_cast<unsigned int>(KY)},
                 {static_cast<unsigned int>(SX), static_cast<unsigned int>(SY)},
                 {static_cast<unsigned int>(padsTileConf.top), static_cast<unsigned int>(padsTileConf.bottom),
                  static_cast<unsigned int>(padsTileConf.left), static_cast<unsigned int>(padsTileConf.right)},
                 getExecutionMode(params.mpeMode)});
        workloadCost.push_back(result);
    }
    return VPUNN::dpu_schedule(static_cast<unsigned int>(params.numDPU), workloadCost,
                               static_cast<unsigned int>(RUNTIME_OVERHEAD_PER_WORKLOAD));
}

double vpux::VPUIP::DpuTiler::simpleCost(const OutputTiling& dpuTiles, const WorkloadCostParams& params) {
    VPUX_THROW_WHEN(params.kernelSize.size() < 2, "kernel array size less than 2");
    VPUX_THROW_WHEN(params.kernelStride.size() < 2, "kernel stride array size less than 2");

    auto mpeMode = getMode();
    double max_cost = 0;

    for (const auto& dpuTile : dpuTiles) {
        const auto W = dpuTile.shape[Dims4D::Act::W];
        const auto H = dpuTile.shape[Dims4D::Act::H];
        const auto C = dpuTile.shape[Dims4D::Act::C];
        double cost = ceil(W / mpeMode.second) * ceil(H / mpeMode.first) * ceil(C / 16.0);

        max_cost = std::max(max_cost, cost);
    }
    return max_cost;
}

std::set<OutputTiling> vpux::VPUIP::DpuTiler::getSplitPool() {
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

StringLiteral vpux::VPUIP::stringifySplitDimension(SplitDimension splitDimension) {
    switch (splitDimension) {
    case SplitDimension::SPLIT_OVER_HW:
        return "SPLIT_OVER_HW";
    case SplitDimension::SPLIT_OVER_H:
        return "SPLIT_OVER_H";
    case SplitDimension::SPLIT_OVER_W:
        return "SPLIT_OVER_W";
    default:
        return "UNKNOWN";
    }
}
