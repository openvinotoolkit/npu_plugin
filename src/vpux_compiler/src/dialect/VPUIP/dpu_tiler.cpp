//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/dpu_tiler.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/factors.hpp"

#include "vpux/utils/core/numeric.hpp"

#include <numeric>
#include <set>

using namespace vpux;

namespace {

constexpr int64_t DEFAULT_ZTILE_VALUE = 16;
constexpr int64_t MIN_VALID_ZTILE_EXPONENT = 4;
constexpr int64_t MAX_VALID_ZTILE_EXPONENT = 8;

// FIXME: POR runtime currently doesn't enable management tasks feature which will leads to a big overhead for large
// number of workloads. Estimate runtime overhead by this value. For further development, please refer comments in
// E#24298
constexpr int64_t RUNTIME_OVERHEAD_PER_WORKLOAD = 105;

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

SmallVector<int64_t> getSplitsFromRange(int64_t maxSplitRange, int64_t maxLimit) {
    SmallVector<int64_t> splits;
    for (int64_t idx = 0; idx < std::log2(maxSplitRange); idx++) {
        auto powIdx = static_cast<int64_t>(std::pow(2, idx));
        auto splitCandidate = maxSplitRange / powIdx;
        if (maxSplitRange % powIdx == 0 && splitCandidate <= maxLimit) {
            splits.push_back(splitCandidate);
        }
    }
    return splits;
}

using MpeModeSize = std::pair<int64_t, int64_t>;

MpeModeSize getMpeModeSize(VPU::MPEMode mpeMode) {
    switch (mpeMode) {
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
        VPUX_THROW("Unsupported MPE mode {0}", mpeMode);
    }
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

VPUIP::WorkloadSplit createWorkloadSplitOverHW(ShapeRef shape, int64_t widthFactor, int64_t heightFactor,
                                               VPU::MPEMode mpeMode) {
    const auto mpeModeSize = getMpeModeSize(mpeMode);

    auto width = shape[Dims4D::Act::W];
    auto height = shape[Dims4D::Act::H];

    auto maxWidth = divUp(width, widthFactor);
    maxWidth = alignVal(maxWidth, mpeModeSize.second);
    auto actualWidthSplitsNum = divUp(width, maxWidth);
    auto maxHeight = divUp(height, heightFactor);
    maxHeight = alignVal(maxHeight, mpeModeSize.first);
    auto actualHeightSplitsNum = divUp(height, maxHeight);

    VPUIP::WorkloadSplit split;

    auto remainedHeight = height;
    for (int64_t idx = 0; idx < actualHeightSplitsNum; idx++) {
        auto currentHeightStep = remainedHeight > maxHeight ? maxHeight : remainedHeight;

        auto remainedWidth = width;
        for (int64_t idy = 0; idy < actualWidthSplitsNum; idy++) {
            TileInfo outTile(shape);
            outTile.shape[Dims4D::Act::C] = shape[Dims4D::Act::C];
            outTile.shape[Dims4D::Act::H] = currentHeightStep;
            outTile.shape[Dims4D::Act::W] = remainedWidth > maxWidth ? maxWidth : remainedWidth;
            outTile.offsets[Dims4D::Act::H] = idx * maxHeight;
            outTile.offsets[Dims4D::Act::W] = idy * maxWidth;
            outTile.offsets[Dims4D::Act::C] = 0;
            outTile.axis[Dims4D::Act::H] = actualHeightSplitsNum;
            outTile.axis[Dims4D::Act::W] = actualWidthSplitsNum;

            remainedWidth -= outTile.shape[Dims4D::Act::W];

            split.emplace_back(std::move(outTile), mpeMode);
        }

        remainedHeight -= currentHeightStep;
    }

    return split;
}

}  // namespace

vpux::VPUIP::DpuTiler::DpuTiler(ShapeRef outShape, VPU::MPEMode mpeMode): _outShape(outShape.raw()), _mpeMode(mpeMode) {
    VPUX_THROW_WHEN(outShape.size() != 4, "Only 4D shape is supported, got '{0}'", outShape);
}

SmallVector<int64_t> vpux::VPUIP::DpuTiler::generateSplitNumberPool(int64_t numDPU, int64_t maxSplits) const {
    SmallVector<int64_t> validZTiles;

    // Note: refer the values from workload number pool implementation in presilicon setting by
    // ./blob/main/src/Controllers/WorkloadGen.py#L84
    // 2^4 equals to the CMX word size in bytes,  2^8 is an up bound to limit the number of splits
    for (int64_t i = MIN_VALID_ZTILE_EXPONENT; i < MAX_VALID_ZTILE_EXPONENT; ++i) {
        validZTiles.push_back(static_cast<int64_t>(std::pow(2, i)));
        validZTiles.push_back(validZTiles.back() + DEFAULT_ZTILE_VALUE);
    }

    SmallVector<int64_t> maxSplitsInZ;
    SmallVector<int64_t> maxSplitsInXY;
    const auto mpeModeSize = getMpeModeSize(_mpeMode);

    for (const auto& zTile : validZTiles) {
        maxSplitsInZ.push_back(static_cast<int64_t>(std::ceil(_outShape[Dims4D::Act::C] / static_cast<double>(zTile))));
    }

    maxSplitsInXY.push_back(
            static_cast<int64_t>(std::ceil(_outShape[Dims4D::Act::H] / static_cast<double>(mpeModeSize.first)) *
                                 std::ceil(_outShape[Dims4D::Act::W] / static_cast<double>(mpeModeSize.second))));

    auto maxZ = *std::max_element(maxSplitsInZ.begin(), maxSplitsInZ.end());
    auto maxXY = *std::max_element(maxSplitsInXY.begin(), maxSplitsInXY.end());

    maxSplits = std::min(maxSplits, std::max(maxZ, maxXY));
    VPUX_THROW_WHEN(maxSplits == 0, "Invalid max split number: {0}", maxSplits);

    std::set<int64_t> dpuMulSplits;
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
    return to_small_vector(dpuMulSplits);
}

void vpux::VPUIP::DpuTiler::tileOverH(int64_t numDPU, WorkloadSplitPool& splitPool) {
    const int64_t minTileSize = 1;

    const int64_t minTilesCount = 1;
    const int64_t maxTilesCount = numDPU;

    int64_t tilesCount = _outShape[Dims4D::Act::H] / minTileSize;
    tilesCount = std::min(std::max(tilesCount, minTilesCount), maxTilesCount);

    Shape nTilesOnDim(_outShape.size(), minTilesCount);
    nTilesOnDim[Dims4D::Act::H] = tilesCount;

    auto outTiles = fillDividedTiles(nTilesOnDim, _outShape);

    VPUIP::WorkloadSplit split;
    for (auto& outTile : outTiles) {
        split.emplace_back(std::move(outTile), _mpeMode);
    }

    splitPool.insert(std::move(split));
}

void vpux::VPUIP::DpuTiler::tileOverZ(int64_t splitNumber, WorkloadSplitPool& splitPool, bool requiresEqualZ) {
    VPUX_THROW_WHEN(splitNumber == 0, "Invalid split number: {0}", splitNumber);

    const auto C = _outShape[Dims4D::Act::C];
    auto maxChannelPerWL = divUp(C, splitNumber);
    maxChannelPerWL = alignVal(maxChannelPerWL, DEFAULT_ZTILE_VALUE);
    if (maxChannelPerWL < DEFAULT_ZTILE_VALUE) {
        return;
    }
    if (requiresEqualZ && (maxChannelPerWL * splitNumber) != C) {
        return;
    }

    VPUIP::WorkloadSplit split;

    auto actualSplitNumber = divUp(C, maxChannelPerWL);
    auto remainedChannels = C;

    for (int64_t idx = 0; idx < actualSplitNumber; idx++) {
        TileInfo outTile(_outShape);
        outTile.shape[Dims4D::Act::C] = remainedChannels > maxChannelPerWL ? maxChannelPerWL : remainedChannels;
        outTile.offsets[Dims4D::Act::W] = 0;
        outTile.offsets[Dims4D::Act::H] = 0;
        outTile.offsets[Dims4D::Act::C] = idx * maxChannelPerWL;
        outTile.axis[Dims4D::Act::C] = actualSplitNumber;

        if (outTile.shape[Dims4D::Act::C] % DEFAULT_ZTILE_VALUE != 0) {
            return;
        }

        remainedChannels -= outTile.shape[Dims4D::Act::C];

        split.emplace_back(std::move(outTile), _mpeMode);
    }

    splitPool.insert(std::move(split));
}

void vpux::VPUIP::DpuTiler::tileOverHW(int64_t splitNumber, SplitDimension splitDimension,
                                       WorkloadSplitPool& splitPool) {
    VPUX_THROW_WHEN(splitNumber == 0, "Invalid split number: {0}", splitNumber);

    auto width = _outShape[Dims4D::Act::W];
    auto height = _outShape[Dims4D::Act::H];

    switch (splitDimension) {
    case SplitDimension::SPLIT_OVER_H: {
        splitPool.emplace(createWorkloadSplitOverHW(_outShape, 1, splitNumber, _mpeMode));
        break;
    }
    case SplitDimension::SPLIT_OVER_W: {
        splitPool.emplace(createWorkloadSplitOverHW(_outShape, splitNumber, 1, _mpeMode));
        break;
    }
    case SplitDimension::SPLIT_OVER_HW: {
        const auto factorList = getFactorsList(splitNumber);

        for (const auto factor : factorList) {
            // Map factor.larger , factor.smaller -> width, height
            if (factor.larger <= width && factor.smaller <= height) {
                splitPool.emplace(createWorkloadSplitOverHW(_outShape, factor.larger, factor.smaller, _mpeMode));
            }

            // Map factor.smaller, factor.larger -> width, height
            if (factor.smaller <= width && factor.larger <= height) {
                splitPool.emplace(createWorkloadSplitOverHW(_outShape, factor.smaller, factor.larger, _mpeMode));
            }
        }
        break;
    }
    default:
        VPUX_THROW("Unsupported split dimension {0}", splitDimension);
    }
}

void vpux::VPUIP::DpuTiler::tileOverHWMixedPrecision(WorkloadSplitPool& splitPool) {
    VPUX_THROW_WHEN(_mpeMode != VPU::MPEMode::VECTOR_FP16, "Invalid MPE mode, actual: {0}, expected: {1}", _mpeMode,
                    VPU::MPEMode::VECTOR_FP16);
    VPUIP::WorkloadSplit split;
    split.emplace_back(TileInfo(_outShape), _mpeMode);
    splitPool.insert(std::move(split));
}

int64_t vpux::VPUIP::computeSplitCost(const WorkloadSplit& split, const WorkloadCostParams& params,
                                      const std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    VPUX_THROW_WHEN(params.kernelSize.size() < 2, "Kernel array size less than 2");
    const auto KY = params.kernelSize[Dims4D::Kernel::Y.ind()];
    const auto KX = params.kernelSize[Dims4D::Kernel::X.ind()];

    VPUX_THROW_WHEN(params.kernelStride.size() < 2, "Kernel stride array size less than 2");
    const auto SY = params.kernelStride[Dims4D::Strides::Y.ind()];
    const auto SX = params.kernelStride[Dims4D::Strides::X.ind()];

    const auto opType = getOperationType(params.nceTaskType);

    std::vector<int64_t> workloadCost;
    workloadCost.reserve(split.size());

    for (const auto& wl : split) {
        const auto& outputTile = std::get<0>(wl);
        const auto mpeMode = std::get<1>(wl);

        const auto padsTileConf = backInferPadsTile(outputTile, params.fullInputShape, params.padInfo,
                                                    makeArrayRef({KY, KX}), makeArrayRef({SY, SX}));

        const auto outputTensor = getVPUTensor(outputTile.shape, params.dataType);

        const auto IW = (outputTile.shape[Dims4D::Act::W] - 1) * SX + KX - padsTileConf.left - padsTileConf.right;
        const auto IH = (outputTile.shape[Dims4D::Act::H] - 1) * SY + KY - padsTileConf.top - padsTileConf.bottom;
        const auto IC = params.nceTaskType == VPUIP::NCETaskType::CONV ||
                                        params.nceTaskType == VPUIP::NCETaskType::CMCONV ||
                                        params.nceTaskType == VPUIP::NCETaskType::FCL
                                ? params.inputShape[Dims4D::Act::C]
                                : outputTile.shape[Dims4D::Act::C];
        const auto IN = outputTile.shape[Dims4D::Act::N];

        const auto inputTensor = getVPUTensor(ShapeRef({IN, IC, IH, IW}), params.dataType);

        const auto wlCost = costModel->DPU(
                {getVPUDeviceType(params.arch),
                 opType,
                 {inputTensor},
                 {outputTensor},
                 {static_cast<unsigned int>(KX), static_cast<unsigned int>(KY)},
                 {static_cast<unsigned int>(SX), static_cast<unsigned int>(SY)},
                 {static_cast<unsigned int>(padsTileConf.top), static_cast<unsigned int>(padsTileConf.bottom),
                  static_cast<unsigned int>(padsTileConf.left), static_cast<unsigned int>(padsTileConf.right)},
                 getExecutionMode(mpeMode)});

        workloadCost.push_back(static_cast<int64_t>(wlCost));
    }

    return VPUNN::dpu_schedule(params.numDPU, workloadCost, RUNTIME_OVERHEAD_PER_WORKLOAD);
}

StringLiteral vpux::VPUIP::stringifyEnum(SplitDimension splitDimension) {
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
