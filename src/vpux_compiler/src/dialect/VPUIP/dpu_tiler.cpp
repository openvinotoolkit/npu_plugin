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
#include <vpu_cost_model.h>
#include <functional>
#include <numeric>
#include <set>

using namespace vpux;

namespace {
std::unique_ptr<VPUNN::VPUCostModel> gCostModel{nullptr};

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
    case VPU::ArchKind::KMB:
    case VPU::ArchKind::TBH:
        return VPUNN::VPUDevice::VPU_2_0;
    case VPU::ArchKind::MTL:
        return VPUNN::VPUDevice::VPU_2_7;
    default:
        VPUX_THROW("Unsupported VPU arch type: '{0}'", archKind);
    }
}

VPUNN::VPUTensor getOutputTensor(const TileInfo& tileInfo, VPUNN::DataType dataType) {
    return VPUNN::VPUTensor({static_cast<unsigned int>(tileInfo.shape[Dims4D::Act::W]),
                             static_cast<unsigned int>(tileInfo.shape[Dims4D::Act::H]),
                             static_cast<unsigned int>(tileInfo.shape[Dims4D::Act::C]), 1},
                            dataType);
}

bool isCostModelInited() {
    return gCostModel != nullptr;
}

void initCostModel(VPU::ArchKind archKind) {
    SmallString modelDir;
    // probe for OV_BUILD_DIR
    auto ovBuildDir = InferenceEngine::getIELibraryPath();

    modelDir = ovBuildDir;
    llvm::sys::path::append(modelDir, "vpunnd");
    switch (archKind) {
    case VPU::ArchKind::KMB:
    case VPU::ArchKind::TBH:
        llvm::sys::path::append(modelDir, "vpu_2_0.vpunn");
        break;
    case VPU::ArchKind::MTL:
        llvm::sys::path::append(modelDir, "vpu_2_7.vpunn");
        break;
    default:
        VPUX_THROW("Unsupported VPU arch type: '{0}'", archKind);
    }

    VPUX_THROW_UNLESS(llvm::sys::fs::exists(modelDir), "vpunn model {0} does not exist", modelDir);
    gCostModel = std::make_unique<VPUNN::VPUCostModel>(modelDir.str().str());
}

static const uint32_t DEFAULT_ZTILE_VALUE = 16;
template <typename T>
T divRoundUp(T x, T m) {
    return (x + m - 1) / m;
}

template <typename T>
T padRoundUp(T x, T m) {
    return divRoundUp(x, m) * m;
}

SmallVector<uint32_t> getSplitsFromRange(uint32_t maxSplitRange, uint32_t maxLimit) {
    SmallVector<uint32_t> splits;
    for (uint32_t idx = 0; idx < std::log2(maxSplitRange); idx++) {
        auto powIdx = static_cast<uint32_t>(std::pow(2, idx));
        if (maxSplitRange % powIdx == 0 && maxSplitRange / powIdx <= maxLimit) {
            splits.push_back(maxSplitRange / powIdx);
        }
    }
    return splits;
}
}  // namespace

bool vpux::VPUIP::DpuTiler::generateSplitNumberPool(int64_t numDPU, uint32_t maxSplits,
                                                    ArrayRef<uint32_t> supportedZTiles) {
    SmallVector<uint32_t> validZTiles(supportedZTiles.begin(), supportedZTiles.end());
    if (validZTiles.empty()) {
        for (size_t i = 4; i < 8; ++i) {
            validZTiles.push_back(static_cast<uint32_t>(std::pow(2, i)));
            validZTiles.push_back(validZTiles.back() + DEFAULT_ZTILE_VALUE);
        }
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
    if (maxSplits == 0) {
        return false;
    }

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
    _splitNumberPool = SmallVector<uint32_t>(dpuMulSplits.begin(), dpuMulSplits.end());
    return true;
}

void vpux::VPUIP::DpuTiler::tileOverH(int64_t numDPU) {
    const int64_t minTileSize = 1;

    const int64_t minTilesCount = 1;
    const int64_t maxTilesCount = numDPU;

    int64_t tilesCount = _outShape[Dims4D::Act::H] / minTileSize;
    tilesCount = std::min(std::max(tilesCount, minTilesCount), maxTilesCount);

    Shape nTilesOnDim(_outShape.size(), minTilesCount);
    nTilesOnDim[Dims4D::Act::H] = tilesCount;

    const auto outTiles = fillDividedTiles(nTilesOnDim, _outShape);
    _splitPool.push_back(std::move(outTiles));
}

bool vpux::VPUIP::DpuTiler::tileOverZ(uint32_t splitNumber, SmallVector<uint32_t> validZTiles) {
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

uint32_t vpux::VPUIP::DpuTiler::cost(const OutputTiling& dpuTiles, const WorkloadCostParams& params) {
    if (!isCostModelInited()) {
        initCostModel(params.arch);
    }
    const auto opType = getOperationType(params.nceTaskType);
    const auto elemType = getElementType(params.dataType);
    VPUX_THROW_WHEN(params.kernelSize.size() < 2, "kernel array size less than 2");
    const auto KY = static_cast<unsigned int>(params.kernelSize[0]);
    const auto KX = static_cast<unsigned int>(params.kernelSize[1]);

    VPUX_THROW_WHEN(params.kernelStride.size() < 2, "kernel stride array size less than 2");
    const auto SY = static_cast<unsigned int>(params.kernelStride[0]);
    const auto SX = static_cast<unsigned int>(params.kernelStride[1]);

    std::vector<unsigned int> workloadCost;
    for (const auto& dpuTile : dpuTiles) {
        const auto padsTileConf = backInferPadsTile(dpuTile, params.outputShape, params.padInfo);
        VPUNN::VPUTensor outputTensor = getOutputTensor(dpuTile, elemType);
        VPUNN::VPUTensor inputTensor({(outputTensor.x() - 1) * SX + KX - static_cast<unsigned int>(padsTileConf.left) -
                                              static_cast<unsigned int>(padsTileConf.right),
                                      (outputTensor.y() - 1) * SY + KY - static_cast<unsigned int>(padsTileConf.top) -
                                              static_cast<unsigned int>(padsTileConf.bottom),
                                      static_cast<unsigned int>(params.inputShape[Dims4D::Act::C]), 1},
                                     elemType);

        workloadCost.push_back(gCostModel->DPU(
                {getVPUDeviceType(params.arch),
                 opType,
                 {inputTensor},
                 {outputTensor},
                 {KX, KY},
                 {SX, SY},
                 {static_cast<unsigned int>(padsTileConf.top), static_cast<unsigned int>(padsTileConf.bottom),
                  static_cast<unsigned int>(padsTileConf.left), static_cast<unsigned int>(padsTileConf.right)},
                 getExecutionMode(params.mpeMode)}));
    }
    return VPUNN::dpu_schedule(static_cast<unsigned int>(params.numDPU), workloadCost);
}

double vpux::VPUIP::DpuTiler::simpleCost(const OutputTiling& dpuTiles, const WorkloadCostParams& params) {
    if (!isCostModelInited()) {
        initCostModel(params.arch);
    }

    VPUX_THROW_WHEN(params.kernelSize.size() < 2, "kernel array size less than 2");
    VPUX_THROW_WHEN(params.kernelStride.size() < 2, "kernel stride array size less than 2");

    auto mpeMode = getMode();
    double max_cost = 0;

    for (const auto& dpuTile : dpuTiles) {
        const auto W = dpuTile.shape[Dims4D::Act::W];
        const auto H = dpuTile.shape[Dims4D::Act::H];
        const auto C = dpuTile.shape[Dims4D::Act::C];
        double cost = ceil(W / mpeMode.first) * ceil(H / mpeMode.second) * ceil(C / 16.0);

        max_cost = std::max(max_cost, cost);
    }
    return max_cost;
}

SmallVector<OutputTiling> vpux::VPUIP::DpuTiler::getSplitPool() {
    return _splitPool;
}

SmallVector<uint32_t> vpux::VPUIP::DpuTiler::getSplitNumberPool() {
    return _splitNumberPool;
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
