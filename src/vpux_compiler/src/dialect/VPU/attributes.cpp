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

#include "vpux/compiler/dialect/VPU/attributes.hpp"

#include "vpux/compiler/dialect/IE/attributes/structs.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Identifier.h>

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Run-time resources
//

namespace {

constexpr StringLiteral derateFactorAttrName = "VPU.derateFactor";
constexpr StringLiteral bandwidthAttrName = "VPU.bandwidth";
constexpr StringLiteral processorFrequencyAttrName = "VPU.processorFrequency";

}  // namespace

StringLiteral vpux::VPU::getMemoryDerateAttrName() {
    return derateFactorAttrName;
}

StringLiteral vpux::VPU::getMemoryBandwidthAttrName() {
    return bandwidthAttrName;
}

StringLiteral vpux::VPU::getProcessorFrequencyAttrName() {
    return processorFrequencyAttrName;
}

namespace {

constexpr int KMB_MAX_DPU_GROUPS = 4;
constexpr int MTL_MAX_DPU_GROUPS = 2;

}  // namespace

uint32_t vpux::VPU::getMaxDPUClusterNum(mlir::Operation* op) {
    const auto kind = VPU::getArch(op);

    switch (kind) {
    case VPU::ArchKind::KMB:
        return KMB_MAX_DPU_GROUPS;
    case VPU::ArchKind::TBH:
        return KMB_MAX_DPU_GROUPS;
    case VPU::ArchKind::MTL:
        return MTL_MAX_DPU_GROUPS;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", kind);
    }
}

Byte vpux::VPU::getTotalCMXSize(mlir::Operation* op) {
    auto module = getTopLevelModule(op);

    auto cmxRes = IE::getAvailableMemory(module, VPU::MemoryKind::CMX_NN);
    VPUX_THROW_UNLESS(cmxRes != nullptr, "Can't get information about {0} memory", VPU::MemoryKind::CMX_NN);

    return cmxRes.size();
}

//
// ArchKind
//

namespace {

constexpr StringLiteral archAttrName = "VPU.arch";

constexpr Byte DDR_HEAP_SIZE = 500_MB;
constexpr Byte CSRAM_SIZE = 24_MB;

// See https://github.com/movidius/vpuip_2/blob/develop/system/nn/inference_runtime_common/inc/nn_cmx_memory_map.h
constexpr Byte KMB_CMX_WORKSPACE_SIZE = Byte(896_KB);

// See https://github.com/movidius/vpuip_2/blob/develop/system/nn_mtl/common_runtime/inc/nn_cmx_memory_map.h
constexpr Byte MTL_CMX_WORKSPACE_SIZE = Byte(1936_KB);

}  // namespace

void vpux::VPU::setArch(mlir::ModuleOp module, ArchKind kind, Optional<int> numOfDPUGroups) {
    VPUX_THROW_WHEN(module->hasAttr(archAttrName),
                    "Architecture is already defined, probably you run '--init-compiler' twice");

    module->setAttr(archAttrName, ArchKindAttr::get(module.getContext(), kind));

    const auto addMem = [&](MemoryKind kind, Byte size, double derateFactor, uint32_t bandwidth) {
        auto mem = IE::addAvailableMemory(module, kind, size);
        mem->setAttr(derateFactorAttrName, getFPAttr(module.getContext(), derateFactor));
        mem->setAttr(bandwidthAttrName, getIntAttr(module.getContext(), bandwidth));
    };

    const auto addExecutor = [&](ExecutorKind kind, uint32_t count) {
        return IE::addAvailableExecutor(module, kind, count);
    };

    const auto getNumOfDPUGroupsVal = [&](int maxDpuGroups) {
        int numOfDPUGroupsVal = numOfDPUGroups.hasValue() ? numOfDPUGroups.getValue() : maxDpuGroups;
        VPUX_THROW_UNLESS(1 <= numOfDPUGroupsVal && numOfDPUGroupsVal <= maxDpuGroups,
                          "Invalid number of DPU groups: '{0}'", numOfDPUGroupsVal);
        return numOfDPUGroupsVal;
    };

    IE::ExecutorResourceOp nceCluster;

    switch (kind) {
    case ArchKind::KMB: {
        addMem(MemoryKind::DDR, DDR_HEAP_SIZE, 0.6, 8);
        addMem(MemoryKind::CMX_NN, KMB_CMX_WORKSPACE_SIZE, 1.0, 32);

        addExecutor(ExecutorKind::DMA_NN, 1);
        addExecutor(ExecutorKind::SHAVE_UPA, 16);
        nceCluster = IE::addAvailableExecutor(module, ExecutorKind::NCE, getNumOfDPUGroupsVal(KMB_MAX_DPU_GROUPS));
        nceCluster.addSubExecutor(ExecutorKind::DPU, 5);

        break;
    }
    case ArchKind::TBH: {
        addMem(MemoryKind::DDR, DDR_HEAP_SIZE, 0.6, 8);
        addMem(MemoryKind::CSRAM, CSRAM_SIZE, 0.85, 64);
        addMem(MemoryKind::CMX_NN, KMB_CMX_WORKSPACE_SIZE, 1.0, 32);

        addExecutor(ExecutorKind::DMA_NN, 2);
        addExecutor(ExecutorKind::SHAVE_UPA, 16);
        nceCluster = IE::addAvailableExecutor(module, ExecutorKind::NCE, getNumOfDPUGroupsVal(KMB_MAX_DPU_GROUPS));
        nceCluster.addSubExecutor(ExecutorKind::DPU, 5);

        break;
    }
    case ArchKind::MTL: {
        addMem(MemoryKind::DDR, DDR_HEAP_SIZE, 0.6, 8);
        addMem(MemoryKind::CMX_NN, MTL_CMX_WORKSPACE_SIZE, 1.0, 32);

        addExecutor(ExecutorKind::DMA_NN, 2);
        // TODO: SHAVE_NN shouldn't be used here
        addExecutor(ExecutorKind::SHAVE_NN, 1);
        // TODO: move SHAVE_ACT as a sub-executor for NCE
        // TODO: use actual number of ACT SHAVES
        addExecutor(ExecutorKind::SHAVE_ACT, 1);
        nceCluster = IE::addAvailableExecutor(module, ExecutorKind::NCE, getNumOfDPUGroupsVal(MTL_MAX_DPU_GROUPS));
        nceCluster.addSubExecutor(ExecutorKind::DPU, 1);

        break;
    }
    default:
        VPUX_THROW("Unsupported architecture '{0}'", kind);
    }

    nceCluster->setAttr(processorFrequencyAttrName, getFPAttr(module.getContext(), 700.0));
}

VPU::ArchKind vpux::VPU::getArch(mlir::Operation* op) {
    auto module = getTopLevelModule(op);

    if (auto attr = module->getAttr(archAttrName)) {
        VPUX_THROW_UNLESS(attr.isa<VPU::ArchKindAttr>(), "Module attribute '{0}' has unsupported value '{1}'",
                          archAttrName, attr);
        return attr.cast<VPU::ArchKindAttr>().getValue();
    }

    return VPU::ArchKind::UNKNOWN;
}

//
// CompilationMode
//

namespace {

constexpr StringLiteral compilationModeAttrName = "VPU.compilationMode";

}  // namespace

void vpux::VPU::setCompilationMode(mlir::ModuleOp module, CompilationMode compilationMode) {
    VPUX_THROW_WHEN(module->hasAttr(compilationModeAttrName),
                    "CompilationMode is already defined, probably you run '--init-compiler' twice");

    module->setAttr(compilationModeAttrName, VPU::CompilationModeAttr::get(module.getContext(), compilationMode));
}

VPU::CompilationMode vpux::VPU::getCompilationMode(mlir::Operation* op) {
    auto module = getTopLevelModule(op);

    if (auto attr = module->getAttr(compilationModeAttrName)) {
        VPUX_THROW_UNLESS(attr.isa<VPU::CompilationModeAttr>(), "Module attribute '{0}' has unsupported value '{1}'",
                          compilationModeAttrName, attr);

        return attr.cast<VPU::CompilationModeAttr>().getValue();
    }

    // Use DefaultHW as a default mode
    return VPU::CompilationMode::DefaultHW;
}

//
// PaddingAttr
//

VPU::PaddingAttr vpux::VPU::getPaddingAttr(mlir::MLIRContext* ctx, int64_t left, int64_t right, int64_t top,
                                           int64_t bottom) {
    return PaddingAttr::get(getIntAttr(ctx, left), getIntAttr(ctx, right), getIntAttr(ctx, top),
                            getIntAttr(ctx, bottom), ctx);
}

VPU::PaddingAttr vpux::VPU::getPaddingAttr(mlir::MLIRContext* ctx, ArrayRef<int64_t> padsBegin,
                                           ArrayRef<int64_t> padsEnd) {
    VPUX_THROW_UNLESS(padsBegin.size() == 2, "Paddings array has unsuppoted size '{0}'", padsBegin.size());
    VPUX_THROW_UNLESS(padsEnd.size() == 2, "Paddings array has unsuppoted size '{0}'", padsEnd.size());
    return getPaddingAttr(ctx, padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0]);
}

VPU::PaddingAttr vpux::VPU::getPaddingAttr(mlir::MLIRContext* ctx, const PadInfo& pad) {
    return getPaddingAttr(ctx, pad.left, pad.right, pad.top, pad.bottom);
}

PadInfo vpux::VPU::toPadInfo(PaddingAttr attr) {
    const auto left = attr.left().getValue().getSExtValue();
    const auto right = attr.right().getValue().getSExtValue();
    const auto top = attr.top().getValue().getSExtValue();
    const auto bottom = attr.bottom().getValue().getSExtValue();
    return PadInfo(left, right, top, bottom);
}

//
// PPETaskAttr
//

VPU::PPETaskAttr vpux::VPU::getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode) {
    return VPU::PPETaskAttr::get(VPU::PPEModeAttr::get(ctx, mode), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, ctx);
}

VPU::PPETaskAttr vpux::VPU::getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode, int64_t clampLow,
                                           int64_t clampHigh, int64_t lreluMult, int64_t lreluShift) {
    return VPU::PPETaskAttr::get(VPU::PPEModeAttr::get(ctx, mode), getIntAttr(ctx, clampLow),
                                 getIntAttr(ctx, clampHigh), getIntAttr(ctx, lreluMult), getIntAttr(ctx, lreluShift),
                                 nullptr, nullptr, nullptr, ctx);
}

VPU::PPETaskAttr vpux::VPU::getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode, int64_t clampLow,
                                           int64_t clampHigh, int64_t lreluMult, int64_t lreluShift,
                                           ArrayRef<int64_t> quantMult, ArrayRef<int64_t> quantShift,
                                           int64_t quantPostShift) {
    return VPU::PPETaskAttr::get(VPU::PPEModeAttr::get(ctx, mode), getIntAttr(ctx, clampLow),
                                 getIntAttr(ctx, clampHigh), getIntAttr(ctx, lreluMult), getIntAttr(ctx, lreluShift),
                                 getIntArrayAttr(ctx, quantMult), getIntArrayAttr(ctx, quantShift),
                                 getIntAttr(ctx, quantPostShift), ctx);
}

VPU::PPEMode vpux::VPU::getPPEMode(VPU::EltwiseType type) {
    switch (type) {
    case VPU::EltwiseType::ADD:
        return vpux::VPU::PPEMode::ADD;
    case VPU::EltwiseType::AND:
        return vpux::VPU::PPEMode::AND;
    case VPU::EltwiseType::MULTIPLY:
        return vpux::VPU::PPEMode::MULT;
    case VPU::EltwiseType::SUBTRACT:
        return vpux::VPU::PPEMode::SUB;
    case VPU::EltwiseType::MIN:
        return vpux::VPU::PPEMode::MINIMUM;
    case VPU::EltwiseType::MAX:
        return vpux::VPU::PPEMode::MAXIMUM;
    default:
        VPUX_THROW("Unsupported EltwiseType '{0}' for PPEMode", type);
    }
}

//
// DistributedTensorAttr
//

mlir::LogicalResult vpux::VPU::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                      DistributedTensorAttr distributedAttr) {
    const auto distributionMode = distributedAttr.mode().getValue();

    if (distributionMode != VPU::DistributionMode::NONE) {
        if (distributedAttr.num_clusters() == nullptr) {
            return printTo(emitError(), "Missing number of clusters.");
        }

        auto numClusters = distributedAttr.num_clusters().getInt();
        if (numClusters <= 0) {
            return printTo(emitError(), "The number of clusters must be greater than 0. Got: {0}", numClusters);
        }
    }

    const auto isTiledMode = [](VPU::DistributionMode mode) {
        return VPU::bitEnumContains(mode, VPU::DistributionMode::SEGMENTED) ||
               VPU::bitEnumContains(mode, VPU::DistributionMode::OVERLAPPED);
    };

    if (!isTiledMode(distributionMode)) {
        return mlir::success();
    }

    if (isTiledMode(distributionMode)) {
        if (distributedAttr.num_tiles() == nullptr || distributedAttr.num_clusters() == nullptr) {
            return printTo(emitError(), "Missing number of tiles and clusters.");
        }

        // Check for validity of tiling scheme
        const auto tilingScheme = parseIntArrayAttr<int64_t>(distributedAttr.num_tiles());
        const auto numClusters = distributedAttr.num_clusters().getInt();

        const auto isValidTile = [](auto dim) {
            return dim > 1;
        };

        if (llvm::count_if(tilingScheme, isValidTile) != 1) {
            return printTo(emitError(), "Currently supporting single axis cluster tiling.");
        }

        const auto axis = std::distance(tilingScheme.begin(), llvm::find_if(tilingScheme, isValidTile));

        if (tilingScheme[axis] != numClusters) {
            return printTo(emitError(), "Incompatibility between tiling scheme '{0}' and number of clusters '{1}'",
                           tilingScheme[axis], numClusters);
        }

        // Limitations on tiling axes
        if (VPU::bitEnumContains(distributionMode, VPU::DistributionMode::OVERLAPPED) &&
            (axis != Dims4D::Act::H.ind())) {
            return printTo(emitError(), "Overlapped cluster tiling is only supported for dimension H");
        }

        if (VPU::bitEnumContains(distributionMode, VPU::DistributionMode::SEGMENTED) &&
            !(axis == Dims4D::Act::H.ind() || axis == Dims4D::Act::C.ind())) {
            return printTo(emitError(), "Segmented cluster tiling is only supported for dimensions H and K");
        }
    }

    return mlir::success();
}

SmallVector<Shape> vpux::VPU::getPerClusterComputeShapes(ShapeRef shapeRef, DistributedTensorAttr distributionAttr) {
    auto shape = to_small_vector(shapeRef.raw());
    const auto distributionMode = distributionAttr.mode().getValue();

    const auto numClusters = distributionAttr.num_clusters().getInt();
    auto tiledComputeShapes = SmallVector<Shape>(numClusters);

    if (bitEnumContains(distributionMode, DistributionMode::SEGMENTED)) {
        const auto isValidTile = [](auto dim) {
            return dim > 1;
        };
        const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.num_tiles());
        const auto axis = std::distance(tilingScheme.begin(), llvm::find_if(tilingScheme, isValidTile));

        // Segmentation logic operates on schema and runtime asumption that
        // a segmented tensor should be split equally across the axis, with
        // the remainder cluster possibly having a smaller tile.

        auto tiledShape = shape;
        tiledShape[axis] = divUp(tiledShape[axis], tilingScheme[axis]);

        auto remainderTileShape = shape;
        remainderTileShape[axis] = divUpRemainder(shape[axis], tilingScheme[axis]);
        VPUX_THROW_UNLESS(remainderTileShape[axis] > 0, "Improper split, '{0}' over '{1}' tiles", shape[axis],
                          tilingScheme[axis]);

        std::fill_n(tiledComputeShapes.begin(), numClusters - 1, Shape(tiledShape));
        tiledComputeShapes[numClusters - 1] = Shape(remainderTileShape);

    } else if (bitEnumContains(distributionMode, DistributionMode::OVERLAPPED)) {
        VPUX_THROW("OVERLAPPED distribution mode is not supported yet");
    } else {
        std::fill_n(tiledComputeShapes.begin(), tiledComputeShapes.size(), Shape(shape));
    }

    return tiledComputeShapes;
}

//
// Generated
//

#include <vpux/compiler/dialect/VPU/generated/attributes/enums.cpp.inc>
#include <vpux/compiler/dialect/VPU/generated/attributes/structs.cpp.inc>
