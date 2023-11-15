//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/attributes.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/attributes.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

#include "vpu/performance.h"

using namespace vpux;

//
// Dialect hooks
//

void VPU::VPUDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/dialect/VPU/attributes.cpp.inc>
            >();
}

//
// Run-time resources
//

namespace {

constexpr StringLiteral derateFactorAttrName = "VPU.derateFactor";
constexpr StringLiteral bandwidthAttrName = "VPU.bandwidth"; /*!< This attribute corresponds to a single JSON field
                      nested at header>resources>memory_bandwidth>number in the deserialized version of the blob.
                      */

}  // namespace

StringLiteral vpux::VPU::getMemoryDerateAttrName() {
    return derateFactorAttrName;
}

StringLiteral vpux::VPU::getMemoryBandwidthAttrName() {
    return bandwidthAttrName;
}

namespace {

constexpr int VPUX30XX_MAX_DPU_GROUPS = 4;
constexpr int VPUX37XX_MAX_DPU_GROUPS = 2;

constexpr int VPUX30XX_MAX_DMA_PORTS = 1;
constexpr int VPUX37XX_MAX_DMA_PORTS = 2;

}  // namespace

uint32_t vpux::VPU::getMaxDPUClusterNum(ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
        return VPUX30XX_MAX_DPU_GROUPS;
    case VPU::ArchKind::VPUX37XX:
        return VPUX37XX_MAX_DPU_GROUPS;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", arch);
    }
}

uint32_t vpux::VPU::getMaxDPUClusterNum(mlir::Operation* op) {
    return VPU::getMaxDPUClusterNum(VPU::getArch(op));
}

uint32_t vpux::VPU::getMaxDMAPorts(ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
        return VPUX30XX_MAX_DMA_PORTS;
    case VPU::ArchKind::VPUX37XX:
        return VPUX37XX_MAX_DMA_PORTS;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", arch);
    }
}

double vpux::VPU::getDMABandwidth(ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
        return 20000.0 / 700;
    case VPU::ArchKind::VPUX37XX:
        return 31200.0f / 1300;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", arch);
    }
}

double vpux::VPU::getNCEThroughput(ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
        return 7168000.0;
    case VPU::ArchKind::VPUX37XX:
        return 8000000.0;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", arch);
    }
}

unsigned int vpux::VPU::getDpuFrequency(vpux::VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
        return VPUNN::get_dpu_fclk(VPUNN::VPUDevice::VPU_2_0);
    case VPU::ArchKind::VPUX37XX:
        return VPUNN::get_dpu_fclk(VPUNN::VPUDevice::VPU_2_7); /*!< The value 1300 corresponds to Highvcc of dpuclk.
                (See VPUX37XX HAS #voltage-and-frequency-targets section).
                 */
    default:
        VPUX_THROW("Unsupported architecture '{0}'", arch);
    }
}

double vpux::VPU::getDmaBandwidthGBps(mlir::ModuleOp module) {
    const ArchKind arch = getArch(module);
    return getDmaBandwidthGBps(arch);
}

double vpux::VPU::getDmaBandwidthGBps(vpux::VPU::ArchKind arch) {
    double BW = 0;
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
        BW = VPUNN::get_dram_bandwidth_MBps(VPUNN::VPUDevice::VPU_2_0);  // 20000 MB/s
        break;
    case VPU::ArchKind::VPUX37XX:
        BW = VPUNN::get_dram_bandwidth_MBps(VPUNN::VPUDevice::VPU_2_7);  // 27000 MB/s
        break;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", arch);
    };

    BW /= 1000;  // convert to GB/s
    return BW;
}

Byte vpux::VPU::getTotalCMXSize(mlir::ModuleOp module) {
    auto cmxRes = IE::getAvailableMemory(module, VPU::MemoryKind::CMX_NN);
    const ArchKind arch = getArch(module);

    // This function is used to determine the best tile size. It tries to put maximum data in CMX.
    // Available CMX memory is decreased by two profilingBufferSize even if profiling is disabled
    // because we want to get exactly same compiled networks with profiling enabled and disabled.
    // Two buffer sizes are required in case when profiling allocates new buffer and old buffer
    // is still not disposed. Second buffer can be treated as an optimisation that prevents spilling.
    const int64_t profilingBufferSize =
            vpux::VPUIP::HW_DMA_PROFILING_MAX_BUFFER_SIZE + vpux::VPUIP::HW_DPU_PROFILING_MAX_BUFFER_SIZE +
            ((arch == ArchKind::VPUX37XX) ? vpux::VPUIP::HW_ACT_SHAVE_PROFILING_MAX_BUFFER_SIZE : 0);

    return cmxRes.size() - Byte(2 * profilingBufferSize);
}

Byte vpux::VPU::getTotalCMXSize(mlir::Operation* op) {
    return getTotalCMXSize(getModuleOp(op));
}

Byte vpux::VPU::getTotalCMXFragmentationAwareSize(mlir::ModuleOp module) {
    auto cmxRes =
            IE::getAvailableMemory(module, mlir::StringAttr::get(module.getContext(), VPU::CMX_NN_FragmentationAware));
    VPUX_THROW_UNLESS(cmxRes != nullptr, "Can't get information about {0} memory", VPU::CMX_NN_FragmentationAware);

    const ArchKind arch = getArch(module);

    // This function is used to determine the best tile size. It tries to put maximum data in CMX.
    // Available CMX memory is decreased by two profilingBufferSize even if profiling is disabled
    // because we want to get exactly same compiled networks with profiling enabled and disabled.
    // Two buffer sizes are required in case when profiling allocates new buffer and old buffer
    // is still not disposed. Second buffer can be treated as an optimisation that prevents spilling.
    const int64_t profilingBufferSize =
            vpux::VPUIP::HW_DMA_PROFILING_MAX_BUFFER_SIZE + vpux::VPUIP::HW_DPU_PROFILING_MAX_BUFFER_SIZE +
            ((arch == ArchKind::VPUX37XX) ? vpux::VPUIP::HW_ACT_SHAVE_PROFILING_MAX_BUFFER_SIZE : 0);

    return cmxRes.size() - Byte(2 * profilingBufferSize);
}

Byte vpux::VPU::getTotalCMXFragmentationAwareSize(mlir::Operation* op) {
    return getTotalCMXFragmentationAwareSize(getModuleOp(op));
}

//
// ArchKind
//

namespace {

constexpr StringLiteral archAttrName = "VPU.arch";

constexpr Byte DDR_HEAP_SIZE = 2200_MB;

constexpr Byte KMB_CMX_WORKSPACE_SIZE = Byte(896_KB);
constexpr Byte KMB_CMX_WORKSPACE_FRAGMENTATION_AWARE_SIZE =
        Byte(static_cast<double>(KMB_CMX_WORKSPACE_SIZE.count()) * FRAGMENTATION_AVOID_RATIO);

constexpr Byte VPUX37XX_CMX_WORKSPACE_SIZE = Byte(1936_KB);
constexpr Byte VPUX37XX_CMX_WORKSPACE_FRAGMENTATION_AWARE_SIZE =
        Byte(static_cast<double>(VPUX37XX_CMX_WORKSPACE_SIZE.count()) * FRAGMENTATION_AVOID_RATIO);

struct Resources {
    Optional<int> numOfDPUGroups = None;
    Optional<int> numOfDMAPorts = None;

    Resources(Optional<int> numOfDPUGroups, Optional<int> numOfDMAPorts)
            : numOfDPUGroups(numOfDPUGroups), numOfDMAPorts(numOfDMAPorts) {
    }
};

struct SetResoursesFuncs {
    using AddExecutorFuncType = FuncRef<IE::ExecutorResourceOp(VPU::ExecutorKind, uint32_t)>;
    using AddSubExecutorFuncType = FuncRef<IE::ExecutorResourceOp(IE::ExecutorResourceOp, VPU::ExecutorKind, uint32_t)>;
    using AddMemoryFuncType = FuncRef<IE::MemoryResourceOp(mlir::StringAttr, Byte)>;
    using AddMemoryWithAttrsFuncType = FuncRef<void(mlir::StringAttr, Byte, double, uint32_t)>;
    using AddInnerMemoryFuncType = FuncRef<IE::MemoryResourceOp(IE::ExecutorResourceOp, mlir::StringAttr, Byte)>;
    using AddInnerMemoryWithAttrsFuncType =
            FuncRef<void(IE::ExecutorResourceOp, mlir::StringAttr, Byte, double, uint32_t)>;

    AddExecutorFuncType addExecutor;
    AddSubExecutorFuncType addSubExecutor;
    AddMemoryFuncType addMemory;
    AddMemoryWithAttrsFuncType addMemoryWithAttrs;
    AddInnerMemoryFuncType addInnerMemory;
    AddInnerMemoryWithAttrsFuncType addInnerMemoryWithAttrs;

    SetResoursesFuncs(AddExecutorFuncType addExecutor, AddSubExecutorFuncType addSubExecutor,
                      AddMemoryFuncType addMemory, AddMemoryWithAttrsFuncType addMemoryWithAttrs,
                      AddInnerMemoryFuncType addInnerMemory, AddInnerMemoryWithAttrsFuncType addInnerMemoryWithAttrs)
            : addExecutor(addExecutor),
              addSubExecutor(addSubExecutor),
              addMemory(addMemory),
              addMemoryWithAttrs(addMemoryWithAttrs),
              addInnerMemory(addInnerMemory),
              addInnerMemoryWithAttrs(addInnerMemoryWithAttrs) {
    }
};

void setArch(mlir::ModuleOp module, VPU::ArchKind kind, const Resources& res, const SetResoursesFuncs& funcs,
             bool allowCustom) {
    VPUX_THROW_WHEN(!allowCustom && module->hasAttr(archAttrName),
                    "Architecture is already defined, probably you run '--init-compiler' twice");

    if (!module->hasAttr(archAttrName)) {
        module->setAttr(archAttrName, VPU::ArchKindAttr::get(module.getContext(), kind));
    }

    auto numOfDPUGroups = res.numOfDPUGroups;
    auto numOfDMAPorts = res.numOfDMAPorts;

    const auto getNumOfDPUGroupsVal = [&](int maxDpuGroups) {
        int numOfDPUGroupsVal = numOfDPUGroups.has_value() ? numOfDPUGroups.value() : maxDpuGroups;
        VPUX_THROW_UNLESS(1 <= numOfDPUGroupsVal && numOfDPUGroupsVal <= maxDpuGroups,
                          "Invalid number of DPU groups: '{0}'", numOfDPUGroupsVal);
        return numOfDPUGroupsVal;
    };

    const auto getNumOfDMAPortsVal = [&](int maxDmaPorts) {
        int numOfDMAPortsVal = numOfDMAPorts.has_value() ? numOfDMAPorts.value() : maxDmaPorts;
        VPUX_THROW_UNLESS(1 <= numOfDMAPortsVal && numOfDMAPortsVal <= maxDmaPorts,
                          "Invalid number of DMA ports: '{0}'", numOfDMAPortsVal);
        return numOfDMAPortsVal;
    };

    IE::ExecutorResourceOp nceCluster;

    const auto ddrStringAttr = mlir::StringAttr::get(module.getContext(), stringifyEnum(VPU::MemoryKind::DDR));
    const auto cmxStringAttr = mlir::StringAttr::get(module.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN));
    const auto cmxFragAwareStringAttr = mlir::StringAttr::get(module.getContext(), VPU::CMX_NN_FragmentationAware);

    switch (kind) {
    case VPU::ArchKind::VPUX30XX: {
        funcs.addMemoryWithAttrs(ddrStringAttr, DDR_HEAP_SIZE, 0.6, 8);

        funcs.addExecutor(VPU::ExecutorKind::DMA_NN, getNumOfDMAPortsVal(VPUX30XX_MAX_DMA_PORTS));
        funcs.addExecutor(VPU::ExecutorKind::SHAVE_UPA, 16);
        nceCluster = funcs.addExecutor(VPU::ExecutorKind::NCE, getNumOfDPUGroupsVal(VPUX30XX_MAX_DPU_GROUPS));
        funcs.addSubExecutor(nceCluster, VPU::ExecutorKind::DPU, 5);
        funcs.addSubExecutor(nceCluster, VPU::ExecutorKind::SHAVE_NN, 5);
        funcs.addInnerMemoryWithAttrs(nceCluster, cmxStringAttr, KMB_CMX_WORKSPACE_SIZE, 1.0, 32);
        funcs.addInnerMemory(nceCluster, cmxFragAwareStringAttr, KMB_CMX_WORKSPACE_FRAGMENTATION_AWARE_SIZE);

        break;
    }
    case VPU::ArchKind::VPUX37XX: {
        funcs.addMemoryWithAttrs(ddrStringAttr, DDR_HEAP_SIZE, 0.6, 8);

        // Have NN_DMA as shared resource across clusters
        funcs.addExecutor(VPU::ExecutorKind::DMA_NN, getNumOfDMAPortsVal(VPUX37XX_MAX_DMA_PORTS));
        nceCluster = funcs.addExecutor(VPU::ExecutorKind::NCE, getNumOfDPUGroupsVal(VPUX37XX_MAX_DPU_GROUPS));
        funcs.addSubExecutor(nceCluster, VPU::ExecutorKind::DPU, 1);
        funcs.addSubExecutor(nceCluster, VPU::ExecutorKind::SHAVE_NN, 1);
        funcs.addSubExecutor(nceCluster, VPU::ExecutorKind::SHAVE_ACT, 2);
        funcs.addInnerMemoryWithAttrs(nceCluster, cmxStringAttr, VPUX37XX_CMX_WORKSPACE_SIZE, 1.0, 32);
        funcs.addInnerMemory(nceCluster, cmxFragAwareStringAttr, VPUX37XX_CMX_WORKSPACE_FRAGMENTATION_AWARE_SIZE);

        break;
    }
    default:
        VPUX_THROW("Unsupported architecture '{0}'", kind);
    }

    VPUX_THROW_WHEN(!allowCustom && nceCluster.hasProcessorFrequency(),
                    "Processor frequencyis already defined, probably you run '--init-compiler' twice");

    if (!nceCluster.hasProcessorFrequency()) {
        nceCluster.setProcessorFrequency(getFPAttr(module.getContext(), getDpuFrequency(kind)));
    }
}

}  // namespace

void vpux::VPU::setArch(mlir::ModuleOp module, ArchKind kind, Optional<int> numOfDPUGroups, Optional<int> numOfDMAPorts,
                        bool allowCustomValues) {
    const auto addExecutor = [&](VPU::ExecutorKind kind, uint32_t count) {
        VPUX_THROW_WHEN(!allowCustomValues && IE::hasExecutor(module, kind),
                        "Available executor kind '{0}' was already added", kind);
        if (IE::hasExecutor(module, kind)) {
            return IE::getAvailableExecutor(module, kind);
        }

        return IE::addAvailableExecutor(module, kind, count);
    };

    const auto addSubExecutor = [&](IE::ExecutorResourceOp resouce, VPU::ExecutorKind kind, uint32_t count) {
        VPUX_THROW_WHEN(!allowCustomValues && resouce.hasSubExecutor(kind),
                        "Available executor kind '{0}' was already added", kind);
        if (resouce.hasSubExecutor(kind)) {
            return resouce.getSubExecutor(kind);
        }

        return resouce.addSubExecutor(kind, count);
    };

    const auto addAvailableMemory = [&](mlir::StringAttr memSpace, Byte size) {
        VPUX_THROW_WHEN(!allowCustomValues && IE::hasAvailableMemory(module, memSpace),
                        "Available memory kind '{0}' was already added", memSpace);
        if (IE::hasAvailableMemory(module, memSpace)) {
            return IE::getAvailableMemory(module, memSpace);
        }

        return IE::addAvailableMemory(module, memSpace, size);
    };

    const auto addMemWithAttrs = [&](mlir::StringAttr memSpace, Byte size, double derateFactor, uint32_t bandwidth) {
        auto mem = addAvailableMemory(memSpace, size);
        if (!mem->hasAttr(derateFactorAttrName)) {
            mem->setAttr(derateFactorAttrName, getFPAttr(module.getContext(), derateFactor));
        }

        if (!mem->hasAttr(bandwidthAttrName)) {
            mem->setAttr(bandwidthAttrName, getIntAttr(module.getContext(), bandwidth));
        }
    };

    const auto addInnerAvailableMemory = [&](IE::ExecutorResourceOp executorOp, mlir::StringAttr memSpace, Byte size) {
        VPUX_THROW_WHEN(!allowCustomValues && executorOp.hasAvailableMemory(memSpace),
                        "Available memory kind '{0}' was already added", memSpace);
        if (executorOp.hasAvailableMemory(memSpace)) {
            return executorOp.getAvailableMemory(memSpace);
        }

        return executorOp.addAvailableMemory(memSpace, size);
    };

    const auto addInnerAvailableMemoryWithAttrs = [&](IE::ExecutorResourceOp executorOp, mlir::StringAttr memSpace,
                                                      Byte size, double derateFactor, uint32_t bandwidth) {
        auto mem = addInnerAvailableMemory(executorOp, memSpace, size);
        if (!mem->hasAttr(derateFactorAttrName)) {
            mem->setAttr(derateFactorAttrName, getFPAttr(module.getContext(), derateFactor));
        }

        if (!mem->hasAttr(bandwidthAttrName)) {
            mem->setAttr(bandwidthAttrName, getIntAttr(module.getContext(), bandwidth));
        }
    };

    ::Resources res(numOfDPUGroups, numOfDMAPorts);
    ::SetResoursesFuncs funcs(addExecutor, addSubExecutor, addAvailableMemory, addMemWithAttrs, addInnerAvailableMemory,
                              addInnerAvailableMemoryWithAttrs);

    return ::setArch(module, kind, res, funcs, allowCustomValues);
}

VPU::ArchKind vpux::VPU::getArch(mlir::Operation* op) {
    auto module = getModuleOp(op);

    if (auto attr = module->getAttr(archAttrName)) {
        VPUX_THROW_UNLESS(attr.isa<VPU::ArchKindAttr>(), "Module attribute '{0}' has unsupported value '{1}'",
                          archAttrName, attr);
        return attr.cast<VPU::ArchKindAttr>().getValue();
    }

    return VPU::ArchKind::UNKNOWN;
}

// To discern between VPUX3XXX and later on architectures
bool vpux::VPU::isArchVPUX3XXX(VPU::ArchKind arch) {
    return (arch == VPU::ArchKind::VPUX37XX) || (arch == VPU::ArchKind::VPUX30XX);
}

//
// CompilationMode
//

namespace {

constexpr StringLiteral compilationModeAttrName = "VPU.compilationMode";

}  // namespace

void vpux::VPU::setCompilationMode(mlir::ModuleOp module, CompilationMode compilationMode) {
    module->setAttr(compilationModeAttrName, VPU::CompilationModeAttr::get(module.getContext(), compilationMode));
}

bool vpux::VPU::hasCompilationMode(mlir::ModuleOp module) {
    return module->hasAttr(compilationModeAttrName);
}

VPU::CompilationMode vpux::VPU::getCompilationMode(mlir::Operation* op) {
    auto module = getModuleOp(op);

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
    return PaddingAttr::get(ctx, getIntAttr(ctx, left), getIntAttr(ctx, right), getIntAttr(ctx, top),
                            getIntAttr(ctx, bottom));
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
    const auto left = attr.getLeft().getValue().getSExtValue();
    const auto right = attr.getRight().getValue().getSExtValue();
    const auto top = attr.getTop().getValue().getSExtValue();
    const auto bottom = attr.getBottom().getValue().getSExtValue();
    return PadInfo(left, right, top, bottom);
}

//
// PPETaskAttr
//

VPU::PPETaskAttr vpux::VPU::getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode) {
    return VPU::PPETaskAttr::get(ctx, VPU::PPEModeAttr::get(ctx, mode), nullptr, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}

VPU::PPETaskAttr vpux::VPU::getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode, int64_t clampLow,
                                           int64_t clampHigh, int64_t lreluMult, int64_t lreluShift) {
    return VPU::PPETaskAttr::get(ctx, VPU::PPEModeAttr::get(ctx, mode), getIntAttr(ctx, clampLow),
                                 getIntAttr(ctx, clampHigh), getIntAttr(ctx, lreluMult), getIntAttr(ctx, lreluShift),
                                 nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}

VPU::PPETaskAttr vpux::VPU::getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode, int64_t clampLow,
                                           int64_t clampHigh, int64_t lreluMult, int64_t lreluShift,
                                           ArrayRef<int64_t> quantMult, ArrayRef<int64_t> quantShift,
                                           int64_t quantPostShift) {
    return VPU::PPETaskAttr::get(ctx, VPU::PPEModeAttr::get(ctx, mode), getIntAttr(ctx, clampLow),
                                 getIntAttr(ctx, clampHigh), getIntAttr(ctx, lreluMult), getIntAttr(ctx, lreluShift),
                                 nullptr, getIntArrayAttr(ctx, quantMult), getIntArrayAttr(ctx, quantShift),
                                 getIntAttr(ctx, quantPostShift), nullptr, nullptr, nullptr);
}

VPU::PPETaskAttr vpux::VPU::getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode, int64_t clampLow,
                                           int64_t clampHigh, int64_t lreluMult, int64_t lreluShift,
                                           ArrayRef<double> quantScale) {
    return VPU::PPETaskAttr::get(ctx, VPU::PPEModeAttr::get(ctx, mode), getIntAttr(ctx, clampLow),
                                 getIntAttr(ctx, clampHigh), getIntAttr(ctx, lreluMult), getIntAttr(ctx, lreluShift),
                                 getFPArrayAttr(ctx, quantScale), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}

VPU::PPETaskAttr vpux::VPU::getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode, int64_t clampLow,
                                           int64_t clampHigh, int64_t lreluMult, int64_t lreluShift,
                                           ArrayRef<int64_t> quantMult, ArrayRef<int64_t> quantShift,
                                           int64_t quantPostShift, ArrayRef<int64_t> in1QuantMult,
                                           ArrayRef<int64_t> in2QuantMult, double fpPReluAlpha) {
    return VPU::PPETaskAttr::get(ctx, VPU::PPEModeAttr::get(ctx, mode), getIntAttr(ctx, clampLow),
                                 getIntAttr(ctx, clampHigh), getIntAttr(ctx, lreluMult), getIntAttr(ctx, lreluShift),
                                 nullptr, getIntArrayAttr(ctx, quantMult), getIntArrayAttr(ctx, quantShift),
                                 getIntAttr(ctx, quantPostShift), getIntArrayAttr(ctx, in1QuantMult),
                                 getIntArrayAttr(ctx, in2QuantMult), getFPAttr(ctx, fpPReluAlpha));
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
                                      DistributedTensorAttr distributedAttr, ArrayRef<int64_t> shape) {
    if (distributedAttr.getComputeShapes() != nullptr && distributedAttr.getComputeOffsets() == nullptr) {
        return printTo(emitError(), "Missing compute_offsets.");
    }

    if (distributedAttr.getComputeShapes() == nullptr && distributedAttr.getComputeOffsets() != nullptr) {
        return printTo(emitError(), "Missing compute_shapes.");
    }

    if (distributedAttr.getMemoryShapes() != nullptr && distributedAttr.getMemoryOffsets() == nullptr) {
        return printTo(emitError(), "Missing memory_offsets.");
    }

    if (distributedAttr.getMemoryShapes() == nullptr && distributedAttr.getMemoryOffsets() != nullptr) {
        return printTo(emitError(), "Missing memory_shapes.");
    }

    const bool hasComputeShapesOffsets =
            distributedAttr.getComputeShapes() != nullptr && distributedAttr.getComputeOffsets() != nullptr;
    const bool hasMemoryShapesOffsets =
            distributedAttr.getMemoryShapes() != nullptr && distributedAttr.getMemoryOffsets() != nullptr;

    if (hasComputeShapesOffsets && !hasMemoryShapesOffsets) {
        return printTo(emitError(), "Missing memory shapes and offsets.");
    }

    if (!hasComputeShapesOffsets && hasMemoryShapesOffsets) {
        return printTo(emitError(), "Missing compute shapes and offsets.");
    }

    const auto distributionMode = distributedAttr.getMode().getValue();

    if (distributionMode != VPU::DistributionMode::NONE) {
        if (distributedAttr.getNumClusters() == nullptr) {
            return printTo(emitError(), "Missing number of clusters.");
        }

        auto numClusters = distributedAttr.getNumClusters().getInt();
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

    if (distributedAttr.getNumTiles() == nullptr || distributedAttr.getNumClusters() == nullptr) {
        return printTo(emitError(), "Missing number of tiles and clusters.");
    }

    // Check for validity of tiling scheme
    const auto tilingScheme = parseIntArrayAttr<int64_t>(distributedAttr.getNumTiles());
    const auto numClusters = distributedAttr.getNumClusters().getInt();

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
    if (VPU::bitEnumContains(distributionMode, VPU::DistributionMode::OVERLAPPED)) {
        if (axis != Dims4D::Act::H.ind() && axis != Dims4D::Act::W.ind()) {
            return printTo(emitError(), "Overlapped cluster tiling is only supported for dimensions H and W");
        }

        if (distributedAttr.getAlignment() != nullptr) {
            const auto alignment = parseIntArrayAttr<int64_t>(distributedAttr.getAlignment());
            if (alignment[axis] != 1) {
                return printTo(
                        emitError(),
                        "Overlapped cluster tiling does not support alignment on the same axis used for tiling.");
            }
        }

        const bool overlappedWithKernelStridesPads = distributedAttr.getKernel() != nullptr &&
                                                     distributedAttr.getPads() != nullptr &&
                                                     distributedAttr.getStrides() != nullptr;

        if (!overlappedWithKernelStridesPads && !hasComputeShapesOffsets) {
            return printTo(emitError(), "Overlapped cluster tiling requires kernel, pads and strides or compute "
                                        "shapes and offsets to be set");
        }

        if (overlappedWithKernelStridesPads && hasComputeShapesOffsets) {
            return printTo(emitError(), "Overlapped cluster tiling must be defined by either kernel/strides/pads "
                                        "or compute shape/offsets, not both");
        }
    }

    if (distributedAttr.getAlignment() != nullptr) {
        const auto alignment = parseIntArrayAttr<int64_t>(distributedAttr.getAlignment());
        if (shape.size() != alignment.size()) {
            return printTo(emitError(), "Incompatibility in sizes between tensor shape '{0}' and alignment '{1}'",
                           shape.size(), alignment.size());
        }
    }

    if (distributedAttr.getNumTiles() != nullptr) {
        const auto numTiles = parseIntArrayAttr<int64_t>(distributedAttr.getNumTiles());
        if (shape.size() != numTiles.size()) {
            return printTo(emitError(), "Incompatibility in sizes between tensor shape '{0}' and tiling scheme '{1}'",
                           shape.size(), numTiles.size());
        }
    }

    if (distributedAttr.getKernel() != nullptr) {
        const auto kernel = parseIntArrayAttr<int64_t>(distributedAttr.getKernel());
        if (kernel.size() != 2) {
            return printTo(emitError(), "Expected kernel size to be 2. Got '{0}'", kernel.size());
        }
        const auto KY = kernel[Dims4D::Kernel::Y.ind()];
        const auto KX = kernel[Dims4D::Kernel::X.ind()];
        if (KY <= 0 || KX <= 0) {
            return printTo(emitError(), "Invalid kernel size: height '{0}', width '{1}'", KY, KX);
        }
    }

    if (distributedAttr.getPads() != nullptr) {
        const auto padTop = distributedAttr.getPads().getTop().getInt();
        const auto padBottom = distributedAttr.getPads().getBottom().getInt();
        const auto padLeft = distributedAttr.getPads().getLeft().getInt();
        const auto padRight = distributedAttr.getPads().getRight().getInt();
        if (padTop < 0 || padBottom < 0 || padLeft < 0 || padRight < 0) {
            return printTo(emitError(), "Invalid pads: top '{0}', bottom '{1}', left '{2}', right '{3}'", padTop,
                           padBottom, padLeft, padRight);
        }
    }

    if (distributedAttr.getStrides() != nullptr) {
        const auto strides = parseIntArrayAttr<int64_t>(distributedAttr.getStrides());
        if (strides.size() != 2) {
            return printTo(emitError(), "Expected strides size to be 2. Got '{0}'", strides.size());
        }
        const auto SY = strides[Dims4D::Strides::Y.ind()];
        const auto SX = strides[Dims4D::Strides::X.ind()];
        if (SY <= 0 || SX <= 0) {
            return printTo(emitError(), "Invalid strides: height '{0}', width '{1}'", SY, SX);
        }
    }

    auto areShapesOffsetsValidForShape = [&shape, &distributedAttr](mlir::ArrayAttr perClusterShapesAttr,
                                                                    mlir::ArrayAttr perClusterOffsetsAttr) -> bool {
        const auto numClusters = static_cast<size_t>(distributedAttr.getNumClusters().getInt());
        if (perClusterShapesAttr.size() != perClusterOffsetsAttr.size() || perClusterShapesAttr.size() != numClusters) {
            return false;
        }

        auto perClusterShapes = vpux::parseIntArrayOfArrayAttr<int64_t>(perClusterShapesAttr);
        auto perClusterOffsets = vpux::parseIntArrayOfArrayAttr<int64_t>(perClusterOffsetsAttr);
        for (size_t cluster = 0; cluster < numClusters; cluster++) {
            if (shape.size() != perClusterShapes[cluster].size() || shape.size() != perClusterOffsets[cluster].size()) {
                return false;
            }

            for (size_t dim = 0; dim < shape.size(); dim++) {
                if (perClusterOffsets[cluster][dim] < 0 || perClusterOffsets[cluster][dim] >= shape[dim]) {
                    return false;
                }

                if (perClusterShapes[cluster][dim] <= 0 || perClusterShapes[cluster][dim] > shape[dim]) {
                    return false;
                }
            }
        }

        return true;
    };

    if (hasComputeShapesOffsets &&
        !areShapesOffsetsValidForShape(distributedAttr.getComputeShapes(), distributedAttr.getComputeOffsets())) {
        return printTo(emitError(), "Invalid compute shapes/offsets for tensor shape = {0}.", distributedAttr);
    }

    if (hasMemoryShapesOffsets &&
        !areShapesOffsetsValidForShape(distributedAttr.getMemoryShapes(), distributedAttr.getMemoryOffsets())) {
        return printTo(emitError(), "Invalid memory shapes/offsets for tensor shape.");
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::areDistributionModesCompatible(DistributionMode sourceMode,
                                                              DistributionMode targetMode) {
    // Consecutive distribution modes for a SOK chain or from HKSwitch to SOK
    if ((sourceMode == (DistributionMode::DUPLICATED | DistributionMode::SEGMENTED) ||
         sourceMode == (DistributionMode::MULTICASTED | DistributionMode::SEGMENTED)) &&
        targetMode == DistributionMode::DUPLICATED) {
        return mlir::success();
    }

    // None const weights for Matmul
    if (sourceMode == DistributionMode::DUPLICATED &&
        targetMode == (DistributionMode::DUPLICATED | DistributionMode::SEGMENTED)) {
        return mlir::success();
    }

    return mlir::failure();
}

mlir::LogicalResult vpux::VPU::areDistributionNumClustersCompatible(mlir::IntegerAttr sourceNumClusters,
                                                                    mlir::IntegerAttr targetNumClusters) {
    return mlir::success(sourceNumClusters.getInt() >= targetNumClusters.getInt());
}

mlir::LogicalResult vpux::VPU::areDistributionElementTypesCompatible(mlir::Type inType, mlir::Type outType) {
    if (inType != outType) {
        // allow different quantization parameters
        if (!inType.isa<mlir::quant::QuantizedType>() || !outType.isa<mlir::quant::QuantizedType>()) {
            return mlir::failure();
        }
        if (vpux::getElemTypeSize(inType) != vpux::getElemTypeSize(outType)) {
            return mlir::failure();
        }
    }

    return mlir::success();
}

int64_t vpux::VPU::getDistributedTilingAxis(ArrayRef<int64_t> tilingScheme) {
    const auto isValidTile = [](auto dim) {
        return dim > 1;
    };

    return std::distance(tilingScheme.begin(), llvm::find_if(tilingScheme, isValidTile));
}

bool vpux::VPU::isDistributedAttrWithExplicitShapesAndOffsets(VPU::DistributedTensorAttr distributionAttr) {
    const bool hasComputeShapesOffsets =
            distributionAttr.getComputeShapes() != nullptr && distributionAttr.getComputeOffsets() != nullptr;
    const bool hasMemoryShapesOffsets =
            distributionAttr.getMemoryShapes() != nullptr && distributionAttr.getMemoryOffsets() != nullptr;

    return hasComputeShapesOffsets && hasMemoryShapesOffsets;
}

//
// Tiling utils
//

// Segmentation logic operates on schema and runtime assumption that a segmented tensor should be split equally
// across the axis, with the remainder cluster possibly having a smaller tile.
SmallVector<Shape> splitSegmentedShape(ArrayRef<int64_t> shape, ArrayRef<int64_t> tilingScheme,
                                       const int64_t numClusters, const int64_t axis,
                                       Optional<ArrayRef<int64_t>> alignment, bool uniformDistributedSegments = false) {
    VPUX_THROW_UNLESS(axis < int64_t(shape.size()),
                      "An invalid split axis {0} specified, the shape tensor is {1} dimensional", axis, shape.size());
    VPUX_THROW_UNLESS(tilingScheme[axis] == numClusters,
                      "The number of tiles on axis {0} must be equal to the number of clusters specified for "
                      "compilation {1} but got {2}",
                      axis, tilingScheme[axis], numClusters);

    SmallVector<Shape> segmentedTiles;
    auto tiledShape = to_small_vector(shape);
    auto remainderTileShape = to_small_vector(shape);
    if (!uniformDistributedSegments) {
        // Split in an equal manner such that first N-1 tiles are equal
        // and the last tile can be less or equal.
        tiledShape[axis] = divUp(tiledShape[axis], tilingScheme[axis]);
        tiledShape = alignShape(tiledShape, alignment, alignValUp<int64_t>);

        // Last tile will have the remainder and it doesn't have to be aligned
        remainderTileShape[axis] = shape[axis] - tiledShape[axis] * (tilingScheme[axis] - 1);
        VPUX_THROW_UNLESS(remainderTileShape[axis] > 0, "Improper split, '{0}' over '{1}' tiles", shape[axis],
                          tilingScheme[axis]);
        segmentedTiles.insert(segmentedTiles.end(), numClusters - 1, Shape(tiledShape));
        segmentedTiles.push_back(Shape(remainderTileShape));
    } else {
        // Split into a more balanced approach such that there's
        // a minimum different between the segments sizes.
        // For example a height of 6 is split across 4 tile as [2, 2, 1, 1].

        // Compute baseline tile, specifically also align it
        tiledShape[axis] = tiledShape[axis] / tilingScheme[axis];
        tiledShape = alignShape(tiledShape, alignment, alignValDown<int64_t>);
        VPUX_THROW_UNLESS(tiledShape[axis] > 0, "Improper split, '{0}' over '{1}' tiles", shape[axis],
                          tilingScheme[axis]);
        // Remainder of data is distributed across first few tiles
        remainderTileShape = tiledShape;
        auto remainderCount = shape[axis] - tiledShape[axis] * tilingScheme[axis];
        auto axisAlignment = 1;
        if (alignment.has_value()) {
            axisAlignment = alignment.value()[axis];
        }
        VPUX_THROW_WHEN(remainderCount % axisAlignment, "Tiling remainder '{0}' is not aligned to '{1}'.",
                        remainderCount, axisAlignment);
        auto remainderElements = remainderCount / axisAlignment;
        remainderTileShape[axis] = tiledShape[axis] + axisAlignment;

        segmentedTiles.insert(segmentedTiles.end(), remainderElements, Shape(remainderTileShape));
        segmentedTiles.insert(segmentedTiles.end(), numClusters - remainderElements, Shape(tiledShape));
    }
    return segmentedTiles;
}

SmallVector<DimRange> getOverlappedInputTileDimRanges(ArrayRef<int64_t> shape, ArrayRef<int64_t> tilingScheme,
                                                      mlir::ArrayAttr kernelAttr, mlir::ArrayAttr stridesAttr,
                                                      VPU::PaddingAttr padAttr, const int64_t axis,
                                                      const int64_t numClusters,
                                                      const bool uniformDistributedSegments) {
    const auto axisDim = Dim(axis);
    VPUX_THROW_UNLESS(axisDim == Dims4D::Act::W || axisDim == Dims4D::Act::H,
                      "Input overlapping supported only for W or H axes");

    const auto N = shape[Dims4D::Act::N.ind()];
    const auto C = shape[Dims4D::Act::C.ind()];
    const auto Y = shape[Dims4D::Act::H.ind()];
    const auto X = shape[Dims4D::Act::W.ind()];

    VPUX_THROW_WHEN(padAttr == nullptr, "Pads shouldn't be nullptr");
    const auto outputHW = vpux::spatialOutputForInputWindowSize({Y, X}, kernelAttr, stridesAttr, toPadInfo(padAttr));
    const SmallVector<int64_t> outputShape{N, C, outputHW.first, outputHW.second};

    // Alignment should only be considered for final input shape,
    // not the intermediary output shape
    const auto outputTiles =
            splitSegmentedShape(outputShape, tilingScheme, numClusters, axis, None, uniformDistributedSegments);

    int64_t offset = 0;
    VPUX_THROW_WHEN(kernelAttr == nullptr, "Kernel shouldn't be nullptr");
    const auto kernel = parseIntArrayAttr<int64_t>(kernelAttr);
    const auto KY = kernel[Dims4D::Kernel::Y.ind()];
    const auto KX = kernel[Dims4D::Kernel::X.ind()];

    VPUX_THROW_WHEN(stridesAttr == nullptr, "Strides shouldn't be nullptr");
    const auto strides = parseIntArrayAttr<int64_t>(stridesAttr);
    const auto SY = strides[Dims4D::Strides::Y.ind()];
    const auto SX = strides[Dims4D::Strides::X.ind()];

    const auto padTop = padAttr.getTop().getInt();
    const auto padBottom = padAttr.getBottom().getInt();
    const auto padLeft = padAttr.getLeft().getInt();
    const auto padRight = padAttr.getRight().getInt();
    SmallVector<DimRange> inputTileDimRanges;
    for (const auto& outputTile : outputTiles) {
        const auto dimSize = outputTile[Dim(axis)];
        const DimRange tileSize(offset, offset + dimSize);
        offset += dimSize;

        DimRange inputTile(0, 0);
        if (axis == Dims4D::Act::H.ind()) {
            std::tie(inputTile, std::ignore, std::ignore) =
                    vpux::inputForOutputDim(tileSize, KY, SY, {0, Y}, padTop, padBottom);
        } else if (axis == Dims4D::Act::W.ind()) {
            std::tie(inputTile, std::ignore, std::ignore) =
                    vpux::inputForOutputDim(tileSize, KX, SX, {0, X}, padLeft, padRight);
        } else {
            VPUX_THROW("Unsupported axis '{0}'", axis);
        }
        inputTileDimRanges.push_back(inputTile);
    }
    return inputTileDimRanges;
}

SmallVector<Shape> vpux::VPU::getPerClusterComputeShapes(ShapeRef shapeRef, DistributedTensorAttr distributionAttr) {
    auto shape = to_small_vector(shapeRef.raw());
    const auto distributionMode = distributionAttr.getMode().getValue();

    const auto numClusters = distributionAttr.getNumClusters().getInt();
    auto tiledComputeShapes = SmallVector<Shape>(numClusters);

    Optional<ArrayRef<int64_t>> optionalAlignment = None;
    auto alignment = SmallVector<int64_t>(numClusters);
    if (distributionAttr.getAlignment() != nullptr) {
        alignment = parseIntArrayAttr<int64_t>(distributionAttr.getAlignment());
        optionalAlignment = Optional<ArrayRef<int64_t>>(alignment);
    }

    auto getComputeSplitIntoSegments = [&]() -> SmallVector<Shape> {
        const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
        const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
        VPUX_THROW_UNLESS(axis < int64_t(tilingScheme.size()), "Segmented tiling scheme requires at least 1 dimension "
                                                               "to be segmented but the tiling schema is [1, 1, 1, 1]");
        return splitSegmentedShape(shape, tilingScheme, numClusters, axis, optionalAlignment,
                                   distributionAttr.getUniformDistributedSegments() != nullptr);
    };

    if (VPU::bitEnumContains(distributionMode, VPU::DistributionMode::SEGMENTED)) {
        return getComputeSplitIntoSegments();
    }

    if (VPU::bitEnumContains(distributionMode, VPU::DistributionMode::OVERLAPPED)) {
        if (distributionAttr.getEqualMemoryAndComputeView() != nullptr) {
            return getPerClusterMemoryShapes(shapeRef, distributionAttr);
        }

        return getComputeSplitIntoSegments();
    }

    if (distributionMode == VPU::DistributionMode::DUPLICATED ||
        distributionMode == VPU::DistributionMode::MULTICASTED) {
        std::fill_n(tiledComputeShapes.begin(), tiledComputeShapes.size(),
                    Shape(alignShape(shape, optionalAlignment, alignValUp<int64_t>)));
        return tiledComputeShapes;
    }

    VPUX_THROW("Cannot get per cluster memory shapes. Unsupported distribution: {0}", distributionAttr);
}

SmallVector<Shape> vpux::VPU::getPerClusterComputeShapeOffsets(ShapeRef shapeRef,
                                                               DistributedTensorAttr distributionAttr) {
    const auto shape = to_small_vector(shapeRef.raw());
    const auto distributionMode = distributionAttr.getMode().getValue();

    const auto numClusters = distributionAttr.getNumClusters().getInt();
    auto tiledComputeShapeOffsets = SmallVector<Shape>(numClusters, Shape(shapeRef.size(), 0));

    auto getOffsetsForSegments = [&](SmallVector<Shape>& perClusterOffsets) -> SmallVector<Shape> {
        const auto tiledComputeShapes = getPerClusterComputeShapes(shapeRef, distributionAttr);
        const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
        const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);

        int64_t offset = 0;
        for (int64_t idx = 0; idx < numClusters; idx++) {
            perClusterOffsets[idx][Dim(axis)] = offset;
            offset += tiledComputeShapes[idx][Dim(axis)];
        }

        return perClusterOffsets;
    };

    if (VPU::bitEnumContains(distributionMode, VPU::DistributionMode::SEGMENTED)) {
        return getOffsetsForSegments(tiledComputeShapeOffsets);
    }

    if (VPU::bitEnumContains(distributionMode, VPU::DistributionMode::OVERLAPPED)) {
        if (distributionAttr.getEqualMemoryAndComputeView() != nullptr) {
            return getPerClusterMemoryShapeOffsets(shapeRef, distributionAttr);
        }

        return getOffsetsForSegments(tiledComputeShapeOffsets);
    }

    if (distributionMode == VPU::DistributionMode::DUPLICATED ||
        distributionMode == VPU::DistributionMode::MULTICASTED) {
        return tiledComputeShapeOffsets;
    }

    VPUX_THROW("Cannot get per cluster memory shapes. Unsupported distribution: {0}", distributionAttr);
}

SmallVector<Shape> vpux::VPU::getPerClusterMemoryShapes(ShapeRef shapeRef, DistributedTensorAttr distributionAttr) {
    auto shape = to_small_vector(shapeRef.raw());
    const auto distributionMode = distributionAttr.getMode().getValue();

    const auto numClusters = distributionAttr.getNumClusters().getInt();
    auto tiledMemoryShapes = SmallVector<Shape>(numClusters);

    Optional<ArrayRef<int64_t>> optionalAlignment = None;
    auto alignment = SmallVector<int64_t>(numClusters);
    if (distributionAttr.getAlignment() != nullptr) {
        alignment = parseIntArrayAttr<int64_t>(distributionAttr.getAlignment());
        optionalAlignment = Optional<ArrayRef<int64_t>>(alignment);
    }

    if (VPU::bitEnumContains(distributionMode, VPU::DistributionMode::DUPLICATED) ||
        VPU::bitEnumContains(distributionMode, VPU::DistributionMode::MULTICASTED)) {
        std::fill_n(tiledMemoryShapes.begin(), tiledMemoryShapes.size(),
                    Shape(alignShape(shape, optionalAlignment, alignValUp<int64_t>)));

        return tiledMemoryShapes;
    }

    if (distributionMode == VPU::DistributionMode::SEGMENTED) {
        const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
        const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
        VPUX_THROW_UNLESS(axis < int64_t(tilingScheme.size()), "Segmented tiling scheme requires at least 1 dimension "
                                                               "to be segmented but the tiling schema is [1, 1, 1, 1]");
        return splitSegmentedShape(shape, tilingScheme, numClusters, axis, optionalAlignment,
                                   distributionAttr.getUniformDistributedSegments() != nullptr);
    }

    if (distributionMode == VPU::DistributionMode::OVERLAPPED) {
        const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
        const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
        const auto inputTileDimRanges = getOverlappedInputTileDimRanges(
                shape, tilingScheme, distributionAttr.getKernel(), distributionAttr.getStrides(),
                distributionAttr.getPads(), axis, numClusters,
                distributionAttr.getUniformDistributedSegments() != nullptr);

        for (auto p : inputTileDimRanges | indexed) {
            const auto inputTile = p.value();
            const auto cluster = p.index();
            shape[axis] = inputTile.end - inputTile.begin;
            tiledMemoryShapes[cluster] = Shape(alignShape(shape, optionalAlignment, alignValUp<int64_t>));
        }

        return tiledMemoryShapes;
    }

    VPUX_THROW("Cannot get per cluster memory shapes. Unsupported distribution: {0}", distributionAttr);
}

SmallVector<Shape> vpux::VPU::getPerClusterMemoryShapeOffsets(ShapeRef shapeRef,
                                                              DistributedTensorAttr distributionAttr) {
    const auto shape = to_small_vector(shapeRef.raw());
    const auto distributionMode = distributionAttr.getMode().getValue();

    const auto numClusters = distributionAttr.getNumClusters().getInt();

    auto tiledMemoryOffsets = SmallVector<Shape>(numClusters, Shape(shapeRef.size(), 0));

    // For distribution mode containing either DUPLICATED or MULTICASTED, the starting offset
    // will be 0 across all dimensions since the entire output tensor can be found in each cluster
    if (VPU::bitEnumContains(distributionMode, VPU::DistributionMode::DUPLICATED) ||
        VPU::bitEnumContains(distributionMode, VPU::DistributionMode::MULTICASTED)) {
        return tiledMemoryOffsets;
    }

    if (distributionMode == VPU::DistributionMode::SEGMENTED) {
        const auto tiledComputeShapes = getPerClusterMemoryShapes(shapeRef, distributionAttr);
        const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
        const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);

        int64_t offset = 0;
        for (int64_t idx = 0; idx < numClusters; idx++) {
            tiledMemoryOffsets[idx][Dim(axis)] = offset;
            offset += tiledComputeShapes[idx][Dim(axis)];
        }

        return tiledMemoryOffsets;
    }

    if (distributionMode == VPU::DistributionMode::OVERLAPPED) {
        const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
        const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);

        const auto inputTileDimRanges = getOverlappedInputTileDimRanges(
                shape, tilingScheme, distributionAttr.getKernel(), distributionAttr.getStrides(),
                distributionAttr.getPads(), axis, numClusters,
                distributionAttr.getUniformDistributedSegments() != nullptr);

        for (auto p : inputTileDimRanges | indexed) {
            const auto inputTile = p.value();
            const auto cluster = p.index();
            tiledMemoryOffsets[cluster][Dim(axis)] = inputTile.begin;
        }

        return tiledMemoryOffsets;
    }

    VPUX_THROW("Cannot get per cluster memory shapes. Unsupported distribution: {0}", distributionAttr);
}

SmallVector<PadInfo> vpux::VPU::getPerClusterPadding(DistributedTensorAttr distributionAttr, PadInfo kernelPadding) {
    const auto mode = distributionAttr.getMode().getValue();
    VPUX_THROW_UNLESS(mode == VPU::DistributionMode::OVERLAPPED,
                      "Currently getting per cluster padding is supported only for OVERLAPPED, mode - {0}",
                      VPU::stringifyDistributionMode(mode));

    const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
    const auto axisDim = Dim(vpux::VPU::getDistributedTilingAxis(tilingScheme));

    VPUX_THROW_UNLESS(axisDim == Dims4D::Act::H || axisDim == Dims4D::Act::W,
                      "Currently getting per cluster padding is supported only for tiling axis H or W, axis - {0}",
                      axisDim);
    VPUX_THROW_UNLESS(distributionAttr.getComputeShapes() == nullptr && distributionAttr.getComputeOffsets() == nullptr,
                      "Currently getting per cluster padding is not supported for distribution with shape/offset "
                      "set, mode - {0}",
                      VPU::stringifyDistributionMode(mode));

    SmallVector<PadInfo> perClusterPadInfo;
    const auto top = kernelPadding.top;
    const auto bottom = kernelPadding.bottom;
    const auto left = kernelPadding.left;
    const auto right = kernelPadding.right;

    const auto firstClusterPadInfo =
            (axisDim == Dims4D::Act::H) ? PadInfo(left, right, top, 0) : PadInfo(left, 0, top, bottom);
    const auto lastClusterPadInfo =
            (axisDim == Dims4D::Act::H) ? PadInfo(left, right, 0, bottom) : PadInfo(0, right, top, bottom);

    perClusterPadInfo.push_back(firstClusterPadInfo);
    for (auto cluster = 1; cluster < distributionAttr.getNumClusters().getInt() - 1; cluster++) {
        const auto padInfo = (axisDim == Dims4D::Act::H) ? PadInfo(left, right, 0, 0) : PadInfo(0, 0, top, bottom);
        perClusterPadInfo.push_back(padInfo);
    }
    perClusterPadInfo.push_back(lastClusterPadInfo);

    return perClusterPadInfo;
}

SmallVector<StridedShape> vpux::VPU::getPerClusterMemoryStridedShapes(ShapeRef shape, StridesRef strides,
                                                                      DimsOrder dimsOrder, DistributionModeAttr mode,
                                                                      ArrayRef<Shape> memoryShapes) {
    const auto distributionMode = mode.getValue();

    SmallVector<StridedShape> stridedShapes;
    if (VPU::bitEnumContains(distributionMode, VPU::DistributionMode::DUPLICATED)) {
        for (const auto& memoryShape : memoryShapes) {
            stridedShapes.emplace_back(memoryShape, strides);
        }
        return stridedShapes;
    }

    if (VPU::bitEnumContains(distributionMode, VPU::DistributionMode::SEGMENTED) ||
        VPU::bitEnumContains(distributionMode, VPU::DistributionMode::OVERLAPPED)) {
        const auto adaptedStrides = adaptStrides(shape, strides, memoryShapes, dimsOrder);
        for (const auto& p : zip(memoryShapes, adaptedStrides)) {
            stridedShapes.emplace_back(std::get<0>(p), std::get<1>(p));
        }
        return stridedShapes;
    }

    VPUX_THROW("Unsupported mode '{0}'", VPU::stringifyEnum(distributionMode));
}

SmallVector<Shape> vpux::VPU::arrayAttrToVecOfShapes(mlir::ArrayAttr arr) {
    SmallVector<Shape> shapesVec;
    const auto parsedVec = parseIntArrayOfArrayAttr<int64_t>(arr);
    for (auto ind : irange(parsedVec.size())) {
        shapesVec.push_back(Shape(parsedVec[ind]));
    }

    return shapesVec;
}

bool vpux::VPU::isSegmentedOverH(VPU::DistributedTensorAttr distAttr) {
    if (distAttr.getMode().getValue() != VPU::DistributionMode::SEGMENTED) {
        return false;
    }
    const auto numTiles = parseIntArrayAttr<int64_t>(distAttr.getNumTiles());
    if (numTiles.size() != 4 || numTiles[Dims4D::Act::N.ind()] > 1 || numTiles[Dims4D::Act::C.ind()] > 1 ||
        numTiles[Dims4D::Act::W.ind()] > 1) {
        return false;
    }
    return true;
}

bool vpux::VPU::isSegmentedOverC(VPU::DistributedTensorAttr distAttr) {
    if (distAttr.getMode().getValue() != VPU::DistributionMode::SEGMENTED) {
        return false;
    }
    const auto numTiles = parseIntArrayAttr<int64_t>(distAttr.getNumTiles());
    if (numTiles.size() != 4 || numTiles[Dims4D::Act::N.ind()] > 1 || numTiles[Dims4D::Act::H.ind()] > 1 ||
        numTiles[Dims4D::Act::W.ind()] > 1) {
        return false;
    }
    return true;
}

bool vpux::VPU::isSegmentedOverN(VPU::DistributedTensorAttr distAttr) {
    if (distAttr.getMode().getValue() != VPU::DistributionMode::SEGMENTED) {
        return false;
    }
    const auto numTiles = parseIntArrayAttr<int64_t>(distAttr.getNumTiles());
    if (numTiles.size() != 4 || numTiles[Dims4D::Act::C.ind()] > 1 || numTiles[Dims4D::Act::H.ind()] > 1 ||
        numTiles[Dims4D::Act::W.ind()] > 1) {
        return false;
    }
    return true;
}

bool vpux::VPU::isOverlappedOverH(VPU::DistributedTensorAttr distAttr) {
    if (distAttr.getMode().getValue() != VPU::DistributionMode::OVERLAPPED) {
        return false;
    }
    const auto numTiles = parseIntArrayAttr<int64_t>(distAttr.getNumTiles());
    if (numTiles.size() != 4 || numTiles[Dims4D::Act::N.ind()] > 1 || numTiles[Dims4D::Act::C.ind()] > 1 ||
        numTiles[Dims4D::Act::W.ind()] > 1) {
        return false;
    }
    return true;
}

//
// CompressionSchemeAttr
//

int64_t VPU::CompressionSchemeAttr::getTotalNumElems() const {
    if (getNumElems().empty()) {
        return 0;
    }
    auto numElems = getNumElems().getValues<int64_t>();
    return std::accumulate(numElems.begin(), numElems.end(), static_cast<int64_t>(0));
}

int64_t VPU::CompressionSchemeAttr::getNumElemsInRange(int64_t startIdx, int64_t size) const {
    const auto numElems = getNumElems().getValues<int64_t>();
    const auto startIt = numElems.begin() + startIdx;
    const auto endIt = startIt + size;
    return std::accumulate(startIt, endIt, static_cast<int64_t>(0));
}

Byte VPU::CompressionSchemeAttr::getAllocSize(mlir::Type elemType) const {
    const auto elemByteSize = getElemTypeSize(elemType).to<Byte>().count();
    const int64_t alignment = (getAlignment() != nullptr) ? getAlignment().getInt() : 1;
    const auto numElems = getNumElems().getValues<int64_t>();
    int64_t totalAllocSize = 0;
    for (auto num : numElems) {
        totalAllocSize += alignValUp<int64_t>(num * elemByteSize, alignment);
    }
    return Byte(totalAllocSize);
}

VPU::CompressionSchemeAttr VPU::getCompressionSchemeAttr(mlir::Type type) {
    if (auto sparseType = type.dyn_cast_or_null<VPU::SparseTensorType>()) {
        return sparseType.getCompressionScheme();
    }
    return nullptr;
}

mlir::Type VPU::setCompressionSchemeAttr(mlir::Type type, VPU::CompressionSchemeAttr compressionSchemeAttr) {
    if (auto sparseType = type.dyn_cast_or_null<VPU::SparseTensorType>()) {
        return VPU::SparseTensorType::get(sparseType.getData(), sparseType.getSparsityMap(),
                                          sparseType.getStorageElementTable(), sparseType.getIsWeights(),
                                          compressionSchemeAttr);
    }
    return type;
}

VPU::CompressionSchemeAttr VPU::tileCompressionScheme(VPU::CompressionSchemeAttr compressionScheme,
                                                      ShapeRef tileOffsets, ShapeRef tileShape) {
    if (compressionScheme == nullptr) {
        return nullptr;
    }
    VPUX_THROW_UNLESS(compressionScheme.getAxis() != nullptr,
                      "Cannot tile compression scheme that is not over an axis");
    const size_t axis = compressionScheme.getAxis().getInt();
    VPUX_THROW_UNLESS(axis < tileOffsets.size() && axis < tileShape.size(),
                      "Axis {0} outside the range of tile dimensions: offsets size {1}, shape size {2}", axis,
                      tileOffsets.size(), tileShape.size());

    const auto numElems = compressionScheme.getNumElems().getValues<int64_t>();
    const auto dimOffset = tileOffsets[Dim(axis)];
    const auto dimShape = tileShape[Dim(axis)];

    const auto startIt = numElems.begin() + dimOffset;
    const auto endIt = startIt + dimShape;
    const auto tileNumElems = SmallVector<int64_t>(startIt, endIt);

    auto ctx = compressionScheme.getContext();
    const auto tileNumElemsType =
            mlir::RankedTensorType::get({static_cast<int64_t>(tileNumElems.size())}, getInt64Type(ctx));
    const auto tileNumElemsAttr = mlir::DenseElementsAttr::get(tileNumElemsType, makeArrayRef(tileNumElems));
    return VPU::CompressionSchemeAttr::get(ctx, compressionScheme.getAxis(), tileNumElemsAttr,
                                           compressionScheme.getAlignment());
}

mlir::LogicalResult VPU::SEInterpolateAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                                   vpux::VPU::NCEInterpolateModeAttr modeAttr,
                                                   vpux::IE::InterpolateCoordModeAttr coordTransformModeAttr,
                                                   mlir::ArrayAttr scalesAttr,
                                                   vpux::IE::InterpolateNearestModeAttr nearestModeAttr,
                                                   mlir::ArrayAttr /*offsetsAttr*/, mlir::ArrayAttr /*sizesAttr*/,
                                                   mlir::ArrayAttr /*initialInputShapeAttr*/,
                                                   mlir::ArrayAttr /*initialOutputShapeAttr*/) {
    if (modeAttr == nullptr) {
        return printTo(emitError(), "Got NULL 'NCEInterpolateMode' in 'SEInterpolateAttr'");
    }
    if (nearestModeAttr == nullptr && modeAttr.getValue() == VPU::NCEInterpolateMode::NEAREST) {
        return printTo(emitError(),
                       "Got NULL 'NCEInterpolateNearestMode' in 'SEInterpolateAttr' with interpolate mode NEAREST");
    }
    if (coordTransformModeAttr == nullptr) {
        return printTo(emitError(), "Got NULL 'NCEInterpolateCoordMode' in 'SEInterpolateAttr'");
    }
    if (scalesAttr == nullptr) {
        return printTo(emitError(), "Got NULL scales in 'SEInterpolateAttr'");
    }
    const auto scales = parseFPArrayAttr<double>(scalesAttr);
    const auto realScale = std::find_if(scales.begin(), scales.end(), [&](auto value) {
        return std::floor(value) != value;
    });
    if (realScale != scales.end()) {
        return printTo(emitError(), "'SEInterpolateAttr' supports only integer scale values but got {0}", realScale);
    }

    return mlir::success();
}

// Calculate output shape without tiling info
static Shape inferFullOutputShape(ShapeRef inputShape, llvm::SmallVector<double> scales, VPU::NCEInterpolateMode mode) {
    std::function<int64_t(int64_t, double)> modifyDim;
    if (mode == VPU::NCEInterpolateMode::NEAREST) {
        modifyDim = [](auto dim, auto scale) {
            return static_cast<int64_t>(std::floor(dim * scale));
        };
    } else if (mode == VPU::NCEInterpolateMode::BILINEAR) {
        modifyDim = [](auto dim, auto scale) {
            return static_cast<int64_t>(std::floor(dim * scale + (scale - 1)));
        };
    } else {
        VPUX_THROW("SEInterpolateAttr: Unsupported NCEInterpolateMode {0}", mode);
    }
    Shape outputShape(inputShape.toValues());
    std::transform(inputShape.begin(), inputShape.end(), scales.begin(), outputShape.begin(), modifyDim);
    return outputShape;
}

/*
 *  Infer final output shape with respect to tiling:
 *  - If sizes attr presents then its values are returned
 *  - If not then shape is calculated with scales and mode
 */
Shape VPU::SEInterpolateAttr::inferOutputShape(ShapeRef inputShape) const {
    if (auto sizes = getSizes()) {
        return Shape(parseIntArrayAttr<int64_t>(sizes));
    }
    const auto scales = parseFPArrayAttr<double>(getScale());
    const auto mode = getMode().getValue();

    return inferFullOutputShape(inputShape, scales, mode);
}

Shape VPU::SEInterpolateAttr::backInferShape(ShapeRef outputShape) const {
    auto scalesAttr = getScale();
    Shape inputShape(outputShape.toValues());

    std::function<int64_t(int64_t, double)> modifyDim;
    if (getMode().getValue() == VPU::NCEInterpolateMode::NEAREST) {
        modifyDim = [](auto dimSize, auto scale) {
            return static_cast<int64_t>(std::ceil(dimSize / scale));
        };
    } else if (getMode().getValue() == VPU::NCEInterpolateMode::BILINEAR) {
        modifyDim = [](auto dimSize, auto scale) {
            return static_cast<int64_t>(std::ceil((dimSize - (scale - 1)) / scale));
        };
    } else {
        VPUX_THROW("SEInterpolateAttr: Unsupported NCEInterpolateMode {0}", getMode().getValue());
    }

    auto scales = parseFPArrayAttr<double>(scalesAttr);

    std::transform(outputShape.begin(), outputShape.end(), scales.begin(), inputShape.begin(), modifyDim);

    return inputShape;
}

Shape VPU::SEInterpolateAttr::backInferCoord(ShapeRef outputTileOffset, ShapeRef inputShape) const {
    const auto scalesAttr = getScale();
    const auto scales = parseFPArrayAttr<double>(scalesAttr);
    const auto mode = getMode().getValue();

    // Get coordinate transformation function
    auto coordMode = getCoordinateTransformationMode();
    VPUX_THROW_WHEN(coordMode == nullptr, "Missing coordinate transformation mode");
    std::function<double(int64_t, double, int64_t, int64_t)> coordTransform;
    if (mode == VPU::NCEInterpolateMode::NEAREST) {
        switch (coordMode.getValue()) {
        case IE::InterpolateCoordMode::HALF_PIXEL:
            coordTransform = [&](auto coord, auto scale, auto /*inputSize*/, auto /*outputSize*/) {
                return (static_cast<double>(coord) + 0.5) / scale - 0.5;
            };
            break;
        case IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL:
            coordTransform = [&](auto coord, auto scale, auto /*inputSize*/, auto outputSize) {
                return (outputSize > 1) ? (static_cast<double>(coord) + 0.5) / scale - 0.5 : 0.0;
            };
            break;
        case IE::InterpolateCoordMode::ASYMMETRIC:
            coordTransform = [&](auto coord, auto scale, auto /*inputSize*/, auto /*outputSize*/) {
                return coord / scale;
            };
            break;
        case IE::InterpolateCoordMode::TF_HALF_PIXEL_FOR_NN:
            coordTransform = [&](auto coord, auto scale, auto /*inputSize*/, auto /*outputSize*/) {
                return (static_cast<double>(coord) + 0.5) / scale;
            };
            break;
        case IE::InterpolateCoordMode::ALIGN_CORNERS:
            coordTransform = [&](auto coord, auto /*scale*/, auto inputSize, auto outputSize) {
                return (outputSize == 1) ? 0.0 : static_cast<double>(coord) * (inputSize - 1) / (outputSize - 1);
            };
            break;
        }
    } else if (mode == VPU::NCEInterpolateMode::BILINEAR) {
        coordTransform = [&](auto coord, auto scale, auto /*inputSize*/, auto outputSize) {
            // starting from (outputSize - scale) the last colomn/row is just replicated
            if (coord > outputSize - scale) {
                return (outputSize - scale) / scale;
            } else {
                return coord / scale;
            }
        };
    } else {
        VPUX_THROW("SEInterpolateAttr: Unsupported NCEInterpolateMode {0}.", mode);
    }

    // The way, how nearest interpolate gets nearest pixel, depends on mode.
    // Default implementation is for bilinear
    std::function<int64_t(double, double)> nearestDim = [](auto coord, auto scale) {
        VPUX_UNUSED(scale);
        return static_cast<int64_t>(std::floor(coord));
    };

    if (auto nearestModeAttr = getNearestMode()) {
        switch (nearestModeAttr.getValue()) {
        case IE::InterpolateNearestMode::ROUND_PREFER_FLOOR:
            nearestDim = [](auto dim, auto scale) {
                VPUX_UNUSED(scale);
                if (isDoubleEqual(dim, std::floor(dim) + 0.5)) {
                    return static_cast<int64_t>(std::floor(dim));
                }
                return static_cast<int64_t>(std::round(dim));
            };
            break;
        case IE::InterpolateNearestMode::ROUND_PREFER_CEIL:
            nearestDim = [](auto dim, auto scale) {
                VPUX_UNUSED(scale);
                return static_cast<int64_t>(std::round(dim));
            };
            break;
        case IE::InterpolateNearestMode::FLOOR:
            nearestDim = [](auto dim, auto scale) {
                VPUX_UNUSED(scale);
                return static_cast<int64_t>(std::floor(dim));
            };
            break;
        case IE::InterpolateNearestMode::CEIL:
            nearestDim = [](auto dim, auto scale) {
                VPUX_UNUSED(scale);
                return static_cast<int64_t>(std::ceil(dim));
            };
            break;
        case IE::InterpolateNearestMode::SIMPLE:
            nearestDim = [](auto dim, auto scale) {
                if (scale < 1.) {
                    return static_cast<int64_t>(std::ceil(dim));
                }
                return static_cast<int64_t>(dim);
            };
            break;
        default:
            VPUX_THROW("SEInterpolateAttr: Unsupported InterpolateNearestMode. {0}", nearestModeAttr.getValue());
        }
    }

    const auto outputShape = inferFullOutputShape(inputShape, scales, mode);

    const auto initialInputShape = (getInitialInputShape() != nullptr)
                                           ? Shape(parseIntArrayAttr<int64_t>(getInitialInputShape()))
                                           : inputShape;
    const auto initialOutputShape = (getInitialOutputShape() != nullptr)
                                            ? Shape(parseIntArrayAttr<int64_t>(getInitialOutputShape()))
                                            : outputShape;

    auto inputTileOffset = Shape(outputTileOffset.toValues());
    // Get pixel in output which corresponds to this input pixel
    for (size_t dim = 0; dim < outputTileOffset.size(); dim++) {
        auto inCoord = isDoubleEqual(scales[dim], 1.0)
                               ? outputTileOffset[Dim(dim)]
                               : coordTransform(outputTileOffset[Dim(dim)], scales[dim], initialInputShape[Dim(dim)],
                                                initialOutputShape[Dim(dim)]);
        auto nearestPixel = nearestDim(inCoord, scales[dim]);
        inCoord = std::max(int64_t(0), std::min(nearestPixel, inputShape[Dim(dim)] - 1));
        inputTileOffset[Dim(dim)] = inCoord;
    }

    return inputTileOffset;
}

static Shape inferTileStartOffset(ShapeRef inputTileStartOffset, ShapeRef inputShape,
                                  VPU::SEInterpolateAttr seInperpAttr) {
    const auto scalesAttr = seInperpAttr.getScale();
    const auto scales = parseFPArrayAttr<double>(scalesAttr);
    const auto mode = seInperpAttr.getMode().getValue();

    // Get coordinate transformation function
    std::function<double(int64_t, double)> coordTransform;
    if (mode == VPU::NCEInterpolateMode::NEAREST) {
        coordTransform = [&](auto coord, auto scale) {
            return coord * scale;
        };
    } else if (mode == VPU::NCEInterpolateMode::BILINEAR) {
        coordTransform = [&](auto coord, auto scale) {
            return coord * scale;
        };
    } else {
        VPUX_THROW("SEInterpolateAttr: Unsupported NCEInterpolateMode {0}.", mode);
    }

    // The way, how nearest interpolate gets nearest pixel, depends on mode.
    // Default implementation is for bilinear
    std::function<int64_t(double, double)> topLeftMost = [](auto dim, auto scale) {
        VPUX_UNUSED(scale);
        return static_cast<int64_t>(std::round(dim));
    };

    auto outputTileOffset = Shape(inputTileStartOffset.toValues());
    // Get pixel in output which corresponds to this input pixel
    auto outputMaxShape = inferFullOutputShape(inputShape, scales, mode);
    for (size_t dim = 0; dim < inputTileStartOffset.size(); dim++) {
        auto outCoord = coordTransform(inputTileStartOffset[vpux::Dim(dim)], scales[dim]);
        auto rounded = topLeftMost(outCoord, scales[dim]);
        outCoord = std::max(int64_t(0), std::min(rounded, outputMaxShape[vpux::Dim(dim)] - 1));
        outputTileOffset[vpux::Dim(dim)] = outCoord;
    }

    return outputTileOffset;
}

VPU::SEAttr VPU::SEInterpolateAttr::extractTile(ShapeRef outputTileOffset, ShapeRef outputTileShape,
                                                ShapeRef inputShape, Shape& inputTileOffsets,
                                                Shape& inputTileShape) const {
    inputTileOffsets = backInferCoord(outputTileOffset, inputShape);
    Shape outputTileEnd(outputTileOffset.raw());
    std::transform(outputTileEnd.begin(), outputTileEnd.end(), outputTileShape.raw().begin(), outputTileEnd.begin(),
                   [](auto offset, auto size) {
                       return offset + size - 1;
                   });

    auto inputTileEnd = backInferCoord(outputTileEnd, inputShape);
    std::transform(inputTileOffsets.begin(), inputTileOffsets.end(), inputTileEnd.begin(), inputTileShape.begin(),
                   [](auto start, auto end) {
                       return end - start + 1;
                   });

    auto newOutputTileStart = inferTileStartOffset(inputTileOffsets, inputShape, *this);
    Shape relativeOffsets(inputTileOffsets);
    std::transform(outputTileOffset.raw().begin(), outputTileOffset.raw().end(), newOutputTileStart.begin(),
                   relativeOffsets.begin(), [](auto offset, auto outerTileOffset) {
                       return offset - outerTileOffset;
                   });

    return VPU::SEInterpolateAttr::get(getContext(), getMode(), getCoordinateTransformationMode(), getScale(),
                                       getNearestMode(), getIntArrayAttr(getContext(), relativeOffsets),
                                       getIntArrayAttr(getContext(), Shape(outputTileShape.raw())),
                                       getInitialInputShape(), getInitialOutputShape())
            .cast<VPU::SEAttr>();
}

std::vector<int32_t> VPU::SEInterpolateAttr::computeSEOffsets(ShapeRef dataShape, StridesRef /*dataStrides*/,
                                                              Byte elemSize, int64_t seSize) const {
    VPUX_THROW_UNLESS(dataShape.size() == 4, "Expected 4D data shape, got {0} dimensions", dataShape.size());
    VPUX_THROW_UNLESS(elemSize.count() > 0, "Invalid element byte size {0}", elemSize.count());
    VPUX_THROW_UNLESS(seSize > 0 && (seSize % 16 == 0), "Invalid Storage Element size {0}", seSize);
    const auto getInputAddress = [&](ShapeRef inputTileOffset) {
        const auto offsetC = inputTileOffset[Dims4D::Act::C];
        const auto offsetH = inputTileOffset[Dims4D::Act::H];
        const auto offsetW = inputTileOffset[Dims4D::Act::W];

        const auto inputC = dataShape[Dims4D::Act::C];
        const auto inputW = dataShape[Dims4D::Act::W];
        const auto pixelOffset = (offsetH * inputW + offsetW) * inputC;
        const auto channelOffset = offsetC;
        return (pixelOffset + channelOffset) * elemSize.count();
    };

    auto outputShape = inferOutputShape(dataShape);
    const auto outputOffsets =
            (getOffsets() != nullptr) ? parseIntArrayAttr<int64_t>(getOffsets()) : SmallVector<int64_t>({0, 0, 0, 0});

    const auto startH = outputOffsets[Dims4D::Act::H.ind()];
    const auto startW = outputOffsets[Dims4D::Act::W.ind()];
    const auto sizeH = outputShape[Dims4D::Act::H];
    const auto sizeW = outputShape[Dims4D::Act::W];

    const auto outputC = outputShape[Dims4D::Act::C];
    const auto seDepth = outputC / seSize;
    const auto seTableNumElements = sizeH * sizeW * seDepth;
    std::vector<int32_t> sePtrs(seTableNumElements, 0);

    for (int64_t h = 0; h < sizeH; ++h) {
        for (int64_t w = 0; w < sizeW; ++w) {
            for (int64_t se = 0; se < seDepth; ++se) {
                const auto outputTileOffset = Shape({0, se * seSize, startH + h, startW + w});

                const auto inputTileOffset = backInferCoord(outputTileOffset, dataShape);
                const auto offset = getInputAddress(inputTileOffset);

                const auto seSpatialOffset = (h * sizeW + w) * seDepth;
                sePtrs[seSpatialOffset + se] = offset;
            }
        }
    }

    return sePtrs;
}

//
// Generated
//
#include <vpux/compiler/dialect/VPU/enums.cpp.inc>
#include <vpux/compiler/dialect/VPU/structs.cpp.inc>

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/VPU/attributes.cpp.inc>
