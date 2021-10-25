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

#include "vpux/compiler/dialect/VPUIP/attributes/arch.hpp"

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/Builders.h>

using namespace vpux;

namespace {

constexpr StringLiteral archAttrName = "VPUIP.arch";
constexpr StringLiteral derateFactorAttrName = "VPUIP.derateFactor";
constexpr StringLiteral bandwidthAttrName = "VPUIP.bandwidth";
constexpr StringLiteral processorFrequencyAttrName = "VPUIP.processorFrequency";

constexpr int KMB_MAX_DPU_GROUPS = 4;
constexpr int MTL_MAX_DPU_GROUPS = 2;

constexpr Byte DDR_HEAP_SIZE = 500_MB;
constexpr Byte CSRAM_SIZE = 24_MB;

// Run-time will use part of CMX to store DPU workload configuration.
constexpr Byte EXTRA_SPACE_FOR_DPU_WORKLOAD_QUEUE = 128_KB;

constexpr Byte KMB_CMX_SIZE = Byte(1_MB) - EXTRA_SPACE_FOR_DPU_WORKLOAD_QUEUE;
constexpr Byte MTL_CMX_SIZE = Byte(2_MB) - EXTRA_SPACE_FOR_DPU_WORKLOAD_QUEUE;

}  // namespace

void vpux::VPUIP::setArch(mlir::ModuleOp module, ArchKind kind, Optional<int> numOfDPUGroups) {
    VPUX_THROW_UNLESS(module->hasAttr(archAttrName) == false,
                      "Architecture is already defined. Probably you don't need to run '--set-compile-params'.");

    module->setAttr(archAttrName, VPUIP::ArchKindAttr::get(module.getContext(), kind));

    auto builder = mlir::OpBuilder::atBlockBegin(module.getBody());
    auto resources = builder.create<IERT::RunTimeResourcesOp>(module.getLoc());

    const auto addMem = [&](VPUIP::PhysicalMemory kind, Byte size, double derateFactor, uint32_t bandwidth) {
        auto mem = resources.addAvailableMemory(VPUIP::PhysicalMemoryAttr::get(module.getContext(), kind), size);
        mem->setAttr(derateFactorAttrName, getFPAttr(module.getContext(), derateFactor));
        mem->setAttr(bandwidthAttrName, getIntAttr(module.getContext(), bandwidth));
    };

    const auto getProcKind = [&](VPUIP::PhysicalProcessor kind) {
        return VPUIP::PhysicalProcessorAttr::get(module.getContext(), kind);
    };

    const auto getDmaKind = [&](VPUIP::DMAEngine kind) {
        return VPUIP::DMAEngineAttr::get(module.getContext(), kind);
    };

    const auto getNumOfDPUGroupsVal = [&](int maxDpuGroups) {
        int numOfDPUGroupsVal = numOfDPUGroups.hasValue() ? numOfDPUGroups.getValue() : maxDpuGroups;
        VPUX_THROW_UNLESS(1 <= numOfDPUGroupsVal && numOfDPUGroupsVal <= maxDpuGroups,
                          "Invalid number of DPU groups: '{0}'", numOfDPUGroupsVal);
        return numOfDPUGroupsVal;
    };

    IERT::ExecutorResourceOp nceCluster;

    switch (kind) {
    case VPUIP::ArchKind::KMB: {
        addMem(VPUIP::PhysicalMemory::DDR, DDR_HEAP_SIZE, 0.6, 8);
        addMem(VPUIP::PhysicalMemory::CMX_NN, KMB_CMX_SIZE, 1.0, 32);

        resources.addExecutor(getDmaKind(DMAEngine::DMA_NN), 1);

        resources.addExecutor(getProcKind(PhysicalProcessor::SHAVE_UPA), 16);

        nceCluster = resources.addExecutor(getProcKind(PhysicalProcessor::NCE_Cluster),
                                           getNumOfDPUGroupsVal(KMB_MAX_DPU_GROUPS), true);
        nceCluster.addSubExecutor(getProcKind(PhysicalProcessor::NCE_PerClusterDPU), 5);

        break;
    }
    case VPUIP::ArchKind::TBH: {
        addMem(VPUIP::PhysicalMemory::DDR, DDR_HEAP_SIZE, 0.6, 8);
        addMem(VPUIP::PhysicalMemory::CSRAM, CSRAM_SIZE, 0.85, 64);
        addMem(VPUIP::PhysicalMemory::CMX_NN, KMB_CMX_SIZE, 1.0, 32);

        resources.addExecutor(getDmaKind(DMAEngine::DMA_NN), 2);

        resources.addExecutor(getProcKind(PhysicalProcessor::SHAVE_UPA), 16);

        nceCluster = resources.addExecutor(getProcKind(PhysicalProcessor::NCE_Cluster),
                                           getNumOfDPUGroupsVal(KMB_MAX_DPU_GROUPS), true);
        nceCluster.addSubExecutor(getProcKind(PhysicalProcessor::NCE_PerClusterDPU), 5);

        break;
    }
    case VPUIP::ArchKind::MTL: {
        addMem(VPUIP::PhysicalMemory::DDR, DDR_HEAP_SIZE, 0.6, 8);
        addMem(VPUIP::PhysicalMemory::CMX_NN, MTL_CMX_SIZE, 1.0, 32);

        resources.addExecutor(getDmaKind(DMAEngine::DMA_NN), 2);

        nceCluster = resources.addExecutor(getProcKind(PhysicalProcessor::NCE_Cluster),
                                           getNumOfDPUGroupsVal(MTL_MAX_DPU_GROUPS), true);
        nceCluster.addSubExecutor(getProcKind(PhysicalProcessor::NCE_PerClusterDPU), 1);

        break;
    }
    default:
        VPUX_THROW("Unsupported architecture '{0}'", kind);
    }

    nceCluster->setAttr(processorFrequencyAttrName, getFPAttr(module.getContext(), 700.0));
}

VPUIP::ArchKind vpux::VPUIP::getArch(mlir::ModuleOp module) {
    if (auto attr = module->getAttr(archAttrName)) {
        VPUX_THROW_UNLESS(attr.isa<VPUIP::ArchKindAttr>(), "Module attribute '{0}' has unsupported value '{1}'",
                          archAttrName, attr);
        return attr.cast<VPUIP::ArchKindAttr>().getValue();
    }

    return VPUIP::ArchKind::UNKNOWN;
}

double vpux::VPUIP::getMemoryDerateFactor(IERT::MemoryResourceOp mem) {
    VPUX_THROW_UNLESS(mem.kindAttr() != nullptr, "Unsupported memory resource kind '{0}'", mem.kind());
    VPUX_THROW_UNLESS(mem.kindAttr().isa<VPUIP::PhysicalMemoryAttr>(), "Unsupported memory resource kind '{0}'",
                      mem.kind());

    auto attr = mem->getAttr(derateFactorAttrName);
    VPUX_THROW_UNLESS(attr != nullptr, "Memory resource '{0}' has no '{1}' attribute", mem.kind(),
                      derateFactorAttrName);
    VPUX_THROW_UNLESS(attr.isa<mlir::FloatAttr>(), "Memory resource '{0}' has wrong '{1}' attribute : '{2}'",
                      mem.kind(), derateFactorAttrName, attr);

    return attr.cast<mlir::FloatAttr>().getValueAsDouble();
}

uint32_t vpux::VPUIP::getMemoryBandwidth(IERT::MemoryResourceOp mem) {
    VPUX_THROW_UNLESS(mem.kindAttr() != nullptr, "Unsupported memory resource kind '{0}'", mem.kind());
    VPUX_THROW_UNLESS(mem.kindAttr().isa<VPUIP::PhysicalMemoryAttr>(), "Unsupported memory resource kind '{0}'",
                      mem.kind());

    auto attr = mem->getAttr(bandwidthAttrName);
    VPUX_THROW_UNLESS(attr != nullptr, "Memory resource '{0}' has no '{1}' attribute", mem.kind(), bandwidthAttrName);
    VPUX_THROW_UNLESS(attr.isa<mlir::IntegerAttr>(), "Memory resource '{0}' has wrong '{1}' attribute : '{2}'",
                      mem.kind(), bandwidthAttrName, attr);

    return checked_cast<uint32_t>(attr.cast<mlir::IntegerAttr>().getInt());
}

double vpux::VPUIP::getProcessorFrequency(IERT::ExecutorResourceOp res) {
    VPUX_THROW_UNLESS(res.kindAttr() != nullptr, "Unsupported executor resource kind '{0}'", res.kind());

    auto attr = res->getAttr(processorFrequencyAttrName);
    VPUX_THROW_UNLESS(attr != nullptr, "Executor resource '{0}' has no '{1}' attribute", res.kind(),
                      processorFrequencyAttrName);
    VPUX_THROW_UNLESS(attr.isa<mlir::FloatAttr>(), "Executor resource '{0}' has wrong '{1}' attribute : '{2}'",
                      res.kind(), processorFrequencyAttrName, attr);

    return attr.cast<mlir::FloatAttr>().getValueAsDouble();
}

StringLiteral vpux::VPUIP::getProcessorFrequencyAttrName() {
    return processorFrequencyAttrName;
}

StringLiteral vpux::VPUIP::getBandwidthAttrName() {
    return bandwidthAttrName;
}
