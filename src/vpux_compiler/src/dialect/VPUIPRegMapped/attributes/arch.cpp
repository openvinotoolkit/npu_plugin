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

#include "vpux/compiler/dialect/VPUIPRegMapped/attributes/arch.hpp"

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/Builders.h>

using namespace vpux;

namespace {

constexpr StringLiteral archAttrName = "VPUIPRegMapped.arch";
constexpr StringLiteral derateFactorAttrName = "VPUIPRegMapped.derateFactor";
constexpr StringLiteral bandwidthAttrName = "VPUIPRegMapped.bandwidth";

constexpr int MAX_DPU_GROUPS_MTL = 2;
constexpr int MAX_DPU_GROUPS_KMB = 4;

}  // namespace

/*
void vpux::VPUIPRegMapped::setArch(mlir::ModuleOp module, ArchKind kind, Optional<int> numOfDPUGroups) {
    VPUX_THROW_UNLESS(module->hasAttr(archAttrName) == false,
                      "Architecture is already defined. Probably you don't need to run '--set-compile-params'.");

    module->setAttr(archAttrName, VPUIPRegMapped::ArchKindAttr::get(module.getContext(), kind));

    auto builder = mlir::OpBuilder::atBlockBegin(module.getBody());
    auto resources = builder.create<IE::RunTimeResourcesOp>(module.getLoc());

    const auto addMem = [&](VPUIPRegMapped::PhysicalMemory kind, Byte size, double derateFactor, uint32_t bandwidth) {
        auto mem =
                resources.addAvailableMemory(VPUIPRegMapped::PhysicalMemoryAttr::get(module.getContext(), kind), size);
        mem->setAttr(derateFactorAttrName, getFPAttr(module.getContext(), derateFactor));
        mem->setAttr(bandwidthAttrName, getIntAttr(module.getContext(), bandwidth));
    };

    // Size 192 found manually through experimentation. May be incorrect.
    // Changed to 500 for Deblur AA compilation
    addMem(VPUIPRegMapped::PhysicalMemory::DDR, 500_MB, 0.6, 8);

    if (kind == VPUIPRegMapped::ArchKind::TBH) {
        addMem(VPUIPRegMapped::PhysicalMemory::CSRAM, 24_MB, 0.85, 64);
    }

    // Run-time will use part of CMX to store DPU workload configuration.
    const Byte extraSpaceForWorkload = 128_KB;

    if (kind == VPUIPRegMapped::ArchKind::MTL) {
        addMem(VPUIPRegMapped::PhysicalMemory::CMX_NN, Byte(2_MB) - extraSpaceForWorkload, 1.0, 32);
    } else {
        addMem(VPUIPRegMapped::PhysicalMemory::CMX_NN, Byte(1_MB) - extraSpaceForWorkload, 1.0, 32);
    }

    const auto getProcKind = [&](VPUIPRegMapped::PhysicalProcessor kind) {
        return VPUIPRegMapped::PhysicalProcessorAttr::get(module.getContext(), kind);
    };

    const auto getDmaKind = [&](VPUIPRegMapped::DMAEngine kind) {
        return VPUIPRegMapped::DMAEngineAttr::get(module.getContext(), kind);
    };

    const auto getNumOfDPUGroupsVal = [&](int maxDpuGroups) {
        int numOfDPUGroupsVal = numOfDPUGroups.hasValue() ? numOfDPUGroups.getValue() : maxDpuGroups;
        VPUX_THROW_UNLESS(1 <= numOfDPUGroupsVal && numOfDPUGroupsVal <= maxDpuGroups,
                          "Invalid number of DPU groups: '{0}'", numOfDPUGroupsVal);
        return numOfDPUGroupsVal;
    };

    IERT::ExecutorResourceOp nceCluster;

    switch (kind) {
    case VPUIPRegMapped::ArchKind::MTL:
        resources.addExecutor(getDmaKind(DMAEngine::DMA_NN), 2);

        nceCluster = resources.addExecutor(getProcKind(PhysicalProcessor::NCE_Cluster),
                                           getNumOfDPUGroupsVal(MAX_DPU_GROUPS_MTL), true);
        nceCluster.addSubExecutor(getProcKind(PhysicalProcessor::NCE_PerClusterDPU), 1);

        break;

    default:
        if (kind == VPUIPRegMapped::ArchKind::TBH) {
            resources.addExecutor(getDmaKind(DMAEngine::DMA_NN), 2);
        } else {
            resources.addExecutor(getDmaKind(DMAEngine::DMA_NN), 1);
        }

        resources.addExecutor(getProcKind(PhysicalProcessor::SHAVE_UPA), 16);

        nceCluster = resources.addExecutor(getProcKind(PhysicalProcessor::NCE_Cluster),
                                           getNumOfDPUGroupsVal(MAX_DPU_GROUPS_KMB), true);
        nceCluster.addSubExecutor(getProcKind(PhysicalProcessor::NCE_PerClusterDPU), 5);
    }
}
*/

VPUIPRegMapped::ArchKind vpux::VPUIPRegMapped::getArch(mlir::ModuleOp module) {
    auto attr = module->getAttr(archAttrName);
    VPUX_THROW_UNLESS(attr != nullptr, "Module doesn't contain '{0}' attribute", archAttrName);
    VPUX_THROW_UNLESS(attr.isa<VPUIPRegMapped::ArchKindAttr>(), "Module attribute '{0}' has unsupported value '{1}'",
                      archAttrName, attr);
    return attr.cast<VPUIPRegMapped::ArchKindAttr>().getValue();
}

double vpux::VPUIPRegMapped::getMemoryDerateFactor(IE::MemoryResourceOp mem) {
    VPUX_THROW_UNLESS(mem.getKind() != nullptr, "Got empty memory resource kind");
    VPUX_THROW_UNLESS(mem.getKind().isa<VPUIPRegMapped::PhysicalMemoryAttr>(), "Unsupported memory resource kind '{0}'",
                      mem.getKind());

    auto attr = mem->getAttr(derateFactorAttrName);
    VPUX_THROW_UNLESS(attr != nullptr, "Memory resource '{0}' has no '{1}' attribute", mem.getKind(),
                      derateFactorAttrName);
    VPUX_THROW_UNLESS(attr.isa<mlir::FloatAttr>(), "Memory resource '{0}' has wrong '{1}' attribute : '{2}'",
                      mem.getKind(), derateFactorAttrName, attr);

    return attr.cast<mlir::FloatAttr>().getValueAsDouble();
}

uint32_t vpux::VPUIPRegMapped::getMemoryBandwidth(IE::MemoryResourceOp mem) {
    VPUX_THROW_UNLESS(mem.getKind() != nullptr, "Got empty memory resource kind");
    VPUX_THROW_UNLESS(mem.getKind().isa<VPUIPRegMapped::PhysicalMemoryAttr>(), "Unsupported memory resource kind '{0}'",
                      mem.getKind());

    auto attr = mem->getAttr(bandwidthAttrName);
    VPUX_THROW_UNLESS(attr != nullptr, "Memory resource '{0}' has no '{1}' attribute", mem.getKind(),
                      bandwidthAttrName);
    VPUX_THROW_UNLESS(attr.isa<mlir::IntegerAttr>(), "Memory resource '{0}' has wrong '{1}' attribute : '{2}'",
                      mem.getKind(), bandwidthAttrName, attr);

    return checked_cast<uint32_t>(attr.cast<mlir::IntegerAttr>().getInt());
}
