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

}  // namespace

void vpux::VPUIP::setArch(mlir::ModuleOp module, ArchKind kind) {
    module->setAttr(archAttrName, VPUIP::ArchKindAttr::get(module.getContext(), kind));

    auto builder = mlir::OpBuilder::atBlockBegin(module.getBody());
    auto resources = builder.create<IERT::RunTimeResourcesOp>(module.getLoc());

    const auto addMem = [&](VPUIP::PhysicalMemory kind, Byte size, double derateFactor, uint32_t bandwidth) {
        auto mem = resources.addAvailableMemory(VPUIP::PhysicalMemoryAttr::get(module.getContext(), kind), size);
        mem->setAttr(derateFactorAttrName, getFP64Attr(module.getContext(), derateFactor));
        mem->setAttr(bandwidthAttrName, getInt64Attr(module.getContext(), bandwidth));
    };

    // Size 192 found manually through experimentation. May be incorrect.
    // Changed to 500 for Deblur AA compilation
    addMem(VPUIP::PhysicalMemory::DDR, 500_MB, 0.6, 8);

    if (kind == VPUIP::ArchKind::TBH) {
        addMem(VPUIP::PhysicalMemory::CSRAM, 24_MB, 0.85, 64);
    }

    // Run-time will use part of CMX to store DPU workload configuration.
    const Byte extraSpaceForWorkload = 128_KB;

    if (kind == VPUIP::ArchKind::MTL) {
        addMem(VPUIP::PhysicalMemory::CMX_NN, Byte(2_MB) - extraSpaceForWorkload, 1.0, 32);
    } else {
        addMem(VPUIP::PhysicalMemory::CMX_NN, Byte(1_MB) - extraSpaceForWorkload, 1.0, 32);
    }

    const auto getProcKind = [&](VPUIP::PhysicalProcessor kind) {
        return VPUIP::PhysicalProcessorAttr::get(module.getContext(), kind);
    };

    const auto getDmaKind = [&](VPUIP::DMAEngine kind) {
        return VPUIP::DMAEngineAttr::get(module.getContext(), kind);
    };

    IERT::ExecutorResourceOp nceCluster;

    switch (kind) {
    case VPUIP::ArchKind::MTL:
        resources.addExecutor(getDmaKind(DMAEngine::DMA_NN), 2);

        nceCluster = resources.addExecutor(getProcKind(PhysicalProcessor::NCE_Cluster), 1, true);
        nceCluster.addSubExecutor(getProcKind(PhysicalProcessor::NCE_PerClusterDPU), 1);

        break;

    default:
        if (kind == VPUIP::ArchKind::TBH) {
            resources.addExecutor(getDmaKind(DMAEngine::DMA_NN), 2);
        } else {
            resources.addExecutor(getDmaKind(DMAEngine::DMA_NN), 1);
        }

        resources.addExecutor(getProcKind(PhysicalProcessor::SHAVE_UPA), 16);

        nceCluster = resources.addExecutor(getProcKind(PhysicalProcessor::NCE_Cluster), 4, true);
        nceCluster.addSubExecutor(getProcKind(PhysicalProcessor::NCE_PerClusterDPU), 5);
    }
}

VPUIP::ArchKind vpux::VPUIP::getArch(mlir::ModuleOp module) {
    auto attr = module->getAttr(archAttrName);
    VPUX_THROW_UNLESS(attr != nullptr, "Module doesn't contain '{0}' attribute", archAttrName);
    VPUX_THROW_UNLESS(attr.isa<VPUIP::ArchKindAttr>(), "Module attribute '{0}' has unsupported value '{1}'",
                      archAttrName, attr);
    return attr.cast<VPUIP::ArchKindAttr>().getValue();
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
