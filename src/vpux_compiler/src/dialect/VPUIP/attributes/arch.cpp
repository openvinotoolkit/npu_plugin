//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/VPUIP/attributes/arch.hpp"

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/Builders.h>

using namespace vpux;

namespace {

const StringLiteral archAttrName = "VPUIP.arch";
const StringLiteral derateFactorAttrName = "VPUIP.derateFactor";
const StringLiteral bandwidthAttrName = "VPUIP.bandwidth";

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

    resources.addAvailableMemory(nullptr, 1_GB);
    addMem(VPUIP::PhysicalMemory::DDR, 30_MB, 0.6, 8);
    addMem(VPUIP::PhysicalMemory::CMX_UPA, 4_MB, 0.85, 16);
    addMem(VPUIP::PhysicalMemory::CMX_NN, 1_MB, 1.0, 32);
    if (kind == VPUIP::ArchKind::MA3100) {
        addMem(VPUIP::PhysicalMemory::CSRAM, 24_MB, 0.85, 64);
    }

    const auto getProcKind = [&](VPUIP::PhysicalProcessor kind) {
        return VPUIP::PhysicalProcessorAttr::get(module.getContext(), kind);
    };

    const auto getDmaKind = [&](VPUIP::DMAEngine kind) {
        return VPUIP::DMAEngineAttr::get(module.getContext(), kind);
    };
    resources.addAvailableExecutor(getProcKind(PhysicalProcessor::Leon_RT), 1);
    resources.addAvailableExecutor(getProcKind(PhysicalProcessor::Leon_NN), 1);
    resources.addAvailableExecutor(getProcKind(PhysicalProcessor::SHAVE_UPA), 16);
    resources.addAvailableExecutor(getProcKind(PhysicalProcessor::SHAVE_NN), 20);
    auto nceCluster = resources.addAvailableExecutor(getProcKind(PhysicalProcessor::NCE_Cluster), 4);
    nceCluster.addAvailableExecutor(getProcKind(PhysicalProcessor::NCE_PerClusterDPU), 5);
    resources.addAvailableExecutor(getDmaKind(DMAEngine::DMA_UPA), 1);
    if (kind == VPUIP::ArchKind::MA3100) {
        resources.addAvailableExecutor(getDmaKind(DMAEngine::DMA_NN), 2);
    } else {
        resources.addAvailableExecutor(getDmaKind(DMAEngine::DMA_NN), 1);
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
