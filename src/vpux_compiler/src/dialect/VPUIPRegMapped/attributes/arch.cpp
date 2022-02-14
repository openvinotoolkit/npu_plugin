//
// Copyright (C) 2022 Intel Corporation.
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
