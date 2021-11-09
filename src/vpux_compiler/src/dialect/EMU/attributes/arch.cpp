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

#include "vpux/compiler/dialect/EMU/attributes/arch.hpp"

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/Builders.h>

using namespace vpux;

namespace {

constexpr StringLiteral archAttrName = "EMU.arch";

}  // namespace

void vpux::EMU::setArch(mlir::ModuleOp module, ArchKind kind) {
    VPUX_THROW_UNLESS(module->hasAttr(archAttrName) == false,
                      "Architecture is already defined. Probably you don't need to run '--set-compile-params'.");
    module->setAttr(archAttrName, EMU::ArchKindAttr::get(module.getContext(), kind));
}

EMU::ArchKind vpux::EMU::getArch(mlir::ModuleOp module) {
    if (auto attr = module->getAttr(archAttrName)) {
        VPUX_THROW_UNLESS(attr.isa<EMU::ArchKindAttr>(), "Module attribute '{0}' has unsupported value '{1}'",
                          archAttrName, attr);
        return attr.cast<EMU::ArchKindAttr>().getValue();
    }

    return EMU::ArchKind::UNKNOWN;
}
