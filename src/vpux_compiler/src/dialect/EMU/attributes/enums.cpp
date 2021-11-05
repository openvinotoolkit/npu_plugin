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

#include "vpux/compiler/dialect/EMU/attributes/enums.hpp"

#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/StringExtras.h>

using namespace vpux;

namespace {

constexpr StringLiteral compilationModeAttrName = "EMU.compilationMode";

}  // namespace

void vpux::EMU::setCompilationMode(mlir::ModuleOp module, CompilationMode compilationMode) {
    module->setAttr(compilationModeAttrName, EMU::CompilationModeAttr::get(module.getContext(), compilationMode));
}

EMU::CompilationMode vpux::EMU::getCompilationMode(mlir::Operation* op) {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_UNLESS(module != nullptr, "Can't get parent Module from Operation '{0}' at '{1}'", op->getName(),
                      op->getLoc());

    if (auto attr = module->getAttr(compilationModeAttrName)) {
        VPUX_THROW_UNLESS(attr.isa<EMU::CompilationModeAttr>(), "Module attribute '{0}' has unsupported value '{1}'",
                          compilationModeAttrName, attr);

        return attr.cast<EMU::CompilationModeAttr>().getValue();
    }

    // Use ReferenceHW as a default mode
    return EMU::CompilationMode::ReferenceHW;
}

//
// Generated
//

#include <vpux/compiler/dialect/EMU/generated/attributes/enums.cpp.inc>
