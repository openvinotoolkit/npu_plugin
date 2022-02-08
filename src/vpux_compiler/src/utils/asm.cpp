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

#include "vpux/compiler/utils/asm.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/STLExtras.h>

using namespace vpux;

//
// OptionalTypes
//

namespace {

bool isDefinedAbove(mlir::Value val, mlir::Operation* user) {
    auto* producer = val.getDefiningOp();
    if (producer == nullptr) {
        return true;
    }

    if (user != nullptr && producer->getBlock() == user->getBlock()) {
        return producer->isBeforeInBlock(user);
    }

    auto* producerRegion = producer->getParentRegion();
    auto* userAncestor = producerRegion->findAncestorOpInRegion(*user);

    return userAncestor != nullptr && producer->isBeforeInBlock(userAncestor);
}

bool canSkipOperandTypes(mlir::Operation* op) {
    return llvm::all_of(op->getOperands(), [op](mlir::Value val) {
        return isDefinedAbove(val, op);
    });
}

}  // namespace

void vpux::printOptionalTypes(mlir::OpAsmPrinter& printer, mlir::Operation* op, mlir::TypeRange types) {
    if (canSkipOperandTypes(op)) {
        return;
    }

    const auto nonNull = [](mlir::Type type) {
        return type != nullptr;
    };
    VPUX_THROW_UNLESS(llvm::all_of(types, nonNull), "Optional values are not supported");

    printer << ": ";
    llvm::interleaveComma(types, printer);
}

mlir::ParseResult vpux::parseOptionalTypes(mlir::OpAsmParser& parser, SmallVectorImpl<mlir::Type>& types) {
    if (parser.parseOptionalColonTypeList(types)) {
        return mlir::success();
    }

    return mlir::success();
}

mlir::ParseResult vpux::details::parseOptionalTypes(mlir::OpAsmParser& parser, ArrayRef<mlir::Type*> dst) {
    SmallVector<mlir::Type> types;
    if (parser.parseOptionalColonTypeList(types)) {
        return mlir::success();
    }

    if (types.empty()) {
        return mlir::success();
    }

    if (types.size() != dst.size()) {
        return printTo(parser.emitError(parser.getCurrentLocation()),
                       "Got unexpected number of types '{0}', expected '{1}'", types.size(), dst.size());
    }

    for (auto ind : irange(dst.size())) {
        *dst[ind] = types[ind];
    }

    return mlir::success();
}

//
// OptionalRegion
//

void vpux::printOptionalRegion(mlir::OpAsmPrinter& printer, mlir::Operation*, mlir::Region& region) {
    if (region.empty()) {
        return;
    }
    printer.printRegion(region);
}

mlir::ParseResult vpux::parseOptionalRegion(mlir::OpAsmParser& parser, mlir::Region& region) {
    const auto res = parser.parseOptionalRegion(region);
    return res.hasValue() ? res.getValue() : mlir::success();
}

//
// OptionalBlockRegion
//

void vpux::printOptionalBlockRegion(mlir::OpAsmPrinter& printer, mlir::Operation*, mlir::Region& region) {
    if (region.empty()) {
        return;
    }
    if (region.hasOneBlock() && region.front().empty()) {
        return;
    }
    printer.printRegion(region);
}

mlir::ParseResult vpux::parseOptionalBlockRegion(mlir::OpAsmParser& parser, mlir::Region& region) {
    const auto res = parser.parseOptionalRegion(region);
    if (res.hasValue()) {
        return res.getValue();
    }

    region.emplaceBlock();
    return mlir::success();
}
