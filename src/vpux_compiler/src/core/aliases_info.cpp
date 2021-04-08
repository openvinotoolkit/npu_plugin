//
// Copyright Intel Corporation.
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

#include "vpux/compiler/core/aliases_info.hpp"

#include "vpux/compiler/core/ops_interfaces.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

using namespace vpux;

vpux::AliasesInfo::AliasesInfo(mlir::FuncOp func) {
    const auto addAlias = [&](mlir::Value root, mlir::Value alias) {
        _aliases[root].insert(alias);
        _roots.insert({alias, root});
    };

    // Function arguments are roots for themselves
    for (const auto funcArg : func.getArguments()) {
        VPUX_THROW_UNLESS(funcArg.getType().isa<mlir::MemRefType>(),
                          "AliasesInfo analysis works only with MemRef types, got '{0}'", funcArg.getType());
        addAlias(funcArg, funcArg);
    }

    func.walk([&](mlir::Operation* op) {
        llvm::TypeSwitch<mlir::Operation*, void>(op)
                .Case<mlir::ViewLikeOpInterface>([&](mlir::ViewLikeOpInterface viewOp) {
                    const auto result = viewOp->getResult(0);
                    const auto source = viewOp.getViewSource();

                    VPUX_THROW_UNLESS(result.getType().isa<mlir::MemRefType>(),
                                      "AliasesInfo analysis works only with MemRef types, got '{0}'", result.getType());
                    VPUX_THROW_UNLESS(source.getType().isa<mlir::MemRefType>(),
                                      "AliasesInfo analysis works only with MemRef types, got '{0}'", source.getType());

                    const auto root = getRoot(source);
                    addAlias(root, result);
                })
                .Case<MultiViewOpInterface>([&](MultiViewOpInterface viewOp) {
                    for (const auto result : viewOp->getResults()) {
                        VPUX_THROW_UNLESS(result.getType().isa<mlir::MemRefType>(),
                                          "AliasesInfo analysis works only with MemRef types, got '{0}'",
                                          result.getType());

                        const auto source = viewOp.getViewSource(result.getResultNumber());
                        if (source == nullptr) {
                            addAlias(result, result);
                            continue;
                        }

                        VPUX_THROW_UNLESS(source.getType().isa<mlir::MemRefType>(),
                                          "AliasesInfo analysis works only with MemRef types, got '{0}'",
                                          source.getType());

                        const auto root = getRoot(source);
                        addAlias(root, result);
                    }
                })
                .Default([&](mlir::Operation* op) {
                    for (const auto result : op->getResults()) {
                        VPUX_THROW_UNLESS(result.getType().isa<mlir::MemRefType>(),
                                          "AliasesInfo analysis works only with MemRef types, got '{0}'",
                                          result.getType());

                        addAlias(result, result);
                    }
                });
    });
}

const AliasesInfo::ValuesSet& vpux::AliasesInfo::getAliases(mlir::Value val) const {
    const auto it = _aliases.find(val);
    VPUX_THROW_UNLESS(it != _aliases.end(), "Value '{0}' is not covered by aliases analysis", val);
    return it->second;
}

mlir::Value vpux::AliasesInfo::getRoot(mlir::Value val) const {
    const auto it = _roots.find(val);
    VPUX_THROW_UNLESS(it != _roots.end(), "Value '{0}' is not covered by aliases analysis", val);
    return it->second;
}
