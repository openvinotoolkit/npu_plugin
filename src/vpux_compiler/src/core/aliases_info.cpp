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

#include "vpux/compiler/core/aliases_info.hpp"

#include "vpux/compiler/core/ops_interfaces.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

std::string getValueForLog(mlir::Value val) {
    if (const auto arg = val.dyn_cast<mlir::BlockArgument>()) {
        return llvm::formatv("BlockArgument #{0} at '{0}'", arg.getArgNumber(), val.getLoc()).str();
    }

    const auto res = val.cast<mlir::OpResult>();
    return llvm::formatv("Operation result #{0} for '{1}' at '{2}'", res.getResultNumber(), res.getOwner()->getName(),
                         val.getLoc());
}

}  // namespace

vpux::AliasesInfo::AliasesInfo(mlir::FuncOp func): _log(Logger::global().nest("aliases-info", 0)) {
    _log.trace("Analyze aliases for Function '@{0}'", func.getName());
    _log = _log.nest();

    _log.trace("Function arguments are roots for themselves");
    _log = _log.nest();
    for (const auto funcArg : func.getArguments()) {
        _log.trace("Argument #{0}", funcArg.getArgNumber());

        VPUX_THROW_UNLESS(funcArg.getType().isa<mlir::MemRefType>(),
                          "AliasesInfo analysis works only with MemRef types, got '{0}'", funcArg.getType());
        addAlias(funcArg, funcArg);
    }
    _log = _log.unnest();

    _log.trace("Traverse the Function body");
    _log = _log.nest();
    traverse(func.getOps());
}

void vpux::AliasesInfo::addAlias(mlir::Value root, mlir::Value alias) {
    _log.trace("Add alias '{0}' for '{1}'", getValueForLog(alias), getValueForLog(root));

    _aliases[root].insert(alias);
    _roots.insert({alias, root});
}

void vpux::AliasesInfo::traverse(OpRange ops) {
    for (auto& op : ops) {
        llvm::TypeSwitch<mlir::Operation*, void>(&op)
                .Case<mlir::ViewLikeOpInterface>([&](mlir::ViewLikeOpInterface viewOp) {
                    _log.trace("Got ViewLike Operation '{0}' at '{1}'", viewOp->getName(), viewOp->getLoc());
                    _log = _log.nest();

                    const auto result = viewOp->getResult(0);
                    const auto source = viewOp.getViewSource();

                    VPUX_THROW_UNLESS(result.getType().isa<mlir::MemRefType>(),
                                      "AliasesInfo analysis works only with MemRef types, got '{0}'", result.getType());
                    VPUX_THROW_UNLESS(source.getType().isa<mlir::MemRefType>(),
                                      "AliasesInfo analysis works only with MemRef types, got '{0}'", source.getType());

                    const auto root = getRoot(source);
                    addAlias(root, result);

                    _log = _log.unnest();
                })
                .Case<MultiViewOpInterface>([&](MultiViewOpInterface viewOp) {
                    _log.trace("Got MultiView Operation '{0}' at '{1}'", viewOp->getName(), viewOp->getLoc());
                    _log = _log.nest();

                    for (const auto result : viewOp->getResults()) {
                        _log.trace("Result #{0}", result.getResultNumber());

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

                    _log = _log.unnest();
                })
                .Case<mlir::async::ExecuteOp>([&](mlir::async::ExecuteOp executeOp) {
                    // It looks like `async::ExecuteOp` doesn't implement `RegionBranchOpInterface` correctly.
                    // At least it doesn't report correspondance between its operands and the body region arguments.

                    _log.trace("Got 'async.execute' Operation at '{0}'", executeOp->getLoc());
                    _log = _log.nest();

                    const auto outerArgs = executeOp.operands();
                    const auto innerArgs = executeOp.body().getArguments();
                    VPUX_THROW_UNLESS(
                            outerArgs.size() == innerArgs.size(),
                            "Mismatch between 'async.execute' operands and its body region arguments at '{0}'",
                            executeOp->getLoc());

                    for (auto i : irange(outerArgs.size())) {
                        _log.trace("Check operand #{0} and corresponding region argument", i);

                        const auto futureType = outerArgs[i].getType().dyn_cast<mlir::async::ValueType>();
                        VPUX_THROW_UNLESS(futureType != nullptr,
                                          "AliasesInfo analysis works only with !async.value<MemRef> types, got '{0}'",
                                          outerArgs[i].getType());

                        VPUX_THROW_UNLESS(futureType.getValueType().isa<mlir::MemRefType>(),
                                          "AliasesInfo analysis works only with MemRef types, got '{0}'", futureType);
                        VPUX_THROW_UNLESS(innerArgs[i].getType().isa<mlir::MemRefType>(),
                                          "AliasesInfo analysis works only with MemRef types, got '{0}'",
                                          innerArgs[i].getType());

                        const auto root = getRoot(outerArgs[i]);
                        addAlias(root, innerArgs[i]);
                    }

                    _log.trace("Traverse the 'async.execute' body");
                    _log = _log.nest();
                    traverse(executeOp.getOps());
                    _log = _log.unnest();

                    const auto outerResults = executeOp.results();
                    const auto innerResults = executeOp.body().front().getTerminator()->getOperands();
                    VPUX_THROW_UNLESS(
                            innerResults.size() == outerResults.size(),
                            "Mismatch between 'async.yield' operands and its parent 'async.execute' results at '{0}'",
                            executeOp->getLoc());

                    for (auto i : irange(innerResults.size())) {
                        _log.trace("Check result #{0} and corresponding region result", i);

                        const auto futureType = outerResults[i].getType().dyn_cast<mlir::async::ValueType>();
                        VPUX_THROW_UNLESS(futureType != nullptr,
                                          "AliasesInfo analysis works only with !async.value<MemRef> types, got '{0}'",
                                          outerResults[i].getType());

                        VPUX_THROW_UNLESS(innerResults[i].getType().isa<mlir::MemRefType>(),
                                          "AliasesInfo analysis works only with MemRef types, got '{0}'",
                                          innerResults[i].getType());
                        VPUX_THROW_UNLESS(futureType.getValueType().isa<mlir::MemRefType>(),
                                          "AliasesInfo analysis works only with MemRef types, got '{0}'", futureType);

                        const auto root = getRoot(innerResults[i]);
                        addAlias(root, outerResults[i]);
                    }

                    _log = _log.unnest();
                })
                .Case<mlir::async::AwaitOp>([&](mlir::async::AwaitOp waitOp) {
                    _log.trace("Got 'async.await' Operation at '{0}'", waitOp->getLoc());
                    _log = _log.nest();

                    if (const auto result = waitOp.result()) {
                        const auto futureType = waitOp.operand().getType().dyn_cast<mlir::async::ValueType>();
                        VPUX_THROW_UNLESS(futureType != nullptr,
                                          "AliasesInfo analysis works only with !async.value<MemRef> types, got '{0}'",
                                          waitOp.operand().getType());

                        VPUX_THROW_UNLESS(futureType.getValueType().isa<mlir::MemRefType>(),
                                          "AliasesInfo analysis works only with MemRef types, got '{0}'", futureType);
                        VPUX_THROW_UNLESS(result.getType().isa<mlir::MemRefType>(),
                                          "AliasesInfo analysis works only with MemRef types, got '{0}'",
                                          result.getType());

                        const auto root = getRoot(waitOp.operand());
                        addAlias(root, result);
                    }

                    _log = _log.unnest();
                })
                .Default([&](mlir::Operation* op) {
                    _log.trace("Got generic Operation '{0}' at '{1}'", op->getName(), op->getLoc());
                    _log = _log.nest();

                    for (const auto result : op->getResults()) {
                        if (result.getType().isa<mlir::MemRefType>()) {
                            addAlias(result, result);
                        }
                    }

                    _log = _log.unnest();
                });
    }
}

mlir::Value vpux::AliasesInfo::getRoot(mlir::Value val) const {
    const auto it = _roots.find(val);
    VPUX_THROW_UNLESS(it != _roots.end(), "Value '{0}' is not covered by aliases analysis", getValueForLog(val));
    return it->second;
}

const AliasesInfo::ValuesSet& vpux::AliasesInfo::getAliases(mlir::Value val) const {
    const auto it = _aliases.find(val);
    VPUX_THROW_UNLESS(it != _aliases.end(), "Value '{0}' is not covered by aliases analysis", getValueForLog(val));
    return it->second;
}
