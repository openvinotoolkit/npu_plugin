//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/dialect.hpp"

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

using namespace vpux;

//
// initialize
//

void vpux::VPUIP::VPUIPDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPUIP/generated/ops.cpp.inc>
            >();

    registerAttributes();
    registerTypes();
}

//
// Operation executor attributes
//

namespace {

constexpr StringLiteral executorAttrName = "VPUIP.executor";
constexpr StringLiteral executorInstanceMaskAttrName = "VPUIP.executorIdx";

}  // namespace

void vpux::VPUIP::VPUIPDialect::setExecutor(mlir::async::ExecuteOp execOp, IndexedSymbolAttr executor) {
    VPUX_THROW_UNLESS(executor != nullptr, "Got an empty executor");
    execOp->setAttr(executorAttrName, executor);
}

void vpux::VPUIP::VPUIPDialect::setExecutorInstanceMask(mlir::async::ExecuteOp execOp,
                                                        mlir::ArrayAttr executorInstanceMask) {
    VPUX_THROW_UNLESS(executorInstanceMask != nullptr, "Got an empty executor instance");
    execOp->setAttr(executorInstanceMaskAttrName, executorInstanceMask);

    // In case of some DMA tasks port is also maintained as part of operation iself
    if (getExecutorKind(execOp) == VPU::ExecutorKind::DMA_NN) {
        if (executorInstanceMask.getValue().size() != 1) {
            // In case there are multiple executors subsequent passes should take care
            // of handling this
            return;
        }

        auto portIdxAttr = executorInstanceMask[0].cast<mlir::IntegerAttr>();

        auto* bodyBlock = &execOp.body().front();
        for (auto& innerOp : bodyBlock->getOperations()) {
            if (auto copyOp = mlir::dyn_cast<VPUIP::DepthToSpaceDMAOp>(innerOp)) {
                copyOp.portAttr(portIdxAttr);
            } else if (auto copyOp = mlir::dyn_cast<VPUIP::SpaceToDepthDMAOp>(innerOp)) {
                copyOp.portAttr(portIdxAttr);
            } else if (auto copyOp = mlir::dyn_cast<VPUIP::PermuteDMAOp>(innerOp)) {
                copyOp.portAttr(portIdxAttr);
            } else if (auto copyOp = mlir::dyn_cast<VPUIP::ExpandDMAOp>(innerOp)) {
                copyOp.portAttr(portIdxAttr);
            }
        }
    }
}

llvm::StringLiteral vpux::VPUIP::VPUIPDialect::getExecutorAttrName() {
    return executorAttrName;
}

IndexedSymbolAttr vpux::VPUIP::VPUIPDialect::getExecutor(mlir::async::ExecuteOp execOp) {
    const auto executor = execOp->getAttr(executorAttrName);
    VPUX_THROW_UNLESS(executor != nullptr, "Can't find Executor attributes for Operation at '{0}'", execOp->getLoc());

    const auto executorSymbol = executor.dyn_cast<IndexedSymbolAttr>();
    VPUX_THROW_UNLESS(executorSymbol != nullptr, "Unsupported Executor attribute '{0}'", executorSymbol);

    return executorSymbol;
}

VPU::ExecutorKind vpux::VPUIP::VPUIPDialect::getExecutorKind(mlir::async::ExecuteOp execOp) {
    const auto executor = getExecutor(execOp);
    const auto executorKind = executor.getLeafNameAttr().dyn_cast<VPU::ExecutorKindAttr>();
    VPUX_THROW_UNLESS(executorKind != nullptr, "Unsupported Executor Kind attribute '{0}'", executor.getLeafNameAttr());

    return executorKind.getValue();
}

bool vpux::VPUIP::VPUIPDialect::hasExecutorInstanceMask(mlir::async::ExecuteOp execOp) {
    return (execOp->getAttr(executorInstanceMaskAttrName) != nullptr);
}

mlir::ArrayAttr vpux::VPUIP::VPUIPDialect::getExecutorInstanceMask(mlir::async::ExecuteOp execOp) {
    const auto executorInstanceMask = execOp->getAttr(executorInstanceMaskAttrName);
    VPUX_THROW_UNLESS(executorInstanceMask != nullptr, "Can't find Executor Instance attributes for Operation at '{0}'",
                      execOp->getLoc());

    const auto executorInstanceMaskSymbol = executorInstanceMask.dyn_cast<mlir::ArrayAttr>();
    VPUX_THROW_UNLESS(executorInstanceMaskSymbol != nullptr, "Unsupported Executor Instance attribute '{0}'",
                      executorInstanceMaskSymbol);

    return executorInstanceMaskSymbol;
}

//
// materializeConstant
//

mlir::Operation* vpux::VPUIP::VPUIPDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value,
                                                                mlir::Type type, mlir::Location loc) {
    if (!value.isa<Const::ContentAttr>()) {
        (void)errorAt(loc, "Can't materialize VPUIP Constant from Attribute '{0}'", value);
        return nullptr;
    }

    if (!type.isa<mlir::MemRefType>()) {
        (void)errorAt(loc, "Can't materialize VPUIP Constant for Type '{0}'", type);
        return nullptr;
    }

    return builder.create<Const::DeclareOp>(loc, type.cast<vpux::NDTypeInterface>().eraseTiledInfo(),
                                            value.cast<Const::ContentAttr>());
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/generated/dialect.cpp.inc>
