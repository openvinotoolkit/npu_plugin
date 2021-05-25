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

#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/utils/extentions.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <llvm/ADT/SmallPtrSet.h>

using namespace vpux;

namespace {

mlir::LogicalResult updateFunctionSignature(mlir::FuncOp funcOp, ArrayRef<mlir::Type> newArgTypes,
                                            ArrayRef<mlir::Type> newResultTypes, Logger log) {
    const auto origFuncType = funcOp.getType();

    if (newArgTypes.size() != origFuncType.getNumInputs()) {
        log.trace("New inputs size '{0}' doesn't match original prototype", newArgTypes.size());
        return mlir::failure();
    }
    if (newResultTypes.size() != origFuncType.getNumResults()) {
        log.trace("New results size '{0}' doesn't match original prototype", newResultTypes.size());
        return mlir::failure();
    }

    const auto newFuncType = mlir::FunctionType::get(funcOp.getContext(), newArgTypes, newResultTypes);

    if (newFuncType == origFuncType) {
        log.trace("Nothing to change");
        return mlir::success();
    }

    log.trace("Update Function signature : '{0}' -> '{1}'", origFuncType, newFuncType);
    funcOp.setType(newFuncType);

    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::convertFunc(mlir::FuncOp funcOp, ArrayRef<mlir::Type> newArgTypes,
                                      ArrayRef<mlir::Type> newResultTypes, CvtOpBuilderCb cvtOpBuilder, Logger log) {
    log.trace("Convert Function '@{0}' prototype", funcOp.sym_name());
    log = log.nest();

    if (funcOp.isExternal()) {
        log.trace("Can't convert external Function '@{0}'", funcOp.sym_name());
        return mlir::failure();
    }

    if (updateFunctionSignature(funcOp, newArgTypes, newResultTypes, log).failed()) {
        return mlir::failure();
    }

    //
    // Convert arguments
    //

    log.trace("Convert arguments");

    for (const auto& p : funcOp.getArguments() | indexed) {
        const auto ind = checked_cast<uint32_t>(p.index());
        auto val = p.value();

        log.nest().trace("Process argument #{0}", ind);

        const auto origType = val.getType().cast<mlir::ShapedType>();
        const auto newType = newArgTypes[ind];

        if (newType == origType) {
            log.nest(2).trace("Nothing to change");
            continue;
        }

        log.nest(2).trace("Convert the argument type : '{0}' -> '{1}'", origType, newType);

        val.setType(newType);

        auto* firstUser = getFirstUser(val);
        if (firstUser == nullptr) {
            log.nest(2).trace("The argument has no users");
            continue;
        }

        OpBuilderLogger builderLog(log.nest(2));
        mlir::OpBuilder argBuilder(firstUser, &builderLog);

        auto* cvtOp = cvtOpBuilder(argBuilder, firstUser->getLoc(), val, origType);

        val.replaceAllUsesExcept(cvtOp->getResult(0), llvm::SmallPtrSet<mlir::Operation*, 1>{cvtOp});
    }

    //
    // Convert results
    //

    log.trace("Convert results");

    funcOp.walk([&](mlir::ReturnOp retOp) {
        log.nest().trace("Process return Operation '{0}'", retOp.getLoc());

        OpBuilderLogger builderLog(log.nest(3));
        mlir::OpBuilder resBuilder(retOp, &builderLog);

        for (const auto& p : retOp->getOperands() | indexed) {
            const auto ind = checked_cast<uint32_t>(p.index());
            auto val = p.value();

            log.nest(2).trace("Process result #{0}", ind);

            const auto origType = val.getType();
            const auto newType = newResultTypes[ind].cast<mlir::ShapedType>();

            if (newType == origType) {
                log.nest(3).trace("Nothing to change");
                continue;
            }

            log.nest(3).trace("Convert the result type : '{0}' -> '{1}'", newType, origType);

            auto* cvtOp = cvtOpBuilder(resBuilder, retOp.getLoc(), val, newType);

            retOp.setOperand(ind, cvtOp->getResult(0));
        }
    });

    return mlir::success();
}

mlir::LogicalResult vpux::convertBufferizedFunc(mlir::FuncOp funcOp, ArrayRef<mlir::Type> newArgTypes,
                                                ArrayRef<mlir::Type> newResultTypes,
                                                CvtBufferizedOpBuilderCb cvtOpBuilder, Logger log) {
    log.trace("Convert Function '@{0}' prototype", funcOp.sym_name());
    log = log.nest();

    if (funcOp.isExternal()) {
        log.trace("Can't convert external Function '@{0}'", funcOp.sym_name());
        return mlir::failure();
    }

    SmallVector<mlir::Type> newOperandTypes{newArgTypes.begin(), newArgTypes.end()};
    newOperandTypes.insert(newOperandTypes.end(), newResultTypes.begin(), newResultTypes.end());

    if (updateFunctionSignature(funcOp, newOperandTypes, newResultTypes, log).failed()) {
        return mlir::failure();
    }

    using OperationInserter = FuncRef<void(mlir::Operation*, CvtBufferizedOpBuilderCb, mlir::Value, size_t ind,
                                           mlir::ShapedType, Logger)>;
    const auto insertInOp = [](mlir::Operation* firstUser, CvtBufferizedOpBuilderCb cvtOpBuilder, mlir::Value operand,
                               size_t, mlir::ShapedType origType, Logger log) {
        OpBuilderLogger builderLog(log.nest(2));
        mlir::OpBuilder argBuilder(firstUser, &builderLog);

        log.nest(2).trace("Create AllocOp for output buffer and new operation");
        auto allocOp = argBuilder.create<mlir::memref::AllocOp>(firstUser->getLoc(), origType.cast<mlir::MemRefType>());
        auto* cvtOp = cvtOpBuilder(argBuilder, firstUser->getLoc(), operand, allocOp);

        VPUX_THROW_UNLESS(cvtOp->getNumResults() == 1, "Can insert operation with only one result");

        log.nest(2).trace("Replace the use of the current operand '{0}' with the result of the new operation '{1}'",
                          operand, cvtOp->getResult(0));
        operand.replaceAllUsesExcept(cvtOp->getResult(0), llvm::SmallPtrSet<mlir::Operation*, 1>{cvtOp});
    };

    const auto convertOperands = [&log, &cvtOpBuilder](ArrayRef<mlir::Type> newOperandsTypes, mlir::ValueRange operands,
                                                       OperationInserter insertNewOp) {
        const auto argsCount = newOperandsTypes.size();
        for (const auto& ind : irange(argsCount)) {
            log.nest().trace("Process argument #{0}", ind);

            auto operand = operands[ind];
            const auto origType = operand.getType().cast<mlir::ShapedType>();
            const auto newType = newOperandsTypes[ind];

            if (newType == origType) {
                log.nest(2).trace("Nothing to change");
                continue;
            }

            log.nest(2).trace("Convert the argument type : '{0}' -> '{1}'", origType, newType);

            operand.setType(newType);

            auto* firstUser = getFirstUser(operand);
            if (firstUser == nullptr) {
                log.nest(2).trace("The argument has no users");
                continue;
            }

            insertNewOp(firstUser, cvtOpBuilder, operand, ind, origType, log);
        }
    };

    log.trace("Convert arguments");
    convertOperands(newArgTypes, funcOp.getArguments(), insertInOp);

    log.trace("Convert results");
    funcOp.walk([&](mlir::ReturnOp retOp) {
        log.nest().trace("Process return Operation '{0}'", retOp.getLoc());

        OpBuilderLogger builderLog(log.nest(3));
        mlir::OpBuilder resBuilder(retOp, &builderLog);

        const auto results = retOp->getOperands();
        const auto insertOutOp = [&](mlir::Operation* firstUser, CvtBufferizedOpBuilderCb cvtOpBuilder,
                                     mlir::Value operand, size_t ind, mlir::ShapedType origType, Logger log) {
            const auto users = operand.getUsers();
            const auto usersCount = std::distance(users.begin(), users.end());
            VPUX_THROW_UNLESS(usersCount == 1, "There should be only one user for the output buffer");

            OpBuilderLogger builderLog(log.nest(2));
            mlir::OpBuilder argBuilder(firstUser, &builderLog);

            log.nest(2).trace("Create AllocOp for output buffer and new operation");
            auto allocOp =
                    argBuilder.create<mlir::memref::AllocOp>(firstUser->getLoc(), origType.cast<mlir::MemRefType>());
            auto* cvtOp = cvtOpBuilder(resBuilder, retOp->getLoc(), results[ind], operand);

            VPUX_THROW_UNLESS(cvtOp->getNumResults() == 1, "Can insert operation with only one result");

            log.nest(2).trace("Replace the use of the current operand '{0}' with the result of the AllocOp '{1}'",
                              operand, allocOp.getResult());
            operand.replaceAllUsesExcept(allocOp.getResult(), llvm::SmallPtrSet<mlir::Operation*, 1>{cvtOp});
            retOp.setOperand(checked_cast<unsigned>(ind), cvtOp->getResult(0));
        };

        SmallVector<mlir::Value> outBuffers{funcOp.getArguments().begin() + newArgTypes.size(),
                                            funcOp.getArguments().end()};
        convertOperands(newResultTypes, outBuffers, insertOutOp);
    });

    return mlir::success();
}
