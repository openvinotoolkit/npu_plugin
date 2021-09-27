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
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <llvm/ADT/SmallPtrSet.h>

using namespace vpux;

//
// updateFunctionSignature
//

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

//
// convertFunc
//

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

//
// getDefaultGreedyRewriteConfig
//

mlir::GreedyRewriteConfig vpux::getDefaultGreedyRewriteConfig() {
    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = true;
    config.maxIterations = 10;
    return config;
}

//
// appendLoc
//

mlir::Location vpux::appendLoc(mlir::Location baseLoc, StringRef suffix) {
    const auto suffixIdentifier = mlir::Identifier::get(suffix, baseLoc.getContext());
    const mlir::Location suffixLoc = mlir::NameLoc::get(suffixIdentifier);
    return mlir::FusedLoc::get(baseLoc.getContext(), {baseLoc, suffixLoc});
}

//
// BufferizeTypeConverter
//

vpux::BufferizeTypeConverter::BufferizeTypeConverter() {
    addConversion([](mlir::Type type) {
        return type;
    });

    addConversion([](mlir::RankedTensorType type) {
        const auto order = DimsOrder::fromType(type);
        const auto memref = mlir::MemRefType::get(type.getShape(), type.getElementType());
        return changeDimsOrder(memref, order);
    });

    addTargetMaterialization(dummyConverter<mlir::BaseMemRefType>);
    addArgumentMaterialization(dummyConverter<mlir::BaseMemRefType>);
    addSourceMaterialization(dummyConverter<mlir::TensorType>);
}

//
// populateBufferizeMaterializationLegality
//

void vpux::populateBufferizeMaterializationLegality(mlir::ConversionTarget& target) {
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
}
