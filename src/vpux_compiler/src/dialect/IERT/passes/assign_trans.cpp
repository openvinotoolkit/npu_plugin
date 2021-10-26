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

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "mlir/IR/Attributes.h"

#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"

using namespace vpux;

namespace {

//
// AssignTransPass
//

class AssignTransPass final : public IERT::AssignTransBase<AssignTransPass> {
public:
    explicit AssignTransPass(IERT::AttrCreateFunc memSpaceCb, Logger log): _memSpaceCb(std::move(memSpaceCb)) {
        VPUX_THROW_UNLESS(_memSpaceCb != nullptr, "Missing memSpaceCb");
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

private:
    IERT::AttrCreateFunc _memSpaceCb;
    mlir::Attribute _memSpace;
};

void AssignTransPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    // OpBuilderLogger builderLog(_log.nest());
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&netFunc.getBody().front().front(), &builderLog);

    std::cout << "AssignPass" << std::endl;
    // for (auto func : module.getOps<mlir::FuncOp>())
    {
        std::cout << "func.sym_name()" << netFunc.sym_name().str() << std::endl;

        // correct Assign
        IERT::AssignOp assign_op;
        netFunc.walk([&](IERT::AssignOp op) {
            std::cout << "netFunc.walk->AssignOp" << std::endl;
            assign_op = op;
            std::cout << "netFunc.walk->AssignOp END" << std::endl;
        });

        // auto readValueInput = mlir::RankedTensorType::get({1, 143}, mlir::Float16Type::get(ctx));
        auto outputResult = mlir::MemRefType::get({1, 143}, mlir::Float16Type::get(ctx));
        auto profilngResult = netFunc.getBody().front().addArgument(outputResult);

        builder.setInsertionPointAfter(assign_op);
        auto copyLoc2 = mlir::NameLoc::get(mlir::Identifier::get("CopyTemp", ctx));
        auto outputOp = builder.create<IERT::CopyOp>(copyLoc2, assign_op.output(), profilngResult);

        auto funcType = netFunc.getType();
        auto newResultTypes =
                to_small_vector(llvm::concat<const mlir::Type>(funcType.getResults(), makeArrayRef(outputResult)));
        auto newInputsTypes =
                to_small_vector(llvm::concat<const mlir::Type>(funcType.getInputs(), makeArrayRef(outputResult)));

        auto newFunctionType = mlir::FunctionType::get(ctx, newInputsTypes, newResultTypes);
        netFunc.setType(newFunctionType);


        // Adding output to the user info
        auto outputUserResult =
                getTensorType(outputResult.getShape(), outputResult.getElementType(), DimsOrder::fromType(outputResult));
        auto userInfoBuilder = mlir::OpBuilder::atBlockEnd(&netOp.outputsInfo().front(), &builderLog);
        userInfoBuilder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx), mlir::StringAttr::get(ctx, "AssignOutput"),
                                            mlir::TypeAttr::get(outputUserResult));


        // And to the returnOp
        netFunc.walk([&](mlir::ReturnOp op) {
            op.operandsMutable().append(outputOp.output());
        });

        // if (func.isExternal()) {
        //     _log.trace("Can't convert external Function '@{0}'", func.sym_name());
        //     signalPassFailure();
        // }

        // auto readValueInput = mlir::MemRefType::get({1, 143, 1, 1}, mlir::Float16Type::get(ctx));

        // change main function interface
        // auto funcType = netFunc.getType();
        // auto newInputsTypes =
        //         to_small_vector(llvm::concat<const mlir::Type>(funcType.getInputs(), makeArrayRef(readValueInput)));
        // auto newFunctionType = mlir::FunctionType::get(ctx, newInputsTypes, funcType.getResults());
        // netFunc.setType(newFunctionType);
        // auto netFuncResult = netFunc.getBody().front().addArgument(readValueInput);

        // Adding output to the user info
        // auto inputUserResult =
        //         getTensorType(readValueInput.getShape(), readValueInput.getElementType(), DimsOrder::fromType(readValueInput));
        // auto userInfoBuilder = mlir::OpBuilder::atBlockEnd(&netOp.inputsInfo().front(), &builderLog);
        // userInfoBuilder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx), mlir::StringAttr::get(ctx, "readValueInput"),
        //                                        mlir::TypeAttr::get(inputUserResult));

        // And to the returnOp
        // netFunc.walk([&](IERT::ReadValueOp op) {
        //     std::cout << "ReadValueOp" << std::endl;
        //     op.operandsMutable().append(netFuncResult);
        // });

        // SmallVector<mlir::BlockArgument> appendedEntryArgs;
        // updateFuncOp(func, appendedEntryArgs);
        // updateReturnOps(func, appendedEntryArgs);
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IERT::createAssignTransPass(AttrCreateFunc memSpaceCb, Logger log) {
    return std::make_unique<AssignTransPass>(std::move(memSpaceCb), log);
}
