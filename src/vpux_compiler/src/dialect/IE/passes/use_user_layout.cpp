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

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// UseUserLayoutPass
//

class UseUserLayoutPass final : public IE::UseUserLayoutBase<UseUserLayoutPass> {
public:
    explicit UseUserLayoutPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

//
// safeRunOnModule
//

void UseUserLayoutPass::safeRunOnModule() {
    auto module = getOperation();

    IE::CNNNetworkOp netInfo;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netInfo, netFunc);

    const auto funcType = netFunc.getType();

    auto userInputs = netInfo.getInputsInfo();
    auto userOutputs = netInfo.getOutputsInfo();

    const auto getTypesWithUserLayout = [](SmallVector<IE::DataInfoOp, 1>& userDataInfo,
                                           ArrayRef<mlir::Type> originTypes, SmallVector<mlir::Type>& newTypes) {
        for (const auto& p : userDataInfo | indexed) {
            const auto ind = checked_cast<uint32_t>(p.index());

            const auto origType = originTypes[ind].cast<mlir::ShapedType>();
            const auto userDimsOrder = p.value().getDimsOrder();

            // FIXME: remove when ND tensors are allowed at runtime [Track number: E#7613]
            constexpr int64_t TARGET_TENSOR_DIM = 4;
            auto dimsOrder = userDimsOrder;
            if (userDimsOrder.numDims() < TARGET_TENSOR_DIM) {
                VPUX_THROW_UNLESS(userDimsOrder == DimsOrder::fromNumDims(userDimsOrder.numDims()),
                                  "Tensors with rank < 4 are supported only with default order, got: {0}",
                                  userDimsOrder);

                dimsOrder = DimsOrder::fromNumDims(TARGET_TENSOR_DIM);
            }

            newTypes[ind] = changeDimsOrder(origType, dimsOrder);
        }
    };

    SmallVector<mlir::Type> newArgTypes(userInputs.size());
    getTypesWithUserLayout(userInputs, funcType.getInputs(), newArgTypes);

    SmallVector<mlir::Type> newResultTypes(userOutputs.size());
    getTypesWithUserLayout(userOutputs, funcType.getResults(), newResultTypes);

    const auto cvtOpBuilder = [](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val,
                                 mlir::Type newType) -> mlir::Operation* {
        const auto order = DimsOrder::fromType(newType.cast<mlir::ShapedType>());
        return builder.create<IE::ReorderOp>(loc, newType, val, order.toPermutationAffineMap(builder.getContext()));
    };

    if (mlir::failed(convertFunc(netFunc, newArgTypes, newResultTypes, cvtOpBuilder, _log))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUseUserLayoutPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createUseUserLayout(Logger log) {
    return std::make_unique<UseUserLayoutPass>(log);
}
