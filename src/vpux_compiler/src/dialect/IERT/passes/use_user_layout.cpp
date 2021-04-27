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

#include <vpux/compiler/core/aliases_info.hpp>
#include <vpux/compiler/core/attributes/stride_reqs.hpp>
#include <vpux/compiler/dialect/IERT/passes.hpp>
#include <vpux/compiler/utils/logging.hpp>
#include <vpux/compiler/utils/rewriter.hpp>

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// UseUserLayoutPass
//

class UseUserLayoutPass final : public IERT::UseUserLayoutBase<UseUserLayoutPass> {
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

            const auto origType = originTypes[ind].cast<mlir::MemRefType>();
            const auto userType = p.value().userType().cast<mlir::MemRefType>();

            const auto newType = mlir::MemRefType::get(origType.getShape(), origType.getElementType(),
                                                       userType.getAffineMaps(), origType.getMemorySpace());
            newTypes[ind] = newType;
        }
    };

    SmallVector<mlir::Type> newArgTypes(userInputs.size());
    getTypesWithUserLayout(userInputs, funcType.getInputs(), newArgTypes);

    SmallVector<mlir::Type> newResultTypes(userOutputs.size());
    getTypesWithUserLayout(userOutputs, funcType.getResults(), newResultTypes);

    const auto cvtOpBuilder = [](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value inVal,
                                 mlir::Value outVal) -> mlir::Operation* {
        return builder.create<IERT::ReorderOp>(loc, inVal, outVal);
    };

    if (mlir::failed(convertBufferizedFunc(netFunc, newArgTypes, newResultTypes, cvtOpBuilder, _log))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUseUserLayoutPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createUseUserLayout(Logger log) {
    return std::make_unique<UseUserLayoutPass>(log);
}
