//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"

using namespace vpux;

namespace {

//
// ConvertFuncArgsToDeclarationsPass
//

class ConvertFuncArgsToDeclarationsPass final :
        public VPUIP::ConvertFuncArgsToDeclarationsBase<ConvertFuncArgsToDeclarationsPass> {
public:
    explicit ConvertFuncArgsToDeclarationsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertFuncArgsToDeclarationsPass::safeRunOnFunc() {
    auto netFunc = getOperation();

    VPUX_THROW_UNLESS(netFunc.getNumArguments() >= netFunc.getNumResults(), "Function '{0}' is not bufferized",
                      netFunc);
    const auto numInputs = netFunc.getNumArguments() - netFunc.getNumResults();

    auto returnOp = *netFunc.getOps<mlir::func::ReturnOp>().begin();
    auto& firstOp = *netFunc.getOps().begin();

    const auto replaceUse = [&](mlir::ValueRange args, VPURT::BufferSection section) {
        for (auto p : args | indexed) {
            auto val = p.value();

            if (val.getUses().empty()) {
                continue;
            }

            OpBuilderLogger builderLog(_log.nest(2));
            mlir::OpBuilder argBuilder(&firstOp, &builderLog);
            argBuilder.setInsertionPoint(&firstOp);

            auto declareBuffOp =
                    argBuilder.create<VPURT::DeclareBufferOp>(firstOp.getLoc(), val.getType(), section, p.index(), 0);

            _log.trace("Replace all uses of '{0}' with '{1}'", declareBuffOp);
            val.replaceAllUsesExcept(declareBuffOp.getResult(), llvm::SmallPtrSet<mlir::Operation*, 1>{returnOp});
        }
    };

    replaceUse(netFunc.getArguments().take_front(numInputs), VPURT::BufferSection::NetworkInput);
    replaceUse(netFunc.getArguments().drop_front(numInputs), VPURT::BufferSection::NetworkOutput);
}

}  // namespace

//
// createConvertFuncArgsToDeclarationsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertFuncArgsToDeclarationsPass(Logger log) {
    return std::make_unique<ConvertFuncArgsToDeclarationsPass>(log);
}
