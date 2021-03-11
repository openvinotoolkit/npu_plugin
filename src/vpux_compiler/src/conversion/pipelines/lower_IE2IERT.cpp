//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"

#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

namespace {

//
// LowerIE2IERTPass
//

class LowerIE2IERTPass final : public LowerIE2IERTBase<LowerIE2IERTPass> {
public:
    explicit LowerIE2IERTPass(Logger log);

public:
    void runOnOperation() final;

private:
    void passBody();

private:
    Logger _log;
    mlir::OpPassManager _pm;
};

LowerIE2IERTPass::LowerIE2IERTPass(Logger log)
        : _log(log), _pm(mlir::ModuleOp::getOperationName(), mlir::OpPassManager::Nesting::Implicit) {
    _log.setName(Base::getArgumentName());

    _pm.addPass(createConvertIE2IERTPass(_log.nest()));
    _pm.addPass(mlir::createFuncBufferizePass());
    _pm.addPass(mlir::createBufferResultsToOutParamsPass());
    _pm.addPass(mlir::createFinalizingBufferizePass());
    _pm.addPass(mlir::createBufferDeallocationPass());
    _pm.addPass(mlir::createCopyRemovalPass());
}

void LowerIE2IERTPass::runOnOperation() {
    try {
        passBody();
    } catch (const std::exception& e) {
        (void)errorAt(getOperation(), "{0} Pass failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

//
// passBody
//

void LowerIE2IERTPass::passBody() {
    auto module = getOperation();
    if (mlir::failed(runPipeline(_pm, module))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createLowerIE2IERTPass
//

std::unique_ptr<mlir::Pass> vpux::createLowerIE2IERTPass(Logger log) {
    return std::make_unique<LowerIE2IERTPass>(log);
}
