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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/matmul_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux {
namespace IE {
//
// ConvertMatMulPatternToDWConvPass
//

class ConvertMatMulPatternToDWConvPass final :
        public IE::ConvertMatMulPatternToDWConvBase<ConvertMatMulPatternToDWConvPass> {
public:
    explicit ConvertMatMulPatternToDWConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertMatMulPatternToDWConvPass::safeRunOnFunc() {
    getFunction().walk([&](IE::MatMulOp origOp) {
        _log.trace("Check '{0}' operation at '{1}'", origOp->getName(), origOp->getLoc());
        std::cout << llvm::formatv("Check '{0}' operation at '{1}'", origOp->getName(), origOp->getLoc()).str()
                  << std::endl;
        if (checkPermuteMatMulPattern(origOp)) {
            std::cout << "match!" << std::endl;
            //            convertMatMulPatternToDWConv(origOp);
        }
    });
}

}  // namespace IE
}  // namespace vpux

//
// createConvertMatMulPatternToDWConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertMatMulPatternToDWConvPass(Logger log) {
    return std::make_unique<ConvertMatMulPatternToDWConvPass>(log);
}
