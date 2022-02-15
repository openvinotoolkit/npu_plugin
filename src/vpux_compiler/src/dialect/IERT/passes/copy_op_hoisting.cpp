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

#include "vpux/compiler/dialect/IERT/passes.hpp"

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

namespace {

//
// moveCopyOpToProducer
//

void moveCopyOpToProducer(IERT::CopyOp copyOp, Logger log) {
    auto* inputProducer = copyOp.input().getDefiningOp();

    log.trace("Move the copy operation after its input producer '{0}' at '{1}'", inputProducer->getName(),
              inputProducer->getLoc());
    copyOp->moveAfter(inputProducer);

    auto* outputBufProducer = copyOp.output_buff().getDefiningOp();
    mlir::Operation* newProducerPos = copyOp;

    while (outputBufProducer != nullptr && !outputBufProducer->isBeforeInBlock(copyOp)) {
        log.trace("Move the output buffer producer '{0}' at '{1}'", outputBufProducer->getName(),
                  outputBufProducer->getLoc());

        if (!isBufAllocOp(outputBufProducer) && !IERT::isPureViewOp(outputBufProducer)) {
            VPUX_THROW("Got unsupported output_buf producer '{0}' at '{1}'", outputBufProducer->getName(),
                       outputBufProducer->getLoc());
        }

        outputBufProducer->moveBefore(newProducerPos);
        newProducerPos = outputBufProducer;

        if (auto viewOp = mlir::dyn_cast<mlir::ViewLikeOpInterface>(outputBufProducer)) {
            outputBufProducer = viewOp.getViewSource().getDefiningOp();
        } else {
            outputBufProducer = nullptr;
        }
    }
}

//
// CopyOpHoistingPass
//

class CopyOpHoistingPass final : public IERT::CopyOpHoistingBase<CopyOpHoistingPass> {
public:
    explicit CopyOpHoistingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void CopyOpHoistingPass::safeRunOnFunc() {
    //
    // If there is a CopyOp with source in NNCMX, then we need to move it just after the producing task
    // to allow memory allocator logic free this space as soon as possible.
    //
    // Otherwise NNCMX buffer might occupy its location in unnecessarily long
    // causing potential problems with allocating other buffers due to how dealloc
    // logic and memory allocation pass works.
    //

    const auto needToHoist = [this](IERT::CopyOp copyOp) {
        _log.trace("Check '{0}' operation at '{1}'", copyOp->getName(), copyOp->getLoc());

        const auto sourceMemory = copyOp.input().getType().cast<vpux::NDTypeInterface>().getMemoryKind();

        if (sourceMemory != VPU::MemoryKind::CMX_NN) {
            _log.nest().trace("It doesn't work with CMX_NN input");
            return false;
        }

        if (copyOp.input().getDefiningOp() == nullptr) {
            _log.nest().trace("Its input has no producer operation");
            return false;
        }

        return true;
    };

    getFunction().walk([&](IERT::CopyOp copyOp) {
        _log.trace("Check '{0}' operation at '{1}'", copyOp->getName(), copyOp->getLoc());

        if (needToHoist(copyOp)) {
            moveCopyOpToProducer(copyOp, _log.nest());
        }
    });
}

}  // namespace

//
// createCopyOpHoistingPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createCopyOpHoistingPass(Logger log) {
    return std::make_unique<CopyOpHoistingPass>(log);
}
