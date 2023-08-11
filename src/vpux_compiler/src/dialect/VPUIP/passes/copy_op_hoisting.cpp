//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <queue>

using namespace vpux;

namespace {

//
// moveCopyOpToProducer
//

void moveCopyOpToProducer(VPUIP::CopyOp copyOp, Logger log) {
    auto* inputProducer = copyOp.input().getDefiningOp();

    log.trace("Move the copy operation after its input producer '{0}' at '{1}'", inputProducer->getName(),
              inputProducer->getLoc());
    copyOp->moveAfter(inputProducer);

    auto* outputBufProducer = copyOp.output_buff().getDefiningOp();
    mlir::Operation* newProducerPos = copyOp;
    // Queue of elements of format <bufferProducer, insertionPoint>
    std::queue<std::tuple<mlir::Operation*, mlir::Operation*>> opsToMoveQueue;
    opsToMoveQueue.push(std::make_tuple(outputBufProducer, newProducerPos));

    while (!opsToMoveQueue.empty()) {
        std::tie(outputBufProducer, newProducerPos) = opsToMoveQueue.front();
        opsToMoveQueue.pop();

        if (!outputBufProducer || outputBufProducer->isBeforeInBlock(copyOp)) {
            continue;
        }
        log.trace("Move the output buffer producer '{0}' at '{1}'", outputBufProducer->getName(),
                  outputBufProducer->getLoc());

        if (!isBufAllocOp(outputBufProducer) && !VPUIP::isPureViewOp(outputBufProducer)) {
            VPUX_THROW("Got unsupported output_buf producer '{0}' at '{1}'", outputBufProducer->getName(),
                       outputBufProducer->getLoc());
        }

        outputBufProducer->moveBefore(newProducerPos);
        newProducerPos = outputBufProducer;

        if (auto viewOp = mlir::dyn_cast<mlir::ViewLikeOpInterface>(outputBufProducer)) {
            opsToMoveQueue.push(std::make_tuple(viewOp.getViewSource().getDefiningOp(), newProducerPos));
        } else if (auto groupedViewOp = mlir::dyn_cast<vpux::GroupedViewOpInterface>(outputBufProducer)) {
            for (auto operand : groupedViewOp.getViewSources()) {
                opsToMoveQueue.push(std::make_tuple(operand.getDefiningOp(), newProducerPos));
            }
        }
    }
}

//
// CopyOpHoistingPass
//

class CopyOpHoistingPass final : public VPUIP::CopyOpHoistingBase<CopyOpHoistingPass> {
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

    const auto needToHoist = [this](VPUIP::CopyOp copyOp) {
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

    getOperation().walk([&](VPUIP::CopyOp copyOp) {
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

std::unique_ptr<mlir::Pass> vpux::VPUIP::createCopyOpHoistingPass(Logger log) {
    return std::make_unique<CopyOpHoistingPass>(log);
}
