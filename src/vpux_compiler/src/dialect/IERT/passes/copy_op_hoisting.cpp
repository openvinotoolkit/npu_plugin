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
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"

#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"

#include "vpux/compiler/utils/logging.hpp"

#include "vpux/utils/core/range.hpp"

using namespace vpux;

namespace {

// If there is a CopyOp with source in NNCMX, then in IR move it just after NCE task
// to allow memory allocator logic to free this space as soon as possible.
// Otherwise NCE result might occupy its location in NNCMX unnecessarily long
// causing potential problems with allocating other buffers due to how dealloc
// logic and memory allocation pass works.
// Below function will also update allocator or pureView op defininig the output
// buffer of copy op result
void NnCmxCopyOpHoisting(mlir::FuncOp func) {
    func.walk([&](IERT::CopyOp copyOp) {
        auto sourceMemory = VPUIP::getPhysicalMemory(copyOp.input().getType().cast<mlir::MemRefType>());
        if (mlir::failed(sourceMemory)) {
            return;
        }

        if (sourceMemory.getValue() != VPUIP::PhysicalMemory::CMX_NN) {
            return;
        }

        // Check if outbuf buffer is defined by AllocOp or SubView as following patterns
        // are supported only
        //  - AllocOp -> CopyOp.output_buff
        //  - AllocOp -> SubView -> CopyOp.output_buff
        // If not then skip
        auto outputBufOp = copyOp.output_buff().getDefiningOp();
        if (!mlir::isa<mlir::memref::AllocOp, IERT::SubViewOp>(outputBufOp)) {
            return;
        }
        auto subViewOp = mlir::dyn_cast_or_null<IERT::SubViewOp>(outputBufOp);
        if (subViewOp && !mlir::isa<mlir::memref::AllocOp>(subViewOp.source().getDefiningOp())) {
            return;
        }

        copyOp->moveAfter(copyOp.input().getDefiningOp());

        // If op defining output buffer (AllocOp/SubViewOp) are already before then
        // no need to change its location in the Block
        if (outputBufOp->isBeforeInBlock(copyOp)) {
            return;
        }
        outputBufOp->moveBefore(copyOp);

        // In case of SubViewOp check top buffer and relocate if needed
        if (subViewOp) {
            auto subViewSourceOp = subViewOp.source().getDefiningOp();
            if (subViewSourceOp->isBeforeInBlock(subViewOp)) {
                return;
            }

            subViewSourceOp->moveBefore(outputBufOp);
        }
    });
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

//
// safeRunOnFunc
//

void CopyOpHoistingPass::safeRunOnFunc() {
    NnCmxCopyOpHoisting(getFunction());
}

}  // namespace

//
// createCopyOpHoistingPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createCopyOpHoistingPass(Logger log) {
    return std::make_unique<CopyOpHoistingPass>(log);
}
