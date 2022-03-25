//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/enums.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;
using namespace VPU;

namespace {

//
// CMXConcat
//

class CMXConcatPass final : public CMXConcatBase<CMXConcatPass> {
public:
    explicit CMXConcatPass(Logger log): log(log) {
        log.setName(Base::getArgumentName());
    }

public:
    // Storing Concat use chain consisting of NCE -> (Slice) -> Copy -> Concat
    // or Concat -> (Slice) -> Copy -> NCE
    struct ConcatPart {
        IE::CopyOp copyOp;
        IE::SliceOp sliceOp;
        VPU::NCEOpInterface nceOp;
        size_t inputIdx;

        ConcatPart(IE::CopyOp copy, IE::SliceOp slice, VPU::NCEOpInterface nce, size_t idx)
                : copyOp(copy), sliceOp(slice), nceOp(nce), inputIdx(idx) {
        }
        bool isValidPart() {
            return nceOp != nullptr;
        }
        bool hasSliceOp() {
            return sliceOp != nullptr;
        }
    };

    // Stores multiple Concat parts for a given Concat
    struct ConcatPattern {
        SmallVector<ConcatPart> concatParts;
        IE::ConcatOp concat;
        Logger log;

        ConcatPattern(Logger log): concat(nullptr), log(log) {
        }
        void setConcat(IE::ConcatOp rootConcat) {
            concat = rootConcat;
        }
        void addConcatPart(const ConcatPart& concatPart) {
            concatParts.push_back(concatPart);
        }
        bool isValidPattern() {
            return concat != nullptr;
        }

        size_t getSize(mlir::Value val) const;
        bool concatFitsInCMX(size_t cmxSize);
        bool inputsHaveMultipleUsers() const;
        bool isFusedConcat() const;
        bool inputPatternCanBeCMXed(size_t cmxSize);
        bool childOpsFitInCMX(size_t cmxSize);
        size_t getParallelConsumerCount() const;
        bool outputPatternCanBeCMXed(size_t cmxSize);
    };

private:
    void safeRunOnFunc();

private:
    Logger log;

private:
    bool isSplitSupportedOnDPU(IE::SliceOp sliceOp);
    ConcatPattern getInputPattern(IE::ConcatOp concat);
    ConcatPattern getOutputPattern(IE::ConcatOp concat);
    IE::SliceOp convertCopyToSlice(IE::CopyOp copyOp);
    void rewriteInputPattern(ConcatPattern& concatPattern);
    void rewriteOutputPattern(ConcatPattern& concatPattern);
};

size_t CMXConcatPass::ConcatPattern::getSize(mlir::Value val) const {
    return static_cast<size_t>(getTotalSize(val).count());
}

CMXConcatPass::ConcatPattern CMXConcatPass::getInputPattern(IE::ConcatOp concat) {
    ConcatPattern concatPattern(log);
    // store all required input info in a struct
    for (size_t inputIdx = 0; inputIdx < concat.inputs().size(); inputIdx++) {
        auto input = concat.getOperand(static_cast<int>(inputIdx));
        auto inputCopyOp = input.getDefiningOp<IE::CopyOp>();
        if (inputCopyOp == nullptr) {
            return concatPattern;
        }

        auto parentNCEOp = inputCopyOp.input().getDefiningOp<VPU::NCEOpInterface>();
        if (parentNCEOp == nullptr) {
            return concatPattern;
        }

        concatPattern.addConcatPart(ConcatPart(inputCopyOp, nullptr, parentNCEOp, inputIdx));
    }
    // make valid by adding concat
    concatPattern.setConcat(concat);
    return concatPattern;
}

bool CMXConcatPass::isSplitSupportedOnDPU(IE::SliceOp silceOp) {
    // Check if SubView performs a split along major dimension taking into accout order in memory
    // For NCHW that would be split along C
    // For NHWC that would be split along H
    // Only such cases are supported by DPU IDU becasue only then input to DPU is a contiguous
    // block in memory. Otherwise this behavior needs to be performed by DMA
    const auto inputTypeShape = getShape(silceOp.getOperand()).raw();
    const auto outputType = silceOp.result().getType();

    auto shapedType = outputType.cast<vpux::NDTypeInterface>();
    const auto outputTypeShape = shapedType.getShape().raw();

    if (inputTypeShape.size() != outputTypeShape.size()) {
        return false;
    }

    size_t dimsDifference = 0;
    size_t dimsDifferenceCount = 0;
    const auto order = shapedType.getDimsOrder();

    for (size_t i = 0; i < inputTypeShape.size(); i++) {
        if (inputTypeShape[i] != outputTypeShape[i]) {
            dimsDifference = i;
            dimsDifferenceCount++;
        }
    }

    if (dimsDifferenceCount > 1) {
        return false;
    }

    if (static_cast<int32_t>(dimsDifference) == Dims4D::Act::C.ind() && order == DimsOrder::NCHW) {
        return true;
    }

    if (static_cast<int32_t>(dimsDifference) == Dims4D::Act::H.ind() && order == DimsOrder::NHWC) {
        return true;
    }

    return false;
}

CMXConcatPass::ConcatPattern CMXConcatPass::getOutputPattern(IE::ConcatOp concat) {
    ConcatPattern concatPattern(log);
    // store all required output info in a struct
    for (auto user : concat.output().getUsers()) {
        auto outputSliceOp = mlir::dyn_cast<IE::SliceOp>(user);
        auto outputCopyOp = mlir::dyn_cast<IE::CopyOp>(user);
        if (outputCopyOp == nullptr && outputSliceOp == nullptr) {
            return concatPattern;
        }
        if (outputSliceOp && !isSplitSupportedOnDPU(outputSliceOp)) {
            return concatPattern;
        }
        SmallVector<IE::CopyOp> copyOutOps;
        if (outputSliceOp) {
            for (auto copyOp : outputSliceOp.result().getUsers()) {
                auto childCopyOp = mlir::dyn_cast<IE::CopyOp>(copyOp);
                if (childCopyOp == nullptr) {
                    return concatPattern;
                }
                copyOutOps.push_back(childCopyOp);
            }
        } else {
            copyOutOps.push_back(outputCopyOp);
        }
        for (auto& copyOutOp : copyOutOps) {
            llvm::DenseSet<mlir::Operation*> nceUsers;
            for (auto copyUser : copyOutOp.getResult().getUsers()) {
                auto childNCEOp = mlir::dyn_cast<VPU::NCEOpInterface>(copyUser);
                if (childNCEOp == nullptr) {
                    return concatPattern;
                }
                if (nceUsers.find(copyUser) != nceUsers.end()) {
                    // avoid multiple reads from the same location at the same time
                    log.nest(2).trace("Concat input used twice by the same operation, can not cmx");
                    return concatPattern;
                }
                nceUsers.insert(copyUser);
                concatPattern.addConcatPart(ConcatPart(copyOutOp, outputSliceOp, childNCEOp, 0));
            }
        }
    }
    concatPattern.setConcat(concat);
    return concatPattern;
}

bool CMXConcatPass::ConcatPattern::concatFitsInCMX(size_t cmxSize) {
    // check if the concat can fit in CMX
    // in order to CMX a concat the entire output buffer + inputs for the
    // largest tile must fit in CMX at the same time
    size_t concatSize = getSize(concat.getResult());
    size_t maxUserSize = 0;
    size_t currUserSize = 0;

    // from all users find the one with the largest size
    for (auto concatPart : concatParts) {
        currUserSize = 0;
        // consts (weights table and activation window) already exists
        for (auto input : concatPart.nceOp->getOperands()) {
            currUserSize += getSize(input);
        }
        maxUserSize = std::max<size_t>(maxUserSize, currUserSize);
    }

    log.nest(3).trace("Concat size '{0}'", (concatSize + maxUserSize));
    // return concat size smaller than CMX size
    return (concatSize + maxUserSize) <= cmxSize;
}

bool CMXConcatPass::ConcatPattern::inputsHaveMultipleUsers() const {
    // avoid concats which are complex, where the inputs to the concat are used
    // by other operations
    for (auto concatPart : concatParts) {
        for (auto result : concatPart.nceOp->getResults()) {
            for (auto resultUser : result.getUsers()) {
                if (resultUser != concatPart.copyOp.getOperation()) {
                    // the NCE contains a different user
                    return true;
                }
            }
        }
    }

    return false;
}

bool CMXConcatPass::ConcatPattern::inputPatternCanBeCMXed(size_t cmxSize) {
    // if concat is a Result operation
    for (auto concatOutputUser : concat.output().getUsers()) {
        if (mlir::isa<mlir::ReturnOp>(concatOutputUser)) {
            log.nest(2).trace("Concat output is part of network output");
            return false;
        }
    }

    // assert that the concat will fit in CMX
    if (!concatFitsInCMX(cmxSize)) {
        log.nest(2).trace("Concat does not fit in cmx");
        return false;
    }

    if (isFusedConcat()) {
        log.nest(2).trace("Concat is Fused and will not be cmx-ed");
        return false;
    }

    if (inputsHaveMultipleUsers()) {
        // TODO implement complex concat
        // where part of the concatenated buffer is also used by another operation
        // visible in yolo-v4-tiny concatinate 4
        log.nest(2).trace("Concat is complex");
        return false;
    }

    return true;
}

bool CMXConcatPass::ConcatPattern::childOpsFitInCMX(size_t cmxSize) {
    // check if the child operations - operations using the concat output buffer
    // will fit in CMX along with their inputs and output
    size_t concatSize = getSize(concat.getResult());
    size_t parallelConsumerCount = getParallelConsumerCount();
    size_t maxConsumerSize = 0;

    for (auto& concatPart : concatParts) {
        if (!concatPart.isValidPart()) {
            return false;
        }
        size_t consumerInputSize = 0;
        size_t consumerOutputSize = 0;
        // consts (weights table and activation window) already exists
        for (auto input : concatPart.nceOp->getOperands()) {
            if (input.getDefiningOp() == concatPart.copyOp.getOperation()) {
                continue;
            }
            consumerInputSize += getSize(input);
        }
        for (auto output : concatPart.nceOp->getResults()) {
            consumerOutputSize += getSize(output);
        }

        maxConsumerSize = std::max<size_t>(maxConsumerSize, consumerInputSize + consumerOutputSize);
    }

    if (parallelConsumerCount > 1) {
        // in cases of parallel consumers the graph level differences could be large and the
        // NNCMX buffer could be held for many cycles filling up NNCMX space. To avoid this
        // scenario, ensure that there is space for parallel consumer branches.
        // Note: multiplying by 2 since only 1 compute operation can be live at any given time,
        // during second consumer execution first will be freed and the third can be allocated.
        maxConsumerSize = 2 * maxConsumerSize;
    }

    log.nest(3).trace("Concat consumer max size '{0}'", (maxConsumerSize + concatSize));

    // return concat size greater than CMX size
    return (maxConsumerSize + concatSize) <= cmxSize;
}

size_t CMXConcatPass::ConcatPattern::getParallelConsumerCount() const {
    // number of concat output parts

    log.nest(3).trace("Parallel consumer count '{0}'", concatParts.size());
    return concatParts.size();
}

bool CMXConcatPass::ConcatPattern::isFusedConcat() const {
    // search for concat with producers from both tiling
    // and original operations which are fused
    SmallVector<mlir::Location> locations;
    SmallVector<mlir::Location> fusedLocations;

    for (auto concatPart : concatParts) {
        if (const auto fused = concatPart.nceOp.getLoc().dyn_cast<mlir::FusedLoc>()) {
            auto nceLoc = fused.getLocations().front();
            if (llvm::find(fusedLocations, nceLoc) == fusedLocations.end()) {
                // tiling producers
                fusedLocations.push_back(nceLoc);
            }
        } else {
            auto nceLoc = concatPart.nceOp.getLoc();
            if (llvm::find(locations, nceLoc) == locations.end()) {
                // original producers
                locations.push_back(nceLoc);
            }
        }
    }

    // true if concat produced by both tiling and original ops, or different tiling ops
    return (!fusedLocations.empty() && !locations.empty()) || fusedLocations.size() > 1;
}

bool CMXConcatPass::ConcatPattern::outputPatternCanBeCMXed(size_t cmxSize) {
    // verify the following operation can fit in CMX
    if (!childOpsFitInCMX(cmxSize)) {
        log.nest(2).trace("Concat consumers do not fit in cmx");
        return false;
    }

    return true;
}

void CMXConcatPass::rewriteInputPattern(ConcatPattern& concatPattern) {
    /*
        From DDR IR

        NCE      NCE
         |        |
        Copy     Copy
           \    /
           Concat

        TO NNCMX IR

        NCE      NCE
         |        |
      SubView   SubView (added in IERT)
          \    /
          Concat
    */
    for (auto concatPart : concatPattern.concatParts) {
        log.nest(1).trace("Removing input Copy from NNCMX to DDR '{0}' at '{1}'", concatPart.copyOp->getName(),
                          concatPart.copyOp->getLoc());
        // modify only current concat input as it may have multiple uses
        concatPattern.concat.setOperand(static_cast<int>(concatPart.inputIdx), concatPart.copyOp.input());
    }
}

void CMXConcatPass::rewriteOutputPattern(ConcatPattern& concatPattern) {
    /*
                            From DDR IR

           Concat
           /    \
        Slice   Slice                              Concat
         |        |                                  |
        Copy     Copy                               Copy
         |        |                                  |
        NCE      NCE                                NCE
                            TO NNCMX IR

           Concat                                  Concat
           /    \                                    |
        Slice   Slice                               NCE
         |        |
        NCE      NCE
    */
    for (auto concatPart : concatPattern.concatParts) {
        log.nest(1).trace("Removing output Copy from DDR to NNCMX '{0}' at '{1}'", concatPart.copyOp->getName(),
                          concatPart.copyOp->getLoc());
        // change memory space for concat output
        const auto origType = concatPattern.concat.output().getType().cast<vpux::NDTypeInterface>();
        const auto newType = origType.changeMemSpace(VPU::MemoryKind::CMX_NN);
        concatPattern.concat.output().setType(newType);
        // and for slice op
        if (concatPart.hasSliceOp()) {
            concatPart.sliceOp.source().setType(newType);
            const auto sliceOrigType = concatPart.sliceOp.result().getType().cast<vpux::NDTypeInterface>();
            const auto sliceNewType = sliceOrigType.changeMemSpace(VPU::MemoryKind::CMX_NN);
            concatPart.sliceOp.result().setType(sliceNewType);
        }
        // remove the copy out op
        concatPart.copyOp.output().replaceAllUsesWith(concatPart.copyOp.input());
    }
}

void CMXConcatPass::safeRunOnFunc() {
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto availableMem = IE::getAvailableMemory(module, VPU::MemoryKind::CMX_NN);
    const auto cmxSize = checked_cast<size_t>(availableMem.size().count());

    func->walk([&](IE::ConcatOp concat) {
        // check concat input pattern
        log.trace("Got '{0}' at '{1}'", concat->getName(), concat->getLoc());
        auto inputPattern = getInputPattern(concat);
        if (!inputPattern.isValidPattern()) {
            log.nest(1).trace("Concat input pattern not valid");
            return;
        }
        if (!inputPattern.inputPatternCanBeCMXed(cmxSize)) {
            log.nest(1).trace("Concat input pattern can not be cmx-ed");
            return;
        }
        // check concat output pattern
        auto outputPattern = getOutputPattern(concat);
        if (!outputPattern.isValidPattern()) {
            log.nest(1).trace("Concat output pattern not valid");
            return;
        }
        if (!outputPattern.outputPatternCanBeCMXed(cmxSize)) {
            log.nest(1).trace("Concat output pattern can not be cmx-ed");
            return;
        }
        // rewrite from DDR to NNCMX
        rewriteInputPattern(inputPattern);
        rewriteOutputPattern(outputPattern);
        log.trace("Concat '{0}' at '{1}' was cmx-ed", concat->getName(), concat->getLoc());
    });
}

}  // namespace

//
// createCMXConcatPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createCMXConcatPass(Logger log) {
    return std::make_unique<CMXConcatPass>(log);
}
