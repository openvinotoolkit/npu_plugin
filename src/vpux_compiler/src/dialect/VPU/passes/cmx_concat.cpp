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
    explicit CMXConcatPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    // Storing Concat use chain consisting of NCE -> (Slice) -> Copy -> Concat
    // or Concat -> (Slice) -> Copy -> NCE
    struct ConcatPart {
        IE::CopyOp copyOp;
        VPU::NCEClusterTilingOp copyClusterOp;
        IE::SliceOp sliceOp;
        VPU::NCEOpInterface nceOp;
        VPU::NCEClusterTilingOp nceClusterOp;
        size_t inputIdx;

        ConcatPart(IE::CopyOp copy, IE::SliceOp slice, VPU::NCEOpInterface nce, size_t idx)
                : copyOp(copy),
                  copyClusterOp(nullptr),
                  sliceOp(slice),
                  nceOp(nce),
                  nceClusterOp(nullptr),
                  inputIdx(idx) {
        }

        ConcatPart(IE::CopyOp copy, VPU::NCEClusterTilingOp copyCluster, IE::SliceOp slice, VPU::NCEOpInterface nce,
                   VPU::NCEClusterTilingOp nceCluster, size_t idx)
                : copyOp(copy),
                  copyClusterOp(copyCluster),
                  sliceOp(slice),
                  nceOp(nce),
                  nceClusterOp(nceCluster),
                  inputIdx(idx) {
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
        Logger _log;

        ConcatPattern(Logger log): concat(nullptr), _log(log) {
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
        bool inputPatternCanBeCMXed(size_t cmxSize);
        bool childOpsFitInCMX(size_t cmxSize);
        size_t getParallelConsumerCount() const;
        bool outputPatternCanBeCMXed(size_t cmxSize);
    };

private:
    void safeRunOnFunc();

private:
    Logger _log;

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
    ConcatPattern concatPattern(_log);
    // store all required input info in a struct
    for (size_t inputIdx = 0; inputIdx < concat.inputs().size(); inputIdx++) {
        auto input = concat.getOperand(static_cast<int>(inputIdx));

        // auto inputCopyOp = input.getDefiningOp<IE::CopyOp>();
        IE::CopyOp inputCopyOp;
        VPU::NCEClusterTilingOp inputClusterTilingOp = input.getDefiningOp<VPU::NCEClusterTilingOp>();

        if (inputClusterTilingOp) {
            // auto bodyBlock = &spillWriteExecOp.body().front();
            // auto& bodyBlock = inputClusterTilingOp.body().front();
            // auto& innerOp = bodyBlock.front();
            auto& innerOp = inputClusterTilingOp.body().front().front();
            if (mlir::isa<IE::CopyOp>(innerOp)) {
                inputCopyOp = mlir::dyn_cast<IE::CopyOp>(innerOp);
            }
        } else {
            inputCopyOp = input.getDefiningOp<IE::CopyOp>();
        }

        if (inputCopyOp == nullptr) {
            return concatPattern;
        }

        // auto parentNCEOp = inputCopyOp.input().getDefiningOp<VPU::NCEOpInterface>();
        VPU::NCEOpInterface parentNCEOp;
        VPU::NCEClusterTilingOp parentNCEClusterTilingOp;

        if (inputClusterTilingOp) {
            auto inputClusterTiling = inputClusterTilingOp.getOperand(0);
            if ((parentNCEClusterTilingOp = inputClusterTiling.getDefiningOp<VPU::NCEClusterTilingOp>())) {
                auto& innerOp = parentNCEClusterTilingOp.body().front().front();
                if (mlir::isa<VPU::NCEOpInterface>(innerOp)) {
                    parentNCEOp = mlir::dyn_cast<VPU::NCEOpInterface>(innerOp);
                }
            }
        } else {
            parentNCEOp = inputCopyOp.input().getDefiningOp<VPU::NCEOpInterface>();
        }

        if (parentNCEOp == nullptr) {
            return concatPattern;
        }

        concatPattern.addConcatPart(ConcatPart(inputCopyOp, inputClusterTilingOp, nullptr, parentNCEOp,
                                               parentNCEClusterTilingOp, inputIdx));
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

    if (dimsDifference == 1 && order == DimsOrder::NCHW) {
        return true;
    }

    if (dimsDifference == 2 && order == DimsOrder::NHWC) {
        return true;
    }

    return false;
}

CMXConcatPass::ConcatPattern CMXConcatPass::getOutputPattern(IE::ConcatOp concat) {
    ConcatPattern concatPattern(_log);
    // store all required output info in a struct
    for (auto user : concat.output().getUsers()) {
        auto outputSliceOp = mlir::dyn_cast<IE::SliceOp>(user);
        auto outputCopyOp = mlir::dyn_cast<IE::CopyOp>(user);
        auto outputClusterCopyOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(user);

        if (outputClusterCopyOp) {
            auto& innerOp = outputClusterCopyOp.body().front().front();
            if (mlir::isa<IE::CopyOp>(innerOp)) {
                outputCopyOp = mlir::dyn_cast<IE::CopyOp>(innerOp);
            }
        }

        if (outputCopyOp == nullptr && outputSliceOp == nullptr) {
            return concatPattern;
        }

        if (outputSliceOp && !isSplitSupportedOnDPU(outputSliceOp)) {
            return concatPattern;
        }
        // SmallVector<IE::CopyOp> copyOutOps;
        SmallVector<mlir::Operation*> copyOutOps;
        if (outputSliceOp) {
            // TODO: Update for NCEClsuterTiling
            for (auto copyOp : outputSliceOp.result().getUsers()) {
                auto childCopyOp = mlir::dyn_cast<IE::CopyOp>(copyOp);

                auto outputClusterCopyOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(copyOp);

                if (outputClusterCopyOp) {
                    auto& innerOp = outputClusterCopyOp.body().front().front();
                    if (mlir::isa<IE::CopyOp>(innerOp)) {
                        childCopyOp = mlir::dyn_cast<IE::CopyOp>(innerOp);
                    }
                }

                if (childCopyOp == nullptr) {
                    return concatPattern;
                }
                copyOutOps.push_back(childCopyOp);
            }
        } else {
            if (outputClusterCopyOp) {
                copyOutOps.push_back(outputClusterCopyOp);
            } else {
                copyOutOps.push_back(outputCopyOp);
            }
        }
        for (auto& op : copyOutOps) {
            llvm::DenseSet<mlir::Operation*> nceUsers;
            for (auto opUser : op->getResult(0).getUsers()) {
                auto childNCEOp = mlir::dyn_cast<VPU::NCEOpInterface>(opUser);
                auto childNCEClusterOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(opUser);

                if (childNCEClusterOp) {
                    auto& innerOp = childNCEClusterOp.body().front().front();
                    if (mlir::isa<VPU::NCEOpInterface>(innerOp)) {
                        childNCEOp = mlir::dyn_cast<VPU::NCEOpInterface>(innerOp);
                    }
                }

                if (childNCEOp == nullptr) {
                    return concatPattern;
                }

                if (nceUsers.find(opUser) != nceUsers.end()) {
                    // avoid multiple reads from the same location at the same time
                    _log.nest(2).trace("Concat input used twice by the same operation, can not cmx");
                    return concatPattern;
                }
                nceUsers.insert(opUser);
                if (childNCEClusterOp) {
                    // TODO: fix how copyOp is added
                    IE::CopyOp copyOp;
                    VPU::NCEClusterTilingOp copyClusterOp;
                    if (mlir::isa<VPU::NCEClusterTilingOp>(op)) {
                        copyClusterOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(op);
                        auto& innerOp = copyClusterOp.body().front().front();
                        if (mlir::isa<IE::CopyOp>(innerOp)) {
                            copyOp = mlir::dyn_cast<IE::CopyOp>(innerOp);
                        }

                    } else if (mlir::isa<IE::CopyOp>(op)) {
                        copyOp = mlir::dyn_cast<IE::CopyOp>(op);
                    }

                    if (copyOp == nullptr) {
                        return concatPattern;
                    }

                    concatPattern.addConcatPart(
                            ConcatPart(copyOp, copyClusterOp, outputSliceOp, childNCEOp, childNCEClusterOp, 0));
                } else {
                    concatPattern.addConcatPart(
                            ConcatPart(mlir::dyn_cast<IE::CopyOp>(op), outputSliceOp, childNCEOp, 0));
                }
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

    _log.nest(3).trace("Concat size '{0}'", (concatSize + maxUserSize));
    // return concat size smaller than CMX size
    return (concatSize + maxUserSize) <= cmxSize;
}

bool CMXConcatPass::ConcatPattern::inputsHaveMultipleUsers() const {
    // avoid concats which are complex, where the inputs to the concat are used
    // by other operations
    for (auto concatPart : concatParts) {
        auto nceOp = concatPart.nceOp.getOperation();
        if (concatPart.nceClusterOp) {
            nceOp = concatPart.nceClusterOp.getOperation();
        }
        for (auto result : nceOp->getResults()) {
            for (auto resultUser : result.getUsers()) {
                auto copyOp = concatPart.copyOp.getOperation();
                if (concatPart.nceClusterOp) {
                    copyOp = concatPart.copyClusterOp.getOperation();
                }

                if (resultUser != copyOp) {
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
            _log.nest(2).trace("Concat output is part of network output");
            return false;
        }
    }

    // assert that the concat will fit in CMX

    if (!concatFitsInCMX(cmxSize)) {
        _log.nest(2).trace("Concat does not fit in cmx");
        return false;
    }

    if (inputsHaveMultipleUsers()) {
        // TODO implement complex concat
        // where part of the concatenated buffer is also used by another operation
        // visible in yolo-v4-tiny concatinate 4
        _log.nest(2).trace("Concat is complex");
        return false;
    }

    return true;
}

bool CMXConcatPass::ConcatPattern::childOpsFitInCMX(size_t cmxSize) {
    // check if the child operations - operations using the concat output buffer
    // will fit in CMX along with their inputs and output
    // auto output = concat.getResult();
    size_t concatSize = getSize(concat.getResult());
    size_t parallelConsumerCount = getParallelConsumerCount();
    size_t maxConsumerSize = 0;

    for (auto& concatPart : concatParts) {
        if (!concatPart.isValidPart()) {
            return false;
        }
        size_t currentConsumerSize = 0;
        // consts (weights table and activation window) already exists
        auto nceOp = concatPart.nceOp.getOperation();
        if (concatPart.nceClusterOp) {
            nceOp = concatPart.nceClusterOp.getOperation();
        }
        auto copyOp = concatPart.copyOp.getOperation();
        if (concatPart.copyClusterOp) {
            copyOp = concatPart.copyClusterOp.getOperation();
        }
        for (auto input : nceOp->getOperands()) {
            if (input.getDefiningOp() == copyOp) {
                continue;
            }
            currentConsumerSize += getSize(input);
        }
        for (auto output : nceOp->getResults()) {
            currentConsumerSize += getSize(output);
        }
        maxConsumerSize = std::max<size_t>(maxConsumerSize, currentConsumerSize);
    }
    // in cases of parallel consumers the graph level differences could be large and the
    // NNCMX buffer could be held for many cycles filling up NNCMX space. To avoid this
    // scenario, ensure that there is space for parallel consumer branches
    // assume longest branch is equal to number of parallel consumers
    _log.nest(3).trace("Concat consumer max size '{0}'", (parallelConsumerCount * (maxConsumerSize + concatSize)));
    // return concat size greater than CMX size
    return (parallelConsumerCount * (maxConsumerSize + concatSize)) <= cmxSize;
}

size_t CMXConcatPass::ConcatPattern::getParallelConsumerCount() const {
    // tiling operations are considered a single consumer
    SmallVector<mlir::Location> locations;
    size_t parallelConsumerCount = 0;

    for (auto concatPart : concatParts) {
        // for fused loc ignore tiling details

        auto nceOp = concatPart.nceOp.getOperation();
        if (concatPart.nceClusterOp) {
            nceOp = concatPart.nceClusterOp.getOperation();
        }

        if (const auto fused = nceOp->getLoc().dyn_cast<mlir::FusedLoc>()) {
            auto nceLoc = fused.getLocations().front();
            if (llvm::find(locations, nceLoc) == locations.end()) {
                ++parallelConsumerCount;
                locations.push_back(nceLoc);
            }
        } else {
            auto nceLoc = nceOp->getLoc();
            if (llvm::find(locations, nceLoc) == locations.end()) {
                ++parallelConsumerCount;
                locations.push_back(nceLoc);
            }
        }
    }

    _log.nest(3).trace("Parallel consumer count '{0}'", parallelConsumerCount);
    return parallelConsumerCount;
}

bool CMXConcatPass::ConcatPattern::outputPatternCanBeCMXed(size_t cmxSize) {
    // verify the following operation can fit in CMX
    if (!childOpsFitInCMX(cmxSize)) {
        _log.nest(2).trace("Concat consumers do not fit in cmx");
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
        _log.nest(1).trace("Removing input Copy from NNCMX to DDR '{0}' at '{1}'", concatPart.copyOp->getName(),
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
        _log.nest(1).trace("Removing output Copy from DDR to NNCMX '{0}' at '{1}'", concatPart.copyOp->getName(),
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
        _log.trace("Got '{0}' at '{1}'", concat->getName(), concat->getLoc());
        auto inputPattern = getInputPattern(concat);
        if (!inputPattern.isValidPattern()) {
            _log.nest(1).trace("Concat input pattern not valid");
            return;
        }
        _log.nest(1).trace("Concat input pattern valid");

        if (!inputPattern.inputPatternCanBeCMXed(cmxSize)) {
            _log.nest(1).trace("Concat input pattern can not be cmx-ed");
            return;
        }
        _log.nest(1).trace("Concat input pattern can be cmx-ed");
        // check concat output pattern
        auto outputPattern = getOutputPattern(concat);
        if (!outputPattern.isValidPattern()) {
            _log.nest(1).trace("Concat output pattern not valid");
            return;
        }
        _log.nest(1).trace("Concat output pattern valid");
        if (!outputPattern.outputPatternCanBeCMXed(cmxSize)) {
            _log.nest(1).trace("Concat output pattern can not be cmx-ed");
            return;
        }
        _log.nest(1).trace("Concat output pattern can be cmx-ed");
        // rewrite from DDR to NNCMX

        _log.nest(1).trace("Concat rewriteInputPattern");
        rewriteInputPattern(inputPattern);
        _log.nest(1).trace("Concat rewriteOutputPattern");
        rewriteOutputPattern(outputPattern);
        _log.trace("Concat '{0}' at '{1}' was cmx-ed", concat->getName(), concat->getLoc());
    });
}

}  // namespace

//
// createCMXConcatPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createCMXConcatPass(Logger log) {
    return std::make_unique<CMXConcatPass>(log);
}
