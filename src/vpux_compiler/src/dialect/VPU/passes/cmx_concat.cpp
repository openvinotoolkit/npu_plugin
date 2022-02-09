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
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPURT/types.hpp"

#include "vpux/compiler/core/layers.hpp"

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
    struct ConcatPart {
        ConcatPart() {
            _copyOp = nullptr;
            _nceOp = nullptr;
        }
        ConcatPart(IE::CopyOp copyOp, VPU::NCEOpInterface nceOp) {
            _copyOp = copyOp;
            _nceOp = nceOp;
        }
        bool isValidPart() {
            return _nceOp != nullptr;
        }
        bool hasCopyOp() {
            return _copyOp != nullptr;
        }
        IE::CopyOp _copyOp;
        VPU::NCEOpInterface _nceOp;
    };

    struct ConcatPattern {
        ConcatPattern() {
            _concat = nullptr;
        }
        void setConcat(IE::ConcatOp concat) {
            _concat = concat;
        }
        void addConcatPart(ConcatPart concatPart) {
            _concatParts.push_back(concatPart);
        }
        bool isValidPatern() {
            return _concat != nullptr;
        }
        SmallVector<ConcatPart> _concatParts;
        IE::ConcatOp _concat;
    };

private:
    void safeRunOnFunc();

private:
    Logger _log;

private:
    size_t getSize(mlir::Value val);
    ConcatPattern getInputPattern(IE::ConcatOp concat);
    bool isSplitSupportedOnDPU(IE::SliceOp sliceOp);
    ConcatPattern getOutputPattern(IE::ConcatOp concat);
    bool concatOperationDoesNotFitInCMX(IE::ConcatOp concat, ConcatPattern concatPattern, size_t cmxSize);
    bool isThisAComplexConcat(IE::ConcatOp concat, ConcatPattern concatPattern);
    bool inputPatternCanBeCMXed(IE::ConcatOp concat, ConcatPattern concatPattern, size_t cmxSize);
    bool childOperationsDoNotFitInCMX(IE::ConcatOp concat, ConcatPattern concatPattern, size_t cmxSize);
    bool outputPatternCanBeCMXed(IE::ConcatOp concat, ConcatPattern concatPattern, size_t cmxSize);
    IE::SliceOp convertCopyToSlice(IE::CopyOp copyOp);
    void rewriteInputPattern(IE::ConcatOp concat);
    void rewriteOutputPattern(IE::ConcatOp concat);
};

size_t CMXConcatPass::getSize(mlir::Value val) {
    const auto valueShape = getShape(val).raw();
    const auto totalSize = vpux::details::calcTotalShapeSize(valueShape);
    return static_cast<size_t>(totalSize);
}

CMXConcatPass::ConcatPattern CMXConcatPass::getInputPattern(IE::ConcatOp concat) {
    ConcatPattern concatPattern;
    // store all required input info in a struct
    for (const auto& input : concat.inputs()) {
        auto inputCopyOp = input.getDefiningOp<IE::CopyOp>();
        if (inputCopyOp == nullptr) {
            return concatPattern;
        }

        auto parentNCEOp = inputCopyOp.input().getDefiningOp<VPU::NCEOpInterface>();
        if (parentNCEOp == nullptr) {
            return concatPattern;
        }

        concatPattern.addConcatPart(ConcatPart(inputCopyOp, parentNCEOp));
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

    auto shapedType = outputType.dyn_cast<vpux::ShapedPropertiesTypeInterface>();
    VPUX_THROW_WHEN(shapedType == nullptr, "Got non shaped type '{0}'", outputType);
    const auto outputTypeShape = shapedType.getShape().raw();

    if (inputTypeShape.size() != outputTypeShape.size() || inputTypeShape.size() != 4) {
        return false;
    }

    size_t dimsDifference = 0;
    size_t dimsDifferenceCount = 0;
    const auto order = shapedType.getDimsOrder();

    for (size_t i = 0; i < 4; i++) {
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
    ConcatPattern concatPattern;
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
        for (auto subUser : user->getUsers()) {
            auto childNCEOpWithSlice = mlir::dyn_cast<VPU::NCEOpInterface>(subUser);
            if (childNCEOpWithSlice == nullptr) {
                return concatPattern;
            }
            concatPattern.addConcatPart(ConcatPart(outputCopyOp, childNCEOpWithSlice));
        }
    }
    concatPattern.setConcat(concat);
    return concatPattern;
}

bool CMXConcatPass::concatOperationDoesNotFitInCMX(IE::ConcatOp concat, ConcatPattern concatPattern, size_t cmxSize) {
    // check if the concat can fit in CMX
    // in order to CMX a concat the entire output buffer + inputs for the
    // largest tile must fit in CMX at the same time
    size_t concatSize = getSize(concat.getResult());
    size_t maxUserSize = 0;
    size_t currUserSize = 0;

    // from all users find the one with the largest size
    for (auto concatPart : concatPattern._concatParts) {
        currUserSize = 0;
        for (auto input : concatPart._nceOp->getOperands()) {
            currUserSize += getSize(input);
        }
        // subtract output as reference in inputs
        for (auto output : concatPart._nceOp->getResults()) {
            currUserSize -= getSize(output);
        }
        maxUserSize = std::max<size_t>(maxUserSize, currUserSize);
    }

    // return concat size smaller than CMX size
    return (concatSize + maxUserSize) > cmxSize;
}

bool CMXConcatPass::isThisAComplexConcat(IE::ConcatOp concat, ConcatPattern concatPattern) {
    // avoid concats which are complex, where the inputs to the concat are used
    // by other operations
    for (auto concatPart : concatPattern._concatParts) {
        for (auto result : concatPart._nceOp->getResults()) {
            for (auto resultUser : result.getUsers()) {
                if (concatPart.hasCopyOp()) {
                    if (resultUser != concatPart._copyOp.getOperation()) {
                        // the NCE contains a different user
                        return true;
                    }
                } else {
                    if (resultUser != concat.getOperation()) {
                        // the NCE contains a different user
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

bool CMXConcatPass::inputPatternCanBeCMXed(IE::ConcatOp concat, ConcatPattern concatPattern, size_t cmxSize) {
    // if concat is a Result operation
    if (concat.output().isa<mlir::BlockArgument>()) {
        return false;
    }

    // assert that the concat will fit in CMX
    if (concatOperationDoesNotFitInCMX(concat, concatPattern, cmxSize)) {
        return false;
    }

    if (isThisAComplexConcat(concat, concatPattern)) {
        // TODO implement complex concat
        // where part of the concatinated buffer is also used by another operation
        // visible in yolo-v4-tiny concatinate 4
        return false;
    }

    return true;
}

bool CMXConcatPass::childOperationsDoNotFitInCMX(IE::ConcatOp concat, ConcatPattern concatPattern, size_t cmxSize) {
    // check if the child operations - operations using the concat output buffer
    // will fit in CMX along with their inputs and output
    // auto output = concat.getResult();
    size_t concatSize = getSize(concat.getResult());
    size_t maxConsumerSize = 0;

    for (auto& concatPart : concatPattern._concatParts) {
        if (!concatPart.isValidPart()) {
            return false;
        }

        size_t currentConsumerSize = 0;
        for (auto input : concatPart._nceOp->getOperands()) {
            if (input.getDefiningOp() == concatPart._copyOp.getOperation()) {
                continue;
            }
            currentConsumerSize += getSize(input);
        }
        for (auto output : concatPart._nceOp->getResults()) {
            currentConsumerSize += getSize(output);
        }

        maxConsumerSize = std::max<size_t>(maxConsumerSize, currentConsumerSize);
    }

    // return concat size greater than CMX size
    return (maxConsumerSize + concatSize) > cmxSize;
}

bool CMXConcatPass::outputPatternCanBeCMXed(IE::ConcatOp concat, ConcatPattern concatPattern, size_t cmxSize) {
    // verify the following operation can fit in CMX
    if (childOperationsDoNotFitInCMX(concat, concatPattern, cmxSize)) {
        return false;
    }

    return true;
}

IE::SliceOp CMXConcatPass::convertCopyToSlice(IE::CopyOp copyOp) {
    // TODO
}

void CMXConcatPass::rewriteInputPattern(ConcatPattern concatPattern) {
    // From DDR IR

    // NCE      NCE
    //  |        |
    // Copy     Copy
    //    \    /
    //    Concat

    // TO NNCMX IR

    // NCE      NCE
    //  |        |
    // Slice   Slice
    //    \    /
    //    Concat

    // convert using the above 
    for (auto concatPart : concatPattern._concatParts) {
        concatPart._copyOp.replaceAllUsesWith(convertCopyToSlice(concatPart._copyOp));
        concatPart._copyOp.erase();
    }
}

void CMXConcatPass::rewriteOutputPattern(ConcatPattern concatPattern) {
    /// From DDR IR

    //    Concat
    //    /    \ 
    // Copy     Copy
    //  |        |
    // NCE      NCE

    // TO NNCMX IR

    //    Concat
    //    /    \ 
    // Slice   Slice
    //  |        |
    // NCE      NCE

    // OR
    //    Concat
    //      |
    //     Copy 
    //      |
    //     NCE

    // TO

    //    Concat
    //      |
    //     NCE

    // convert using the above 
    if (concatPattern._concatParts.size() == 1) {
        concatPattern._concatParts.front()._copyOp.input().replaceAllUsesWith(concatPattern._concatParts.front()._copyOp.output());
        concatPattern._concatParts.front()._copyOp.erase();
    } else {
        for (auto concatPart : concatPattern._concatParts) {
            concatPart._copyOp.replaceAllUsesWith(convertCopyToSlice(concatPart._copyOp));
            concatPart._copyOp.erase();
        }
    }
}

void CMXConcatPass::safeRunOnFunc() {
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto availableMem = IE::getAvailableMemory(module, VPU::MemoryKind::CMX_NN);
    const auto cmxSize = checked_cast<size_t>(availableMem.size().count());

    func->walk([&](IE::ConcatOp concat) {
        auto inputPattern = getInputPattern(concat);
        std::cout << "after input pattern" << std::endl;
        auto outputPattern = getOutputPattern(concat);
        std::cout << "after output pattern" << std::endl;
        if (!inputPatternCanBeCMXed(concat, inputPattern, cmxSize)) {
            std::cout << "input pattern can not be cmx-ed" << std::endl;
            return;
        }
        if (!outputPatternCanBeCMXed(concat, outputPattern, cmxSize)) {
            std::cout << "output pattern can not be cmx-ed" << std::endl;
            return;
        }
        // rewrite from DDR to NNCMX
        rewriteInputPattern(inputPattern);
        rewriteOutputPattern(outputPattern);
    });
}

}  // namespace

//
// createCMXConcatPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createCMXConcatPass(Logger log) {
    return std::make_unique<CMXConcatPass>(log);
}
