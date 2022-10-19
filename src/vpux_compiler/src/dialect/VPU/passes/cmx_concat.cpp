//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include "vpux/utils/core/enums.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp>

using namespace vpux;
using namespace VPU;

namespace {

//
// CMXConcat
//

class CMXConcatPass final : public CMXConcatBase<CMXConcatPass> {
public:
    explicit CMXConcatPass(Logger logger): log(logger) {
        log.setName(Base::getArgumentName());
    }

public:
    // Storing Concat use chain consisting of (ClusterOp)NCE -> (Slice) -> (ClusterOp)Copy -> Concat
    // or Concat -> (Slice) -> (ClusterOp)Copy -> (ClusterOp)NCE
    struct ConcatPart {
        VPU::CopyOp copyOp;
        VPU::NCEClusterTilingOp copyClusterOp;
        VPU::SliceOp sliceOp;
        VPU::NCEOpInterface nceOp;
        VPU::NCEClusterTilingOp nceClusterOp;
        union {
            unsigned concatInput;  // concat input's index for input pattern
            unsigned nceInput;     // nceOp's input index for output pattern
        } index;

        ConcatPart(VPU::CopyOp copy, VPU::SliceOp slice, VPU::NCEOpInterface nce, unsigned idx)
                : copyOp(copy),
                  copyClusterOp(nullptr),
                  sliceOp(slice),
                  nceOp(nce),
                  nceClusterOp(nullptr),
                  index({idx}) {
        }

        ConcatPart(VPU::CopyOp copy, VPU::NCEClusterTilingOp copyCluster, VPU::SliceOp slice, VPU::NCEOpInterface nce,
                   VPU::NCEClusterTilingOp nceCluster, unsigned idx)
                : copyOp(copy),
                  copyClusterOp(copyCluster),
                  sliceOp(slice),
                  nceOp(nce),
                  nceClusterOp(nceCluster),
                  index({idx}) {
        }
        bool isValidPart() {
            return nceOp != nullptr;
        }
        bool hasSliceOp() {
            return sliceOp != nullptr;
        }
        bool isMultiCluster() {
            return copyClusterOp != nullptr && nceClusterOp != nullptr;
        }
        mlir::Operation* getNceOp() {
            return isMultiCluster() ? nceClusterOp.getOperation() : nceOp.getOperation();
        }
        mlir::Operation* getCopyOp() {
            return isMultiCluster() ? copyClusterOp.getOperation() : copyOp.getOperation();
        }
    };

    // Stores multiple Concat parts for a given Concat
    struct ConcatPattern {
        SmallVector<ConcatPart> concatParts;
        VPU::ConcatOp concat;
        Logger log;

        ConcatPattern(Logger log): concat(nullptr), log(log) {
        }
        void setConcat(VPU::ConcatOp rootConcat) {
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
        bool areConcatPartsTypesConsistent();
    };

private:
    void safeRunOnFunc();

private:
    Logger log;

private:
    bool isSplitSupportedOnDPU(VPU::SliceOp sliceOp);
    ConcatPattern getInputPattern(VPU::ConcatOp concat);
    ConcatPattern getOutputPattern(VPU::ConcatOp concat);
    void rewriteInputPattern(ConcatPattern& concatPattern);
    void rewriteOutputPattern(ConcatPattern& concatPattern);
    vpux::NDTypeInterface moveConcatToCMX(ConcatPattern& concatPattern);
    void replaceSliceCopy(ConcatPattern& concatPattern, vpux::NDTypeInterface origType,
                          vpux::NDTypeInterface newConcatType);
    bool isPotentialCMXConcat(VPU::ConcatOp concat);
    bool areInputOutputPatternsCompatible(ConcatPattern& inputPattern, ConcatPattern& outputPattern);
};

size_t CMXConcatPass::ConcatPattern::getSize(mlir::Value val) const {
    return static_cast<size_t>(getTotalSize(val).count());
}

CMXConcatPass::ConcatPattern CMXConcatPass::getInputPattern(VPU::ConcatOp concat) {
    ConcatPattern concatPattern(log);
    // store all required input info in a struct
    for (size_t inputIdx = 0; inputIdx < concat.inputs().size(); inputIdx++) {
        auto input = concat.getOperand(static_cast<int>(inputIdx));

        auto inputCopyOp = input.getDefiningOp<VPU::CopyOp>();
        auto inputClusterTilingOp = input.getDefiningOp<VPU::NCEClusterTilingOp>();

        if (inputClusterTilingOp) {
            if (auto innerOp = inputClusterTilingOp.getInnerTaskOpOfType<VPU::CopyOp>()) {
                inputCopyOp = innerOp;
            }
        }

        if (inputCopyOp == nullptr) {
            log.nest(2).trace("InputPattern mismatch: Copy op is not found");
            return concatPattern;
        }

        auto parentNCEOp = inputCopyOp.input().getDefiningOp<VPU::NCEOpInterface>();
        VPU::NCEClusterTilingOp parentNCEClusterTilingOp;

        if (inputClusterTilingOp) {
            auto inputClusterTiling = inputClusterTilingOp.getOperand(0);
            parentNCEClusterTilingOp = inputClusterTiling.getDefiningOp<VPU::NCEClusterTilingOp>();
            if (parentNCEClusterTilingOp) {
                if (auto innerOp = parentNCEClusterTilingOp.getInnerTaskOpOfType<VPU::NCEOpInterface>()) {
                    parentNCEOp = innerOp;
                }
            }
        }

        if (parentNCEOp == nullptr) {
            log.nest(2).trace("InputPattern mismatch: NCE op is not found");
            return concatPattern;
        }

        concatPattern.addConcatPart(ConcatPart(inputCopyOp, inputClusterTilingOp, nullptr, parentNCEOp,
                                               parentNCEClusterTilingOp, static_cast<int>(inputIdx)));
    }
    // make valid by adding concat
    concatPattern.setConcat(concat);
    return concatPattern;
}

bool CMXConcatPass::isSplitSupportedOnDPU(VPU::SliceOp sliceOp) {
    // Check if SubView performs a split along major dimension taking into account order in memory
    // For NCHW that would be split along C
    // For NHWC that would be split along H
    // Only such cases are supported by DPU IDU because only then input to DPU is a contiguous
    // block in memory. Otherwise this behavior needs to be performed by DMA
    const auto inputTypeShape = getShape(sliceOp.getOperand()).raw();
    const auto outputType = sliceOp.result().getType();

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

unsigned getIndexOfInput(mlir::Operation* endOp, mlir::Operation* sourceOp) {
    // Get the input index of an endOp which is from the sourceOp
    for (auto i : irange(endOp->getNumOperands())) {
        if (endOp->getOperand(i).getDefiningOp() == sourceOp) {
            return i;
        }
    }
    VPUX_THROW("Operation {0} at {1} is not a child of Operation {0} at {1}", endOp->getName(), endOp->getLoc(),
               sourceOp->getName(), sourceOp->getLoc());
}

CMXConcatPass::ConcatPattern CMXConcatPass::getOutputPattern(VPU::ConcatOp concat) {
    ConcatPattern concatPattern(log);
    // store all required output info in a struct
    for (auto user : concat.output().getUsers()) {
        auto outputSliceOp = mlir::dyn_cast<VPU::SliceOp>(user);
        auto outputCopyOp = mlir::dyn_cast<VPU::CopyOp>(user);
        auto outputClusterCopyOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(user);

        // Store the CopyOp or ClusterTiling(CopyOp)
        SmallVector<mlir::Operation*> copyOutOps;
        if (outputSliceOp) {
            // case 1. if the child of concat is SliceOp
            if (!isSplitSupportedOnDPU(outputSliceOp)) {
                log.nest(2).trace("OutputPattern mismatch: SliceOp is not supported on DPU");
                return concatPattern;
            }
            for (auto copyOp : outputSliceOp.result().getUsers()) {
                outputCopyOp = mlir::dyn_cast<VPU::CopyOp>(copyOp);
                outputClusterCopyOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(copyOp);

                if (outputClusterCopyOp) {
                    if (outputClusterCopyOp.getInnerTaskOpOfType<VPU::CopyOp>()) {
                        // match Concat->SliceOp->ClusterTiling(CopyOp)
                        copyOutOps.push_back(outputClusterCopyOp);
                        continue;
                    }
                    log.nest(2).trace("OutputPattern mismatch: No CopyOp in the ClusterTilingOp after Slice");
                    return concatPattern;
                } else if (outputCopyOp) {
                    // match Concat->SliceOp->CopyOp
                    copyOutOps.push_back(outputCopyOp);
                    continue;
                }
                log.nest(2).trace("OutputPattern mismatch: No CopyOp after Slice");
                return concatPattern;
            }
        } else if (outputClusterCopyOp) {
            // case 2. if the child of concat is ClusterTilingOp
            if (outputClusterCopyOp.getInnerTaskOpOfType<VPU::CopyOp>()) {
                // match Concat->ClusterTiling(CopyOp)
                copyOutOps.push_back(outputClusterCopyOp);
            } else {
                log.nest(2).trace("OutputPattern mismatch: No CopyOp in the ClusterTilingOp");
                return concatPattern;
            }
        } else if (outputCopyOp) {
            // case 3. if the child of concat is CopyOp
            copyOutOps.push_back(outputCopyOp);
        } else {
            log.nest(2).trace("OutputPattern mismatch: No CopyOp");
            return concatPattern;
        }

        // Look for the NCEOps according to the CopyOps
        for (auto& op : copyOutOps) {
            llvm::DenseSet<mlir::Operation*> nceUsers;
            for (auto opUser : op->getResult(0).getUsers()) {
                auto childNCEOp = mlir::dyn_cast<VPU::NCEOpInterface>(opUser);
                auto childNCEClusterOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(opUser);
                if (childNCEClusterOp) {
                    if (auto innerOp = childNCEClusterOp.getInnerTaskOpOfType<VPU::NCEOpInterface>()) {
                        childNCEOp = innerOp;
                    }
                }

                if (childNCEOp == nullptr) {
                    log.nest(2).trace("OutputPattern mismatch: No NCEOp");
                    return concatPattern;
                }

                if (nceUsers.find(opUser) != nceUsers.end()) {
                    // avoid multiple reads from the same location at the same time
                    log.nest(2).trace("Concat input used twice by the same operation, can not cmx");
                    return concatPattern;
                }
                nceUsers.insert(opUser);
                if (childNCEClusterOp) {
                    auto copyOp = mlir::dyn_cast<VPU::CopyOp>(op);
                    auto copyClusterOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(op);
                    if (copyClusterOp) {
                        if (auto innerOp = copyClusterOp.getInnerTaskOpOfType<VPU::CopyOp>()) {
                            copyOp = innerOp;
                        }
                    }

                    if (copyOp == nullptr) {
                        log.nest(2).trace("OutputPattern mismatch: No CopyOp");
                        return concatPattern;
                    }

                    auto index = getIndexOfInput(childNCEClusterOp, op);
                    concatPattern.addConcatPart(
                            ConcatPart(copyOp, copyClusterOp, outputSliceOp, childNCEOp, childNCEClusterOp, index));
                } else {
                    auto index = getIndexOfInput(childNCEOp, op);
                    concatPattern.addConcatPart(
                            ConcatPart(mlir::dyn_cast<VPU::CopyOp>(op), outputSliceOp, childNCEOp, index));
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

    log.nest(3).trace("Concat size '{0}'", (concatSize + maxUserSize));
    // return concat size smaller than CMX size
    return (concatSize + maxUserSize) <= cmxSize;
}

bool CMXConcatPass::ConcatPattern::inputsHaveMultipleUsers() const {
    // avoid concats which are complex, where the inputs to the concat are used
    // by other operations
    for (auto concatPart : concatParts) {
        auto nceOp = concatPart.getNceOp();
        for (auto result : nceOp->getResults()) {
            for (auto resultUser : result.getUsers()) {
                if (resultUser != concatPart.getCopyOp()) {
                    // the NCE contains a different user
                    return true;
                }
            }
        }
    }

    return false;
}

bool areDistributedTypesConcatenable(VPU::DistributedTensorType firstType, VPU::DistributedTensorType secondType) {
    return firstType.getOrder() == secondType.getOrder() && firstType.getMemSpace() == secondType.getMemSpace() &&
           firstType.getDistribution() == secondType.getDistribution();
}

bool CMXConcatPass::ConcatPattern::inputPatternCanBeCMXed(size_t cmxSize) {
    // if concat is a Result operation
    for (auto concatOutputUser : concat.output().getUsers()) {
        if (mlir::isa<mlir::ReturnOp>(concatOutputUser)) {
            log.nest(2).trace("Concat output is part of network output");
            return false;
        }
    }

    // Check if all inputs are of the same type
    if (!areConcatPartsTypesConsistent()) {
        log.nest(2).trace("Concat contains both single and multi cluster inputs");
        return false;
    }

    // Check compatibility between input distribution types
    if (concatParts[0].isMultiCluster()) {
        if (const auto inTypeDistributed =
                    concatParts[0].nceClusterOp.getResult(0).getType().dyn_cast<VPU::DistributedTensorType>()) {
            for (auto concatPart : llvm::makeArrayRef(concatParts).drop_front()) {
                const auto curType = concatPart.nceClusterOp.getResult(0).getType().cast<VPU::DistributedTensorType>();
                if (!areDistributedTypesConcatenable(inTypeDistributed, curType)) {
                    log.nest(2).trace(
                            "Not matching distributed tensor attributes between concat inputs: `{0}` and `{1}`",
                            inTypeDistributed, curType);
                    return false;
                }
            }
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
        auto nceOp = concatPart.getNceOp();
        auto copyOp = concatPart.getCopyOp();
        for (auto input : nceOp->getOperands()) {
            if (input.getDefiningOp() == copyOp) {
                continue;
            }
            consumerInputSize += getSize(input);
        }
        for (auto output : nceOp->getResults()) {
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
        // for fused loc ignore tiling details
        auto nceOp = concatPart.getNceOp();

        if (const auto fused = nceOp->getLoc().dyn_cast<mlir::FusedLoc>()) {
            auto nceLoc = fused.getLocations().front();
            if (llvm::find(fusedLocations, nceLoc) == fusedLocations.end()) {
                // tiling producers
                fusedLocations.push_back(nceLoc);
            }
        } else {
            auto nceLoc = nceOp->getLoc();
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

    // Check if all output branches are of the same type
    if (!areConcatPartsTypesConsistent()) {
        log.nest(2).trace("Concat contains both single and multi cluster outputs");
        return false;
    }

    // Check compatibility between output distribution types
    if (concatParts[0].isMultiCluster()) {
        auto inputIndex = concatParts[0].index.nceInput;
        if (const auto inTypeDistributed = concatParts[0]
                                                   .nceClusterOp.getOperand(inputIndex)
                                                   .getType()
                                                   .dyn_cast<VPU::DistributedTensorType>()) {
            for (auto concatPart : llvm::makeArrayRef(concatParts).drop_front()) {
                const auto curType =
                        concatPart.nceClusterOp.getOperand(inputIndex).getType().cast<VPU::DistributedTensorType>();
                if (!areDistributedTypesConcatenable(inTypeDistributed, curType)) {
                    log.nest(2).trace(
                            "Not matching distributed tensor attributes between concat outputs: `{0}` and `{1}`",
                            inTypeDistributed, curType);
                    return false;
                }
            }
        }
    }

    return true;
}

bool CMXConcatPass::ConcatPattern::areConcatPartsTypesConsistent() {
    // Check if all branches are of the same type
    // Either all or none should be in multi cluster mode
    size_t nceClusterTilingParts = 0;
    for (auto concatPart : concatParts) {
        if (concatPart.isMultiCluster()) {
            nceClusterTilingParts++;
        }
    }
    if (nceClusterTilingParts > 0 && nceClusterTilingParts != concatParts.size()) {
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
      SubView   SubView (added in VPUIP)
          \    /
          Concat
    */
    for (auto concatPart : concatPattern.concatParts) {
        log.nest(1).trace("Removing input Copy from NNCMX to DDR '{0}' at '{1}'", concatPart.copyOp->getName(),
                          concatPart.copyOp->getLoc());
        // modify only current concat input as it may have multiple uses
        auto newConcatInput = concatPart.getCopyOp()->getOperand(0);
        concatPattern.concat.setOperand(concatPart.index.concatInput, newConcatInput);
    }
}

// Moves the concat to NNCMX, inserts DistributedCast if needed and returns the new output type
// of the Concat or DistributedCast
vpux::NDTypeInterface CMXConcatPass::moveConcatToCMX(ConcatPattern& concatPattern) {
    auto concatOp = concatPattern.concat;
    mlir::OpBuilder builder(concatOp);
    builder.setInsertionPointAfter(concatOp);
    const auto concatType = concatOp.output().getType().cast<vpux::NDTypeInterface>();

    log.nest().trace("Moving output to NNCMX for '{0}' at '{1}'", concatOp->getName(), concatOp->getLoc());

    if (!concatPattern.concatParts[0].isMultiCluster()) {
        const auto memSpaceCMX = IndexedSymbolAttr::get(builder.getContext(), stringifyEnum(MemoryKind::CMX_NN), 0);
        auto newConcatType = concatType.changeMemSpace(memSpaceCMX);
        concatOp.output().setType(newConcatType);
        return newConcatType;
    }

    // If the outputPattern has MC layer, then the inputPattern must have MC layer so the concat could be
    // CMX-ed. Infer concat output type
    auto newConcatType =
            concatOp->getOperand(0).getType().cast<vpux::NDTypeInterface>().changeShape(concatType.getShape());
    concatOp.output().setType(newConcatType);

    // Insert DistributedCast if the input pattern has different distributed mode
    auto inputPatternMode = newConcatType.cast<VPU::DistributedTensorType>().getDistribution().mode().getValue();
    auto outputPatternType = concatPattern.concatParts[0]
                                     .copyClusterOp->getResult(0)
                                     .getType()
                                     .cast<vpux::NDTypeInterface>()
                                     .changeShape(newConcatType.getShape())
                                     .cast<VPU::DistributedTensorType>();
    auto outputPatternMode = outputPatternType.getDistribution().mode().getValue();
    if (inputPatternMode != outputPatternMode) {
        log.nest().trace("Inserting DistributedCast at '{1}'", concatOp->getLoc());
        auto distributedCastOp =
                builder.create<VPU::DistributedCastOp>(concatOp->getLoc(), outputPatternType, concatOp.output());
        concatOp.output().replaceAllUsesExcept(distributedCastOp.output(),
                                               llvm::SmallPtrSet<mlir::Operation*, 1>{distributedCastOp});
        newConcatType = outputPatternType;
    }

    return newConcatType;
}

void CMXConcatPass::replaceSliceCopy(ConcatPattern& concatPattern, vpux::NDTypeInterface origType,
                                     vpux::NDTypeInterface newConcatType) {
    VPUX_THROW_WHEN(concatPattern.concatParts.size() < 1, "Concat has no output parts");
    mlir::OpBuilder builder(concatPattern.concatParts[0].copyOp);
    for (auto concatPart : concatPattern.concatParts) {
        if (concatPart.isMultiCluster()) {
            if (concatPart.hasSliceOp()) {
                // IE.Slice to Copy->IE.Slice
                auto sliceOp = concatPart.sliceOp;
                auto vpuNodeOp = sliceOp.source().getDefiningOp();
                log.nest().trace("Creating VPU.Slice '{0}' at '{1}'", sliceOp->getName(), sliceOp->getLoc());
                builder.setInsertionPoint(concatPart.nceClusterOp);
                auto newSliceOp =
                        builder.create<VPU::SliceOp>(sliceOp->getLoc(), vpuNodeOp->getResult(0),
                                                     sliceOp.static_offsetsAttr(), sliceOp.static_sizesAttr());
                concatPart.nceClusterOp->setOperand(concatPart.index.nceInput, newSliceOp);

                // The original slice op cannot be erased because it is behind the concat
                // A new copy should be inserted to save the original type for the slice's input
                // The new copy will be removed later
                auto newCopy = VPU::createDistributedCopyOut(vpuNodeOp, origType);
                sliceOp->setOperand(0, newCopy.getResult(0));
            } else {
                log.nest().trace("Removing output Copy from DDR to NNCMX '{0}' at '{1}'", concatPart.copyOp->getName(),
                                 concatPart.copyOp->getLoc());
                concatPart.copyClusterOp.getResult(0).replaceAllUsesWith(concatPart.copyClusterOp.getOperand(0));
            }
        } else {
            // correct the tensor type for slice op
            if (concatPart.hasSliceOp()) {
                auto origSliceOp = concatPart.sliceOp;
                origSliceOp.source().setType(newConcatType);
                builder.setInsertionPoint(origSliceOp);
                log.nest(1).trace("Creating VPU.Slice '{0}' at '{1}'", origSliceOp->getName(), origSliceOp->getLoc());
                auto newSliceOp =
                        builder.create<VPU::SliceOp>(origSliceOp->getLoc(), origSliceOp->getOperand(0),
                                                     origSliceOp.static_offsetsAttr(), origSliceOp.static_sizesAttr());
                origSliceOp.replaceAllUsesWith(newSliceOp.result());
            }

            log.nest().trace("Removing output Copy from DDR to NNCMX '{0}' at '{1}'", concatPart.copyOp->getName(),
                             concatPart.copyOp->getLoc());
            concatPart.copyOp.output().replaceAllUsesWith(concatPart.copyOp.input());
        }
    }
}

void CMXConcatPass::rewriteOutputPattern(ConcatPattern& concatPattern) {
    /*
                            From DDR IR

         VPU.Concat
           /    \
     VPU.Slice   VPU.Slice                         VPU.Concat
         |        |                                  |
        Copy     Copy                               Copy
         |        |                                  |
        NCE      NCE                                NCE
                            TO NNCMX IR

         VPU.Concat                              VPU.Concat
             |                                       |
     (DistributedCast)                       (DistributedCast)
           /    \                                    |
     VPU.Slice VPU.Slice                            NCE
         |        |
        NCE      NCE
    */
    auto concatOp = concatPattern.concat;
    const auto origType = concatOp.output().getType().cast<vpux::NDTypeInterface>();

    auto newConcatType = moveConcatToCMX(concatPattern);
    replaceSliceCopy(concatPattern, origType, newConcatType);
}

bool CMXConcatPass::areInputOutputPatternsCompatible(CMXConcatPass::ConcatPattern& inputPattern,
                                                     CMXConcatPass::ConcatPattern& outputPattern) {
    // Check if the input and outputPattern satisfy
    // both are distributed types or both are normal types
    bool inputHasDistributed = false;
    bool outputHasDistributed = false;
    for (auto concatPart : inputPattern.concatParts) {
        if (concatPart.isMultiCluster()) {
            inputHasDistributed = true;
            break;
        }
    }
    for (auto concatPart : outputPattern.concatParts) {
        if (concatPart.isMultiCluster()) {
            outputHasDistributed = true;
            break;
        }
    }
    if (inputHasDistributed != outputHasDistributed) {
        // different input output type
        return false;
    }
    if (inputHasDistributed && outputHasDistributed) {
        auto inputType = inputPattern.concatParts[0]
                                 .nceClusterOp->getResult(0)
                                 .getType()
                                 .cast<vpux::VPU::DistributedTensorType>();
        auto inputMode = inputType.getDistribution().mode().getValue();
        auto outputType = outputPattern.concatParts[0]
                                  .nceClusterOp->getOperand(0)
                                  .getType()
                                  .cast<vpux::VPU::DistributedTensorType>();
        auto outputMode = outputType.getDistribution().mode().getValue();
        if (mlir::failed(areDistributionModesCompatible(inputMode, outputMode))) {
            return false;
        }
        auto inputNumClusters = inputType.getDistribution().num_clusters();
        auto outputNumClusters = outputType.getDistribution().num_clusters();
        if (mlir::failed(areDistributionNumClustersCompatible(inputNumClusters, outputNumClusters))) {
            return false;
        }
    }
    return true;
}

bool CMXConcatPass::isPotentialCMXConcat(VPU::ConcatOp concat) {
    // Check if the Concat op satisfies the CMX Concat conditions or not
    auto isSingleAxisConcat = [](mlir::ArrayAttr offset) {
        // If a concat has at least one static_offset attribute of 2 or more non-zero axis
        // it is considered as multiple-axis concat, vice versa
        // e.g., static_offset of a multiple-axis concat:
        // [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]
        auto offsetVector = parseIntArrayAttr<int64_t>(offset);
        return offsetVector.size() - std::count(offsetVector.begin(), offsetVector.end(), 0) <= 1;
    };

    if (!concat.static_offsetsAttr()) {
        return true;
    }

    return std::all_of(concat.static_offsetsAttr().getAsRange<mlir::ArrayAttr>().begin(),
                       concat.static_offsetsAttr().getAsRange<mlir::ArrayAttr>().end(), isSingleAxisConcat);
}

void CMXConcatPass::safeRunOnFunc() {
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto availableMem = VPU::getTotalCMXSize(module);
    const auto cmxSize = checked_cast<size_t>(availableMem.count());

    func->walk([&](VPU::ConcatOp concat) {
        // check concat input pattern
        log.trace("Got '{0}' at '{1}'", concat->getName(), concat->getLoc());
        if (!isPotentialCMXConcat(concat)) {
            log.nest(1).trace("Concat cannot be executed on CMX");
            return;
        }
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
        if (!areInputOutputPatternsCompatible(inputPattern, outputPattern)) {
            log.nest(1).trace("Concat input and output pattern type mismatch");
            return;
        }
        // rewrite from DDR to NNCMX
        log.trace("Concat '{0}' at '{1}' will be moved to CMX", concat->getName(), concat->getLoc());
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
