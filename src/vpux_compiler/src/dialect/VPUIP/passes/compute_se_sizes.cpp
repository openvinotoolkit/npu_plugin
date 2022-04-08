//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/PassManager.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

SmallVector<VPUIP::NCEClusterTaskOp> findProducerOps(mlir::Value value) {
    SmallVector<VPUIP::NCEClusterTaskOp> producerNCEOps;

    auto producerOp = value.getDefiningOp();
    auto taskOp = producerOp;
    if (auto nceClusterOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(producerOp)) {
        taskOp = nceClusterOp.getInnerTaskOp();
    }

    if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(taskOp)) {
        producerNCEOps.push_back(nceOp);
    } else if (mlir::isa<VPUIP::CopyOp>(taskOp)) {
        const auto ops = findProducerOps(producerOp->getOperand(0));
        producerNCEOps.append(ops);
    } else if (VPUIP::isPureViewOp(producerOp)) {
        llvm::TypeSwitch<mlir::Operation*, void>(producerOp)
                .Case<VPUIP::ConcatViewOp>([&](VPUIP::ConcatViewOp concatOp) {
                    for (auto input : concatOp.inputs()) {
                        const auto ops = findProducerOps(input);
                        producerNCEOps.append(ops);
                    }
                })
                .Case<mlir::ViewLikeOpInterface>([&](mlir::ViewLikeOpInterface viewOp) {
                    const auto ops = findProducerOps(viewOp.getViewSource());
                    producerNCEOps.append(ops);
                })
                .Case<MultiViewOpInterface>([&](MultiViewOpInterface viewOp) {
                    if (auto opResult = value.dyn_cast<mlir::OpResult>()) {
                        const auto source = viewOp.getViewSource(opResult.getResultNumber());
                        const auto ops = findProducerOps(source);
                        producerNCEOps.append(ops);
                    }
                })
                .Case<GroupedViewOpInterface>([&](GroupedViewOpInterface viewOp) {
                    for (auto source : viewOp.getViewSources()) {
                        const auto ops = findProducerOps(source);
                        producerNCEOps.append(ops);
                    }
                });
    }

    return producerNCEOps;
}

// Get the number of channels produced by the operation's variants. Each variant must produce the same
// number of channels in order for the sparse output to be valid.
int64_t getVariantsNumChannels(VPUIP::NCEClusterTaskOp nceOp) {
    Optional<int64_t> numChannels = None;
    for (auto dpuTaskOp : nceOp.variants().getOps<VPUIP::DPUTaskOp>()) {
        const auto outStart = parseIntArrayAttr<int64_t>(dpuTaskOp.outStart());
        const auto outEnd = parseIntArrayAttr<int64_t>(dpuTaskOp.outEnd());
        VPUX_THROW_UNLESS(outStart.size() >= 3 && outEnd.size() >= 3,
                          "Invalid variant shape, expected at least three dimensions, got '{0}'", outStart.size());
        const auto variantNumChannels = outEnd[2] - outStart[2] + 1;
        if (!numChannels.hasValue()) {
            numChannels = variantNumChannels;
            continue;
        }
        VPUX_THROW_UNLESS(numChannels.getValue() == variantNumChannels,
                          "Variant has '{0}' channels while '{1}' were expected", variantNumChannels, numChannels);
    }
    return numChannels.getValue();
}

mlir::IntegerAttr getInputSESizeForConcatOverC(VPUIP::NCEClusterTaskOp nceOp, mlir::Value operand) {
    auto blockArg = operand.dyn_cast<mlir::BlockArgument>();
    auto parentTilingOp = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
    if (blockArg != nullptr && parentTilingOp != nullptr) {
        operand = parentTilingOp->getOperand(blockArg.getArgNumber());
    }

    const auto producerOps = findProducerOps(operand);
    VPUX_THROW_WHEN(producerOps.empty(), "No producer operations found");

    const auto producerOpsNumChannels = to_small_vector(producerOps | transformed([](VPUIP::NCEClusterTaskOp nceOp) {
                                                            return getVariantsNumChannels(nceOp);
                                                        }));
    const auto unequalChannels = std::adjacent_find(producerOpsNumChannels.begin(), producerOpsNumChannels.end(),
                                                    std::not_equal_to<>()) != producerOpsNumChannels.end();
    VPUX_THROW_WHEN(unequalChannels, "Not all producer output channels are equal");

    // Check if the number of output channels of the producer variants is different than the consumer operation's
    // input channels, as this represents a case where the activations are concatenated over channels
    const auto producerChannels = producerOpsNumChannels.front();
    const auto consumerChannels = operand.getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C];
    if (producerChannels != consumerChannels) {
        return getIntAttr(nceOp.getContext(), producerChannels);
    }
    return nullptr;
}

//
// ComputeSESizesPass
//

class ComputeSESizesPass final : public VPUIP::ComputeSESizesBase<ComputeSESizesPass> {
public:
    explicit ComputeSESizesPass(Optional<bool> onlyInputsConcatOverC, Logger log)
            : _onlyInputsConcatOverC(onlyInputsConcatOverC) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

    Optional<bool> _onlyInputsConcatOverC;
};

mlir::LogicalResult ComputeSESizesPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (!onlyInputsConcatOverC.hasValue()) {
        return mlir::success();
    }
    _onlyInputsConcatOverC = onlyInputsConcatOverC;
    return mlir::success();
}

void ComputeSESizesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    // Set the storage element size attributes only for the input operand in case the sparse data is concatenated
    // over channels. This is necessary since sparse activations are sparsified individually by each DPU producer
    // and must be desparsified in the same way.
    // Consumers of sparse activations have a single storage element size register configuration (i.e. se_z_split).
    // Therefore, sparse activations concatenated over channels must all have the same number of channels,
    // regardless of whether the concatenation is done explicitly, using ODU broadcasting, by multiple variants
    // of the same invariant, or a combination of these methods.
    if (_onlyInputsConcatOverC.getValueOr(false)) {
        _log.trace("Setting storage element sizes only for inputs concatenated over channels");

        func.walk([&](VPUIP::NCEClusterTaskOp nceOp) {
            _log.trace("Handling operation '{0}' at '{1}'", nceOp->getName(), nceOp->getLoc());

            if (nceOp.input_sparsity_map() != nullptr) {
                if (auto seSizeAttr = getInputSESizeForConcatOverC(nceOp, nceOp.input())) {
                    VPUX_THROW_UNLESS(isPowerOfTwo(seSizeAttr.getInt()),
                                      "Value '{0}' for concatenated input is not a power of 2", seSizeAttr);
                    _log.nest().trace("Setting input_se_size to '{0}'", seSizeAttr);
                    nceOp.input_se_sizeAttr(seSizeAttr);
                }
            }

            if (nceOp.task_type() == VPUIP::NCETaskType::ELTWISE && nceOp.weights_sparsity_map() != nullptr) {
                if (auto seSizeAttr = getInputSESizeForConcatOverC(nceOp, nceOp.weights())) {
                    const auto inputSeSize = nceOp.input_se_size().getValueOr(seSizeAttr.getInt());
                    VPUX_THROW_UNLESS(
                            inputSeSize == seSizeAttr.getInt(),
                            "Different storage element sizes expected for the two Eltwise inputs: {0} and {1} at '{2}'",
                            inputSeSize, seSizeAttr.getInt(), nceOp->getLoc());
                }
            }
        });

        return;
    }

    // Set the storage element sizes to be equal to the number of channels of the activation, if not already set.
    _log.trace("Setting storage element sizes for all sparse activations");

    func.walk([&](VPUIP::NCEClusterTaskOp nceOp) {
        _log.trace("Handling operation '{0}' at '{1}'", nceOp->getName(), nceOp->getLoc());

        if (nceOp.input_sparsity_map() != nullptr && nceOp.input_se_sizeAttr() == nullptr) {
            auto numChannels = nceOp.input().getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C];
            VPUX_THROW_UNLESS(isPowerOfTwo(numChannels), "Value '{0}' for input is not a power of 2", numChannels);
            _log.nest().trace("Setting input_se_size to '{0}'", numChannels);
            nceOp.input_se_sizeAttr(getIntAttr(&ctx, numChannels));

            if (nceOp.task_type() == VPUIP::NCETaskType::ELTWISE) {
                auto numChannelsInput2 =
                        nceOp.weights().getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C];
                VPUX_THROW_UNLESS(numChannels == numChannelsInput2,
                                  "Different storage element sizes expected for the two Eltwise inputs: {0} and {1}",
                                  numChannels, numChannelsInput2);
            }
        }

        if (nceOp.output_sparsity_map_buff() != nullptr && nceOp.output_se_sizeAttr() == nullptr) {
            auto numChannels = getVariantsNumChannels(nceOp);
            VPUX_THROW_UNLESS(isPowerOfTwo(numChannels), "Value '{0}' for output is not a power of 2", numChannels);
            _log.nest().trace("Setting output_se_size to '{0}'", numChannels);
            nceOp.output_se_sizeAttr(getIntAttr(&ctx, numChannels));
        }
    });
}

}  // namespace

//
// createComputeSESizesPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createComputeSESizesPass(Optional<bool> onlyInputsConcatOverC, Logger log) {
    return std::make_unique<ComputeSESizesPass>(onlyInputsConcatOverC, log);
}
