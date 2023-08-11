//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/dpu_tiler.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;
using namespace VPU;

namespace {

constexpr std::array<int64_t, 3> supportedChannelsDW = {64, 32, 16};
constexpr std::array<int64_t, 10> supportedChannelsPowerOfTwo = {8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16};

// Get a set containing all the channels from the workloads of the given NCE operations
mlir::DenseSet<int64_t> getWorkloadsChannels(const mlir::DenseSet<mlir::Operation*>& nceOps) {
    mlir::DenseSet<int64_t> workloadsChannels;
    for (auto op : nceOps) {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
        VPUX_THROW_UNLESS(nceOp != nullptr, "Expected NCE op, got '{0}'", op);
        auto workloads = nceOp.workloads().getOps<VPU::DPUWorkloadOp>();
        auto channels = to_container<mlir::DenseSet<int64_t>>(
                workloads | transformed([](VPU::DPUWorkloadOp workload) -> int64_t {
                    const auto wlSizes = parseIntArrayAttr<int64_t>(workload.outSizes());
                    return wlSizes[Dims4D::Act::C.ind()];
                }));
        workloadsChannels.insert(channels.begin(), channels.end());
    }
    return workloadsChannels;
}

// Find the operations which can consume the given value. The value should be of sparse type, therefore the consumers
// can be NCE, Desparsify or Return ops
mlir::DenseSet<mlir::Operation*> findConsumerOps(mlir::Value value) {
    mlir::DenseSet<mlir::Operation*> consumerOps;
    for (auto userOp : value.getUsers()) {
        auto taskOp = userOp;
        if (auto nceClusterOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(userOp)) {
            taskOp = nceClusterOp.getInnerTaskOp();
        }

        if (mlir::isa<VPU::NCEOpInterface, VPU::DesparsifyOp, mlir::func::ReturnOp>(taskOp)) {
            consumerOps.insert(userOp);
        } else if (mlir::isa<VPU::CopyOp>(taskOp) || VPU::isPureViewOp(taskOp)) {
            auto ops = findConsumerOps(userOp->getResult(0));
            consumerOps.insert(ops.begin(), ops.end());
        }
    }
    return consumerOps;
}

// Find all the NCE operations that produce the value. Multiple operations can produce a value in case it is
// concatenated or grouped
mlir::DenseSet<mlir::Operation*> findProducerNCEOps(mlir::Value value) {
    mlir::DenseSet<mlir::Operation*> producerNCEOps;

    auto producerOp = value.getDefiningOp();
    auto taskOp = producerOp;
    if (auto nceClusterOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(producerOp)) {
        taskOp = nceClusterOp.getInnerTaskOp();
    }

    if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(taskOp)) {
        producerNCEOps.insert(nceOp);
    } else if (mlir::isa<VPU::CopyOp>(taskOp)) {
        const auto ops = findProducerNCEOps(producerOp->getOperand(0));
        producerNCEOps.insert(ops.begin(), ops.end());
    } else if (VPU::isPureViewOp(producerOp)) {
        if (auto concatOp = mlir::dyn_cast<VPU::ConcatOp>(producerOp)) {
            for (auto input : concatOp.inputs()) {
                const auto ops = findProducerNCEOps(input);
                producerNCEOps.insert(ops.begin(), ops.end());
            }
        } else if (auto viewOp = mlir::dyn_cast<VPU::ViewLikeOpInterface>(producerOp)) {
            const auto ops = findProducerNCEOps(viewOp->getOperand(0));
            producerNCEOps.insert(ops.begin(), ops.end());
        } else if (auto viewOp = mlir::dyn_cast<MultiViewOpInterface>(producerOp)) {
            if (auto opResult = value.dyn_cast<mlir::OpResult>()) {
                const auto source = viewOp.getViewSource(opResult.getResultNumber());
                const auto ops = findProducerNCEOps(source);
                producerNCEOps.insert(ops.begin(), ops.end());
            }
        } else if (auto viewOp = mlir::dyn_cast<GroupedViewOpInterface>(producerOp)) {
            for (auto source : viewOp.getViewSources()) {
                const auto ops = findProducerNCEOps(source);
                producerNCEOps.insert(ops.begin(), ops.end());
            }
        }
    }

    return producerNCEOps;
}

// Find all the consumer operations of the value, then find all the producer NCE operations for the input values of the
// consumers. This is then repeated until no new consumers are identified.
// Chains of operations such as the following ensure that all three input Convolutions are returned when the function
// is called on the value marked with '*':
//   Conv   Conv   Conv
//     \*  /    \  /
//     Concat  Concat
//       |       |
//      Conv    Conv
mlir::DenseSet<mlir::Operation*> findProducersForConsumers(mlir::Value value,
                                                           mlir::DenseSet<mlir::Operation*> processedConsumerOps = {}) {
    mlir::DenseSet<mlir::Operation*> producerNCEOps;

    auto consumerOps = findConsumerOps(value);
    for (auto consumerOp : consumerOps) {
        if (processedConsumerOps.contains(consumerOp)) {
            continue;
        }

        auto producerOps = findProducerNCEOps(consumerOp->getOperand(0));
        producerNCEOps.insert(producerOps.begin(), producerOps.end());

        if (mlir::isa<VPU::NCEEltwiseOp>(consumerOp)) {
            auto producerOpsInput2 = findProducerNCEOps(consumerOp->getOperand(1));
            producerNCEOps.insert(producerOpsInput2.begin(), producerOpsInput2.end());
        }
    }
    processedConsumerOps.insert(consumerOps.begin(), consumerOps.end());

    mlir::DenseSet<mlir::Operation*> newProducerOps;
    for (auto producerOp : producerNCEOps) {
        if (producerOp->getResult(0) == value) {
            continue;
        }
        auto producerOps = findProducersForConsumers(producerOp->getResult(0), processedConsumerOps);
        newProducerOps.insert(producerOps.begin(), producerOps.end());
    }

    producerNCEOps.insert(newProducerOps.begin(), newProducerOps.end());

    return producerNCEOps;
}

// Invariants that produce sparse activations must satisfy two conditions:
// - all variants must produce the same number of channels
// - the number of channels is a power of two
// Additionally, in case a consumer operation has its input produced by multiple NCE operations,
// all of the producer ops need to have the same number of channels for their variants.
mlir::DenseSet<mlir::Operation*> findInvalidSparseOps(VPU::NCEOpInterface nceOp) {
    mlir::DenseSet<mlir::Operation*> invalidSparseOps;

    if (nceOp->getResult(0).getType().dyn_cast<VPU::SparseTensorType>() == nullptr) {
        return invalidSparseOps;
    }

    auto result = nceOp->getResult(0);
    if (auto parentOp = nceOp->getParentOfType<VPU::NCEClusterTilingOp>()) {
        result = parentOp->getResult(0);
    }
    auto producerOps = findProducersForConsumers(result);
    auto workloadsChannels = getWorkloadsChannels(producerOps);

    Optional<int64_t> numChannels = None;
    auto invalidWorkloads = llvm::any_of(workloadsChannels, [&](int64_t channels) -> bool {
        if (!numChannels.hasValue()) {
            numChannels = channels;
        }
        return !isPowerOfTwo(channels) || (channels != numChannels);
    });
    if (invalidWorkloads) {
        invalidSparseOps.insert(producerOps.begin(), producerOps.end());
    }

    return invalidSparseOps;
}

// Depthwise operations must have variants that produce 16, 32 or 64 channels
mlir::DenseSet<mlir::Operation*> findInvalidDepthwiseOps(const mlir::DenseSet<mlir::Operation*>& nceOps) {
    mlir::DenseSet<mlir::Operation*> invalidDepthwiseOps;
    for (auto op : nceOps) {
        if (!mlir::isa<VPU::NCEDepthConvolutionOp, VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp>(op)) {
            continue;
        }
        const auto workloadsChannels = getWorkloadsChannels({op});
        const auto invalidChannels = llvm::any_of(workloadsChannels, [&](const int64_t channels) -> bool {
            return llvm::find(supportedChannelsDW, channels) == supportedChannelsDW.end();
        });
        if (invalidChannels) {
            invalidDepthwiseOps.insert(op);
        }
    }
    return invalidDepthwiseOps;
}

// Since NCE.PermuteQuantize output rotates dimensions, pads cannot be used in conventional way.
// It is necessary to subtract pads from respective spatial dimensions and then set zero padding.
// This allows to set workload size Y according to input Y and amend output Y without extra DMA tasks.
// Consider NCE.PermuteQuantize with input tensor 1x32x3xM and output tensor 1x32x16xM.
// For such input tensor, workload_end_Y must be set to 3 - 1 = 2.
// However, output Y must have size 16 - 1 = 15 to be aligned with requirements of consuming operation.
mlir::DenseSet<mlir::Operation*> findInvalidPermuteQuantizeOps(const mlir::DenseSet<mlir::Operation*>& nceOps) {
    mlir::DenseSet<mlir::Operation*> invalidPermuteQuantizeOps;
    for (auto op : nceOps) {
        if (!mlir::isa<VPU::NCEPermuteQuantizeOp>(op)) {
            continue;
        }
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
        VPUX_THROW_UNLESS(nceOp != nullptr, "Expected NCE op, got '{0}'", op);
        const auto workloads = nceOp.workloads().getOps<VPU::DPUWorkloadOp>();
        const auto nonZeroPadding = llvm::any_of(workloads, [&](VPU::DPUWorkloadOp workload) -> bool {
            const auto pads = workload.pad();
            const auto top = pads.top().getInt();
            const auto bottom = pads.bottom().getInt();
            const auto left = pads.left().getInt();
            const auto right = pads.right().getInt();
            const auto zeroPadding = top == 0 && bottom == 0 && left == 0 && right == 0;
            const auto wlOffsets = parseIntArrayAttr<int64_t>(workload.outOffsetsAttr());
            const auto isZeroPredicate = [](const int64_t value) -> bool {
                return value == 0;
            };
            // Zero offsets are actually helpful when it comes to tiling of PermuteQuantize.
            // Tile over width in segmented mode works properly with the following setup:
            // Parent input buffer shape:    [1, 3, 224, 224]
            // Parent output buffer shape:   [1, 4, 224, 224]
            //
            // Tile 1 DDR2CMX DMA:  offsets = [0, 0,   0, 0] sizes = [1,   3, 112, 224]
            // Tile 1 workload:     offsets = [0, 0,   0, 0] sizes = [1, 224,   3, 112]
            // Tile 1 CMX2DDR DMA:  offsets = [0, 0,   0, 0] sizes = [1, 224,   4, 112]
            //
            // Tile 2 DDR2CMX DMA:  offsets = [0, 0, 112, 0] sizes = [1,   3, 112, 224]
            // Tile 2 workload:     offsets = [0, 0,   0, 0] sizes = [1, 224,   3, 112]
            // Tile 2 CMX2DDR DMA:  offsets = [0, 0,   0, 0] sizes = [1, 224,   4, 112]
            const bool zeroOffsets = std::all_of(wlOffsets.begin(), wlOffsets.end(), isZeroPredicate);
            return !zeroPadding || !zeroOffsets;
        });
        if (nonZeroPadding) {
            invalidPermuteQuantizeOps.insert(op);
        }
    }

    return invalidPermuteQuantizeOps;
}

// Get all of the supported channels that can be used to split all of the given workloads, so that the
// depthwise and sparsity requirements are met
SmallVector<int64_t> getSupportedChannels(const mlir::DenseSet<mlir::Operation*>& nceOps,
                                          const bool depthwiseCorrection, const bool sparsityCorrection) {
    SmallVector<int64_t> supportedChannels;

    if (depthwiseCorrection) {
        supportedChannels.insert(supportedChannels.end(), supportedChannelsDW.begin(), supportedChannelsDW.end());
    }

    if (sparsityCorrection) {
        if (supportedChannels.empty()) {
            supportedChannels.insert(supportedChannels.end(), supportedChannelsPowerOfTwo.begin(),
                                     supportedChannelsPowerOfTwo.end());
        }
        const auto workloadsChannels = getWorkloadsChannels(nceOps);
        supportedChannels.erase(std::remove_if(supportedChannels.begin(), supportedChannels.end(),
                                               [&](const int64_t channels) {
                                                   for (auto wlChannels : workloadsChannels) {
                                                       if (wlChannels % channels != 0) {
                                                           return true;
                                                       }
                                                   }
                                                   return false;
                                               }),
                                supportedChannels.end());
    }

    VPUX_THROW_WHEN(supportedChannels.empty() && (depthwiseCorrection || sparsityCorrection),
                    "Unable to find supported channels");

    return supportedChannels;
}

// Get offset from start of the cluster
SmallVector<Shape> getPerClusterOffsetsCorrection(VPU::NCEOpInterface nceOp) {
    auto nceClusterTilingOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(nceOp->getParentOp());

    if (nceClusterTilingOp == nullptr) {
        return {};
    }

    auto outputType = nceClusterTilingOp->getResult(0).getType();
    auto distributedOut = outputType.dyn_cast<VPU::DistributedTensorType>();
    if (distributedOut == nullptr) {
        return {};
    }

    // TODO: E#73931
    // PermuteQuantize output will always have memory and compute equal for now.
    return distributedOut.getPerClusterMemoryShapeOffsets();
}

// Splits the workload channels so that they are composed out of the values in the `supportedChannels` array, if it is
// provided. Additionally, removes the padding and spatial offsets from the workload based on the `removePadding` flag
void splitWorkload(VPU::DPUWorkloadOp dpuWorkloadOp, ArrayRef<int64_t> supportedChannels, const bool removePadding,
                   ArrayRef<Shape> offsetsCorrectionForPermuteQuantize, Logger log) {
    auto wlSizes = parseIntArrayAttr<int64_t>(dpuWorkloadOp.outSizesAttr());
    auto wlOffsets = parseIntArrayAttr<int64_t>(dpuWorkloadOp.outOffsetsAttr());
    auto padsAttr = dpuWorkloadOp.pad();
    if (removePadding) {
        const auto pads = dpuWorkloadOp.pad();
        const auto top = pads.top().getInt();
        const auto bottom = pads.bottom().getInt();
        const auto left = pads.left().getInt();
        const auto right = pads.right().getInt();
        wlSizes[Dims4D::Act::H.ind()] -= (top + bottom);
        wlSizes[Dims4D::Act::W.ind()] -= (left + right);
        if (!offsetsCorrectionForPermuteQuantize.empty()) {
            const auto clusterId = dpuWorkloadOp.cluster_id().getValue();
            const auto offsetsCorrectionPerCluster = offsetsCorrectionForPermuteQuantize[clusterId].raw();
            std::transform(wlOffsets.begin(), wlOffsets.end(), offsetsCorrectionPerCluster.begin(), wlOffsets.begin(),
                           std::minus<int64_t>());
            log.nest().trace("Applied offset correction {0}", offsetsCorrectionForPermuteQuantize);
        }

        padsAttr = VPU::getPaddingAttr(pads.getContext(), PadInfo(0, 0, 0, 0));

        log.nest().trace("Removed padding from workload '{0}'", dpuWorkloadOp);
    }

    SmallVector<int64_t> newWorkloadChannels;
    auto wlChannels = wlSizes[Dims4D::Act::C.ind()];
    if (supportedChannels.empty()) {
        newWorkloadChannels.push_back(wlChannels);
    } else {
        for (auto supportedChannel : supportedChannels) {
            while (wlChannels >= supportedChannel) {
                newWorkloadChannels.push_back(supportedChannel);
                wlChannels -= supportedChannel;
            }
        }
    }

    mlir::OpBuilder builder(dpuWorkloadOp);
    auto channelOffset = wlOffsets[Dims4D::Act::C.ind()];

    for (auto channelSize : newWorkloadChannels) {
        auto sizes = wlSizes;
        sizes[Dims4D::Act::C.ind()] = channelSize;

        auto offsets = wlOffsets;
        offsets[Dims4D::Act::C.ind()] = channelOffset;
        channelOffset += channelSize;

        const auto offsetsAttr = getIntArrayAttr(builder.getContext(), offsets);
        const auto sizesAttr = getIntArrayAttr(builder.getContext(), sizes);

        builder.create<VPU::DPUWorkloadOp>(dpuWorkloadOp.getLoc(), offsetsAttr, sizesAttr, padsAttr,
                                           dpuWorkloadOp.mpe_modeAttr(), dpuWorkloadOp.cluster_idAttr());
    }

    log.nest().trace("Split workload of size '{0}' into '{1}'", wlSizes[Dims4D::Act::C.ind()], newWorkloadChannels);

    dpuWorkloadOp.erase();
}

//
// CorrectNCEWorkloads
//

class CorrectNCEWorkloadsPass final : public CorrectNCEWorkloadsBase<CorrectNCEWorkloadsPass> {
public:
    explicit CorrectNCEWorkloadsPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void CorrectNCEWorkloadsPass::safeRunOnFunc() {
    auto func = getOperation();

    mlir::DenseSet<mlir::Operation*> handledNCEOps;

    func.walk([&](VPU::NCEOpInterface nceOp) {
        if (handledNCEOps.contains(nceOp)) {
            return;
        }

        // More than one operation might need to be handled at the same time for some sparse activations,
        // to satisfy the requirements of the consumer ops
        mlir::DenseSet<mlir::Operation*> producerNCEOps{nceOp};
        const auto invalidSparseOps = findInvalidSparseOps(nceOp);
        if (!invalidSparseOps.empty()) {
            producerNCEOps.clear();
            producerNCEOps.insert(invalidSparseOps.begin(), invalidSparseOps.end());
        }

        const auto invalidDepthwiseOps = findInvalidDepthwiseOps(producerNCEOps);
        const auto invalidPermuteQuantizeOps = findInvalidPermuteQuantizeOps(producerNCEOps);
        if (invalidSparseOps.empty() && invalidDepthwiseOps.empty() && invalidPermuteQuantizeOps.empty()) {
            return;
        }

        const auto supportedChannels =
                getSupportedChannels(producerNCEOps, /*depthwiseCorrection=*/!invalidDepthwiseOps.empty(),
                                     /*sparsityCorrection=*/!invalidSparseOps.empty());

        int64_t opIdx = 1;
        for (auto op : producerNCEOps) {
            auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
            VPUX_THROW_UNLESS(nceOp != nullptr, "Expected NCE op, got '{0}'", op);

            auto isInvalidDepthwise = invalidDepthwiseOps.contains(op);
            auto isInvalidSparsity = invalidSparseOps.contains(op);
            auto isInvalidPermuteQuantizeOp = invalidPermuteQuantizeOps.contains(op);
            _log.trace("Correcting workloads for operation '{0}' at '{1}'. Necessary corrections: depthwise "
                       "'{2}', sparsity '{3}' ({4}/{5}), remove padding '{6}'",
                       op->getName(), op->getLoc(), isInvalidDepthwise, isInvalidSparsity, opIdx++,
                       producerNCEOps.size(), isInvalidPermuteQuantizeOp);

            const auto offsetsCorrectionForPermuteQuantize = getPerClusterOffsetsCorrection(nceOp);
            auto workloads = nceOp.workloads().getOps<VPU::DPUWorkloadOp>();
            for (auto workloadOp : llvm::make_early_inc_range(workloads)) {
                const auto wlSizes = parseIntArrayAttr<int64_t>(workloadOp.outSizes());
                auto wlChannels = wlSizes[Dims4D::Act::C.ind()];
                if (llvm::find(supportedChannels, wlChannels) != supportedChannels.end()) {
                    continue;
                }

                splitWorkload(workloadOp, supportedChannels, /*removePadding=*/isInvalidPermuteQuantizeOp,
                              offsetsCorrectionForPermuteQuantize, _log);
            }

            handledNCEOps.insert(op);
        }
    });
}

}  // namespace

//
// createCorrectNCEWorkloadsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createCorrectNCEWorkloadsPass(Logger log) {
    return std::make_unique<CorrectNCEWorkloadsPass>(log);
}
