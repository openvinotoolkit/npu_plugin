//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

void removeOutputSparse(VPU::NCEOpInterface origOp, Logger log) {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    if (clusteredOp == nullptr || !clusteredOp.getMultiClusterStrategy().has_value()) {
        return;
    }

    const auto strategy = clusteredOp.getMultiClusterStrategy().value();
    const auto outputTensorType = clusteredOp->getResult(0).getType().cast<vpux::NDTypeInterface>();

    if (strategy != VPU::MultiClusterStrategy::SplitOverKernel) {
        return;
    }

    if (auto sparseOutputType = outputTensorType.dyn_cast<VPU::SparseTensorType>()) {
        VPUX_THROW_UNLESS(sparseOutputType.getSparsityMap() != nullptr, "Missing sparsity map from sparse type {0}",
                          sparseOutputType);
        VPUX_THROW_UNLESS(sparseOutputType.getStorageElementTable() == nullptr,
                          "Dynamically populated storage element table is not supported");

        const auto numClusters =
                VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
        const auto distributedDataType =
                getDistributedOutputTypeFromOp(clusteredOp, sparseOutputType.getData(), numClusters,
                                               /*inputType*/ nullptr, /*hasExplicitDistributedAttr*/ false);
        const auto distributedTensorType = mlir::cast<vpux::VPU::DistributedTensorType>(distributedDataType);
        auto users = to_small_vector(clusteredOp->getUsers());

        auto recursivelyRemoveSparseOutput = [&](VPU::ClusteredOpInterface clusteredOp) -> void {
            clusteredOp->getResult(0).setType(sparseOutputType.getData());
            log.nest().trace("Remove output sparsity for op {0} at {1}", clusteredOp->getName(), clusteredOp->getLoc());

            auto users = to_small_vector(clusteredOp->getUsers());
            while (!users.empty()) {
                auto currentOp = users.back();
                users.pop_back();
                if (mlir::isa_and_nonnull<vpux::VPU::ViewLikeOpInterface>(currentOp)) {
                    vpux::inferReturnTypes(currentOp, vpux::InferShapedTypeMode::ALL);
                    auto nextOps = to_small_vector(currentOp->getUsers());
                    users.insert(users.end(), nextOps.begin(), nextOps.end());
                }
            }
        };

        // Removes SOK layer's output sparsity if SOK layer has different split sizes on clusters excluding the last
        // one. For example, we need to split OC = 128 on 6 tiles, the tiled size will be {32, 32, 16, 16, 16, 16}.
        // If there's output sparsity, we need to split 32 into two pieces of 16 because we must have the same
        // workload channel excluding the last one. However, two workloads with 16 channels have much worse
        // performance than a workload with 32 channels. If there's no sparsity, we can keep the workload with 32
        // channels.
        if (distributedTensorType.getDistribution().getUniformDistributedSegments() != nullptr) {
            recursivelyRemoveSparseOutput(clusteredOp);
            return;
        }

        // Removes SOK layer's output sparsity if SOK layer's output is used by `VPU.Concat`.
        //
        // Conv1_1 (OC = 256, SOK)  Conv1_2 (OC = 256, SOK)
        //       \                               /
        //                   Concat on C
        //                        |
        //                      Conv2
        //
        // Take above graph as an example, we need to split OC = 256 on 6 tiles, the tiled size will be {48, 48, 48,
        // 48, 48, 16}. After concatenation, the combined workloads will be {48, 48, 48, 48, 48, 16, 48, 48, 48, 48,
        // 48, 16}. If there's output sparsity for Conv1_1 and Conv1_2, we need to split 48 into three pieces of 16
        // because we must have the same workload channel excluding the last one. If there's no sparsity, we can
        // keep the workload with 48 channels.
        if (llvm::find_if(users, [](const mlir::Operation* op) {
                if (auto concatOp = mlir::dyn_cast_or_null<VPU::ConcatOp>(op)) {
                    const auto outputType = concatOp.getOutput().getType().cast<NDTypeInterface>();
                    const auto outputShape = outputType.getShape();
                    const auto inputDataType = concatOp.getInputs().front().getType().cast<NDTypeInterface>();
                    const auto inputShape = inputDataType.getShape();

                    if (inputShape[Dims4D::Act::C] != outputShape[Dims4D::Act::C]) {
                        return true;
                    }
                }

                return false;
            }) != users.end()) {
            recursivelyRemoveSparseOutput(clusteredOp);
            return;
        }
    }
}

//
// RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass
//
class RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass final :
        public VPU::RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPassBase<
                RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass> {
public:
    explicit RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

//
// safeRunOnFunc
//
void RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass::safeRunOnFunc() {
    auto func = getOperation();

    // TODO: E#106239
    // This pass could remove activation sparsity after strategy manager. With these changes, multi-clustering and
    // tiling are done with the cost of activation sparsity being present while sparsity can be reverted. This can have
    // an impact over the performance. Hopefully in the future we can look into refactoring the strategy manager to also
    // take the decision on whether to enable activation sparsity or not.
    func->walk([&](VPU::NCEOpInterface op) {
        removeOutputSparse(op, _log);
    });
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createRemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass(Logger log) {
    return std::make_unique<RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass>(log);
}
