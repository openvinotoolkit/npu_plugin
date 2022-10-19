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

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;
using namespace VPU;

namespace {

constexpr std::array<int64_t, 3> supportedChannels = {64, 32, 16};

void splitWorkload(VPU::NCEOpInterface nceOp, VPU::DPUWorkloadOp dpuWorkloadOp, Logger log) {
    const auto wlSizes = parseIntArrayAttr<int64_t>(dpuWorkloadOp.sizesAttr());
    const auto wlOffsets = parseIntArrayAttr<int64_t>(dpuWorkloadOp.offsetsAttr());

    auto wlChannels = wlSizes[Dims4D::Act::C.ind()];

    SmallVector<int64_t> newWorkloadChannels;
    for (auto supportedChannel : supportedChannels) {
        while (wlChannels >= supportedChannel) {
            newWorkloadChannels.push_back(supportedChannel);
            wlChannels -= supportedChannel;
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

        builder.create<VPU::DPUWorkloadOp>(nceOp.getLoc(), offsetsAttr, sizesAttr, dpuWorkloadOp.pad(),
                                           dpuWorkloadOp.mpe_modeAttr(), dpuWorkloadOp.cluster_idAttr());
    }

    log.trace("Splited '{0}' operation original workload size of {1} into {2}", nceOp.getLoc(),
              wlSizes[Dims4D::Act::C.ind()], newWorkloadChannels);

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
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto arch = VPU::getArch(module);
    if (arch != VPU::ArchKind::VPUX37XX) {
        return;
    }

    func.walk([&](VPU::NCEOpInterface nceOp) {
        if (!mlir::isa<VPU::NCEDepthConvolutionOp, VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp>(nceOp))
            return;

        auto workloads = nceOp.workloads().getOps<VPU::DPUWorkloadOp>();
        auto workloadIt = workloads.begin();
        while (workloadIt != workloads.end()) {
            auto nextWorkloadIt = workloadIt;
            ++nextWorkloadIt;

            auto dpuWorkloadOp = *workloadIt;

            const auto wlSizes = parseIntArrayAttr<int64_t>(dpuWorkloadOp.sizes());
            auto wlChannels = wlSizes[Dims4D::Act::C.ind()];
            if (llvm::find(supportedChannels, wlChannels) != supportedChannels.end()) {
                workloadIt = nextWorkloadIt;
                continue;
            }

            splitWorkload(nceOp, dpuWorkloadOp, _log);

            workloadIt = nextWorkloadIt;
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
