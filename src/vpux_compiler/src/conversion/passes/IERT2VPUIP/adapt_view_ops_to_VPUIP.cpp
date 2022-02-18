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

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

using namespace vpux;

namespace {

//
// AdaptViewOps2VPUIPPass
//

class AdaptViewOps2VPUIPPass final : public AdaptViewOps2VPUIPBase<AdaptViewOps2VPUIPPass> {
public:
    explicit AdaptViewOps2VPUIPPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    int64_t calculateStaticOffsetWithStrides(ArrayRef<int64_t> subViewStaticOffsets, StridesRef subViewStrides) const;
    bool invariantAddressNeedsVariantAlignment(int64_t baseOffset, int64_t staticOffsetWithStrides,
                                               StridesRef subViewStrides, size_t cmxSize) const;
};

int64_t AdaptViewOps2VPUIPPass::calculateStaticOffsetWithStrides(ArrayRef<int64_t> subViewStaticOffsets,
                                                                 StridesRef subViewStrides) const {
    Byte offset(0);

    for (auto p : zip(subViewStaticOffsets, subViewStrides)) {
        offset += Byte(std::get<0>(p) * std::get<1>(p));
    }

    return offset.count();
}

bool AdaptViewOps2VPUIPPass::invariantAddressNeedsVariantAlignment(int64_t baseOffset, int64_t staticOffsetWithStrides,
                                                                   StridesRef subViewStrides, size_t cmxSize) const {
    Byte cmxLeft(static_cast<int64_t>(cmxSize) - (baseOffset + staticOffsetWithStrides));

    for (auto stride : subViewStrides) {
        if (Byte(stride) > cmxLeft) {
            return true;
        }
    }

    return false;
}

void AdaptViewOps2VPUIPPass::safeRunOnFunc() {
    // for concatenation in NNCMX with non contigious block memory write
    // prevent a scenario where tensor strides exceed NNCMX size
    // move static offsets from invariants to variants
    auto func = getFunction();
    auto& aliasInfo = getAnalysis<AliasesInfo>();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto availableMem = IE::getAvailableMemory(module, VPU::MemoryKind::CMX_NN);
    const auto cmxSize = static_cast<size_t>(availableMem.size().count());

    func->walk([&](IERT::SubViewOp subView) {
        auto source = aliasInfo.getSource(subView.result());
        if (source == nullptr) {
            return;
        }

        auto declareOp = mlir::dyn_cast_or_null<VPURT::DeclareBufferOp>(source.getDefiningOp());
        if (declareOp == nullptr) {
            return;
        }

        if (declareOp.section() != VPURT::BufferSection::CMX_NN) {
            return;
        }

        const auto subViewStaticOffsets = parseIntArrayAttr<int64_t>(subView.static_offsets());
        const auto subViewStrides = getStrides(subView.source());
        VPUX_THROW_UNLESS(subViewStrides.size() == subViewStaticOffsets.size(),
                          "SubView offsets '{0}' doesn't match strides '{1}'", subViewStaticOffsets, subViewStrides);

        auto baseOffset = declareOp.byteOffset();
        auto staticOffsetWithStrides = calculateStaticOffsetWithStrides(subViewStaticOffsets, subViewStrides);
        if (!invariantAddressNeedsVariantAlignment(baseOffset, staticOffsetWithStrides, subViewStrides, cmxSize)) {
            return;
        }

        _log.trace("Got '{0}' at '{1}' that requires adjustment", subView->getName(), subView->getLoc());
        _log.nest(1).trace("Subview type = '{0}'", subView.getType());
        _log.nest(1).trace("Subview static offsets = '{0}'", subView.static_offsets());
        _log.nest(1).trace("Subview strides = '{0}'", subViewStrides);

        _log.nest(1).trace("Base offset = '{0}'", baseOffset);
        _log.nest(1).trace("Offset difference = '{0}'", staticOffsetWithStrides);

        // Workloads workload_x, workload_y, workload_z
        // X -> W
        // Y -> H
        // Z -> C
        // offsets stored as NCHW so reverse order

        size_t workloadDimension = 0;
        auto offsetsSize = subViewStaticOffsets.size();
        for (size_t idx = 0; idx < offsetsSize; idx++) {
            if (subViewStaticOffsets[idx] != 0) {
                workloadDimension = offsetsSize - 1 - idx;
            }
        }
        _log.nest(1).trace("Workload dimension = '{0}'", workloadDimension);

        ValueOrderedSet convertedResults;
        // change workloads in the parent op
        for (auto user : subView.result().getUsers()) {
            if (auto nceClusterTask = mlir::dyn_cast_or_null<VPUIP::NCEClusterTaskOp>(user)) {
                if (convertedResults.find(nceClusterTask.getResult(0)) != convertedResults.end()) {
                    // skip duplicate change for ops with multiple subview uses
                    continue;
                }
                auto dpuVariants = nceClusterTask.variants().getOps<VPUIP::DPUTaskOp>();
                for (auto dpu : dpuVariants) {
                    _log.nest(2).trace("Adjusting workloads of variant '{0}'", dpu->getName());
                    _log.nest(3).trace("Original start workloads '{0}'", dpu.start());
                    _log.nest(3).trace("Original end workloads '{0}'", dpu.end());

                    auto start = parseIntArrayAttr<int32_t>(dpu.start());
                    auto end = parseIntArrayAttr<int32_t>(dpu.end());
                    // add offset to the correct dimension
                    start[workloadDimension] += staticOffsetWithStrides;
                    end[workloadDimension] += staticOffsetWithStrides;
                    // update start and end
                    dpu->setAttr(dpu.startAttrName(), getIntArrayAttr(subView.getContext(), start));
                    dpu->setAttr(dpu.endAttrName(), getIntArrayAttr(subView.getContext(), end));

                    _log.nest(3).trace("Updated start workloads '{0}'", dpu.start());
                    _log.nest(3).trace("Updated end workloads '{0}'", dpu.end());
                }
                convertedResults.insert(nceClusterTask.getResult(0));
            }
        }

        // modify static offsets from the slice to 0s
        SmallVector<int64_t> staticOffsets(subViewStaticOffsets.size(), 0);
        auto staticOffsetsAttr = getIntArrayAttr(subView.getContext(), staticOffsets);
        subView->setAttr(subView.static_offsetsAttrName(), staticOffsetsAttr);
        _log.nest(1).trace("Subview new static offsets = '{0}'", subView.static_offsets());
        _log.trace("SubView '{0}' at '{1}' was adjusted", subView->getName(), subView->getLoc());
    });
}

}  // namespace

//
// createAdaptViewOps2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createAdaptViewOps2VPUIPPass(Logger log) {
    return std::make_unique<AdaptViewOps2VPUIPPass>(log);
}
