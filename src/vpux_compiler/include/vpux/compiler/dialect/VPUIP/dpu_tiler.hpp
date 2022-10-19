//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <vpu_cost_model.h>

#include <set>
#include <tuple>

namespace vpux {
namespace VPUIP {

struct WorkloadCostParams {
    VPUIP::NCETaskType nceTaskType;
    mlir::Type dataType;
    VPU::ArchKind arch;
    Shape fullInputShape;
    Shape inputShape;
    Shape outputShape;
    PadInfo padInfo;
    int64_t numDPU;
    SmallVector<int64_t> kernelSize;
    SmallVector<int64_t> kernelStride;
};

enum class SplitDimension { SPLIT_OVER_H = 0, SPLIT_OVER_W = 1, SPLIT_OVER_HW = 2 };

StringLiteral stringifyEnum(SplitDimension splitDimension);

using WorkloadTile = std::tuple<TileInfo, VPU::MPEMode>;
using WorkloadSplit = SmallVector<WorkloadTile>;
using WorkloadSplitPool = std::set<WorkloadSplit>;

class DpuTiler final {
public:
    DpuTiler(ShapeRef outShape, VPU::MPEMode mpeMode);

public:
    SmallVector<int64_t> generateSplitNumberPool(int64_t numDPU, int64_t maxSplits) const;

    void tileOverH(int64_t numDPU, WorkloadSplitPool& splitPool);
    void tileOverZ(int64_t splitNumber, WorkloadSplitPool& splitPool);
    void tileOverHW(int64_t splitNumber, SplitDimension splitDimension, WorkloadSplitPool& splitPool);
    void tileOverHWMixedPrecision(WorkloadSplitPool& splitPool);

private:
    Shape _outShape;
    VPU::MPEMode _mpeMode;
};

int64_t computeSplitCost(const WorkloadSplit& split, const WorkloadCostParams& params,
                         const std::shared_ptr<VPUNN::VPUCostModel>& costModel);

}  // namespace VPUIP
}  // namespace vpux
