//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <vpu_cost_model.h>
#include <set>
#pragma once

namespace vpux {
namespace VPUIP {

struct WorkloadCostParams {
    bool isTileOverZSupported;
    VPUIP::NCETaskType nceTaskType;
    mlir::Type dataType;
    VPU::ArchKind arch;
    VPU::MPEMode mpeMode;
    Shape inputShape;
    Shape outputShape;
    PadInfo padInfo;
    int64_t numDPU;
    SmallVector<int64_t> kernelSize;
    SmallVector<int64_t> kernelStride;
};

enum class SplitDimension : int32_t { SPLIT_OVER_H = 0, SPLIT_OVER_W = 1, SPLIT_OVER_HW = 2 };

class DpuTiler final {
public:
    DpuTiler(ShapeRef outShape, VPU::MPEMode mpeMode, std::shared_ptr<VPUNN::VPUCostModel> costModel)
            : _outShape(outShape.raw()), _mpeMode(mpeMode), _costModel(costModel) {
    }

    SmallVector<uint32_t> generateSplitNumberPool(int64_t numDPU, uint32_t maxSplits);
    void tileOverH(int64_t numDPU);
    void tileOverZ(uint32_t splitNumber);
    void tileOverHW(uint32_t splitNumber, const SplitDimension splitDimension);
    std::set<OutputTiling> getSplitPool();

    uint32_t cost(const OutputTiling& dpuTiles, const WorkloadCostParams& params);
    double simpleCost(const OutputTiling& dpuTiles, const WorkloadCostParams& params);

private:
    std::pair<uint8_t, uint8_t> getMode();

    Shape _outShape;
    VPU::MPEMode _mpeMode;
    std::shared_ptr<VPUNN::VPUCostModel> _costModel;
    std::set<OutputTiling> _splitPool;
};

StringLiteral stringifySplitDimension(SplitDimension splitDimension);

}  // namespace VPUIP
}  // namespace vpux
