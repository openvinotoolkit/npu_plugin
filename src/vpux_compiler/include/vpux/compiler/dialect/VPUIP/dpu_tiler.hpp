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

#include "attributes.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#pragma once

namespace vpux {
namespace VPUIP {

struct DpuTile final {
    SmallVector<int64_t> start;
    SmallVector<int64_t> end;
    int64_t padLeft;
    int64_t padRight;
    int64_t padTop;
    int64_t padBottom;
    VPU::MPEMode mpeMode;
};

class DpuTiler final {
public:
    DpuTiler(const ShapeRef& outShape, SmallVector<VPU::MPEMode> mpeModeList)
            : _outShape(outShape), _mpeModeList(mpeModeList) {
    }

    bool generateSplitNumberPool(int64_t numDPU, uint32_t maxSplits = 50, SmallVector<uint32_t> validZTiles = {});
    bool tileOverH(int64_t numDPU, PadInfo padInfo, VPU::MPEMode mpeMode);
    bool tileOverZ(uint32_t splitNumber, PadInfo padInfo, SmallVector<uint32_t> validZTiles = {}, bool sparse = false,
                   bool has_se = false);
    SmallVector<SmallVector<DpuTile>> getSplitPool();
    SmallVector<uint32_t> getSplitNumberPool();

#ifdef __linux__
    uint32_t cost(VPUIP::NCEClusterTaskOp op, const SmallVector<VPUIP::DpuTile>& dpuTiles, unsigned int numDPU,
                  VPU::ArchKind arch);
#endif

private:
    Shape selectPadding(ShapeRef original);
    SmallVector<std::pair<uint8_t, uint8_t>> getModes();

    ShapeRef _outShape;
    SmallVector<VPU::MPEMode> _mpeModeList;
    SmallVector<uint32_t> _splitNumberPool;
    SmallVector<SmallVector<DpuTile>> _splitPool;
};

}  // namespace VPUIP
}  // namespace vpux
