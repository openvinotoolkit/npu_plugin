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

#include "vpux/compiler/core/attributes/shape.hpp"

#pragma once

namespace vpux {
namespace VPUIP {

struct DpuTile final {
    SmallVector<int64_t> start;
    SmallVector<int64_t> end;
    SmallVector<int64_t> padsBegin;
    SmallVector<int64_t> padsEnd;
};

class DpuTiler final {
public:
    static SmallVector<DpuTile> tileOverH(uint32_t numDPU, ShapeRef outShape, ArrayRef<int64_t> opPadsBegin,
                                          ArrayRef<int64_t> opPadsEnd);
};

}  // namespace VPUIP
}  // namespace vpux
