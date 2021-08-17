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
    int64_t padLeft;
    int64_t padRight;
    int64_t padTop;
    int64_t padBottom;
};

class DpuTiler final {
public:
    static SmallVector<DpuTile> tileOverH(int64_t numDPU, ShapeRef outShape, int64_t padLeft, int64_t padRight,
                                          int64_t padTop, int64_t padBottom);
};

}  // namespace VPUIP
}  // namespace vpux
