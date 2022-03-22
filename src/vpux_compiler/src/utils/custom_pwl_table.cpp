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

#include "vpux/compiler/utils/custom_pwl_table.hpp"

namespace vpux {

vpux::PWLTableEntry getLeakyReluPWLEntry() {
    const SmallVector<int32_t> range{-128, -109, -90, -72, -54, -36, -18, 0, 128};
    const SmallVector<int32_t> shift{1, -1, 0, 0, 0, -1, -1, -4};
    const SmallVector<int32_t> bias{-119, 44, -43, -31, -19, 18, 10, 0};
    const double rangeMin = -65504.0, rangeMax = 65504.0;
    const int32_t postShift = 4;
    static const PWLTableEntry leakyReluEntry{range, shift, bias, std::make_pair(rangeMin, rangeMax), postShift};
    return leakyReluEntry;
}

Optional<vpux::PWLTableEntry> findCustomPWLTable(StringRef activationName, mlir::Type outElemType) {
    // create map
    // this create map is a temporary solution, it will be change in a future MR when we will decide if we will add
    // custom tables and compilation train tables to MLIR or an analysis. See:
    // https://github.com/intel-innersource/applications.ai.vpu-accelerators.vpux-plugin/pull/388

    static const PWLTableMap pwlTableMap{
            {PWLTableType{SmallString("IE.LeakyRelu"), mlir::quant::UniformQuantizedType()},
             SmallVector<PWLTableEntry>{getLeakyReluPWLEntry()}}};

    // find
    if (!outElemType.isa<mlir::quant::UniformQuantizedType>()) {
        return None;
    }
    const PWLTableType pwlType = {activationName, mlir::quant::UniformQuantizedType()};

    auto pwlTableIt = pwlTableMap.find(pwlType);
    if (pwlTableIt == pwlTableMap.end()) {
        return None;
    }
    return pwlTableIt->second.data()[0];
}

}  // namespace vpux
