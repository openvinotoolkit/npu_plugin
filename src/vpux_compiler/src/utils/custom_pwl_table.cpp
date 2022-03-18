#include "vpux/compiler/utils/custom_pwl_table.hpp"

namespace vpux {

vpux::PWLTableEntry getLeakyReluPWLEntry() {
    const SmallVector<int64_t> range{-128, -109, -90, -72, -54, -36, -18, 0, 128};
    const SmallVector<int64_t> shift{1, -1, 0, 0, 0, -1, -1, -4};
    const SmallVector<int64_t> bias{-119, 44, -43, -31, -19, 18, 10, 0};
    const double rangeMin = -65504.0, rangeMax = 65504.0;
    const int64_t postShift = 4;

    return PWLTableEntry{range, shift, bias, std::make_pair(rangeMin, rangeMax), postShift};
}

Optional<vpux::PWLTableEntry> findCustomPWLTable(StringRef activationName, mlir::Type outElemType) {
    // create map
    // this create map is a temporary solution, it will be change in a future MR when we will decide if we will add
    // custom tables and compilation train tables to MLIR or an analysis. See:
    // https://github.com/intel-innersource/applications.ai.vpu-accelerators.vpux-plugin/pull/388

    static PWLTableMap pwlTableMap{{PWLTableType{StringRef("IE.LeakyRelu"), mlir::quant::UniformQuantizedType()},
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
