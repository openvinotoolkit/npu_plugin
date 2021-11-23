#include "vpux/compiler/utils/custom_pwl_table.hpp"

namespace vpux {

vpux::PWLTableEntry getLeakyReluPWLEntry(mlir::Type outElemType) {
    PWLTableType leakyType;
    leakyType.activation = llvm::StringRef("IE.LeakyRelu");
    leakyType.dtype = outElemType;

    const SmallVector<int> range{-128, -109, -90, -72, -54, -36, -18, 0, 128};
    const SmallVector<int> shift{1, -1, 0, 0, 0, -1, -1, -4};
    const SmallVector<int> bias{-119, 44, -43, -31, -19, 18, 10, 0};
    const double rangeMin = -65504.0, rangeMax = 65504.0;
    const int postShift = 4;

    return PWLTableEntry{range, shift, bias, std::make_pair(rangeMin, rangeMax), postShift};
}

Optional<vpux::PWLTableEntry> findCustomPWLTable(const llvm::StringRef activationName, mlir::Type outElemType) {
    // create map
    // this create map is a temporary solution, it will be change in a future MR when we will decide if we will add
    // custom tables and compilation train tables to MLIR or an analysis. See:
    // https://github.com/intel-innersource/applications.ai.vpu-accelerators.vpux-plugin/pull/388

    static PWLTableMap pwlTableMap{
            {PWLTableType{llvm::StringRef("IE.LeakyRelu"), mlir::quant::UniformQuantizedType()},
             SmallVector<PWLTableEntry>{getLeakyReluPWLEntry(mlir::quant::UniformQuantizedType())}}};

    // find
    PWLTableType pwlType;
    pwlType.activation = activationName;
    if (outElemType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        pwlType.dtype = mlir::quant::UniformQuantizedType();
    } else {
        return None;
    }

    auto pwlTableIt = pwlTableMap.find(pwlType);
    if (pwlTableIt == pwlTableMap.end()) {
        return None;
    }
    return pwlTableIt->second.data()[0];
}

}  // namespace vpux
