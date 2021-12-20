#include "vpux/compiler/utils/custom_pwl_table.hpp"

namespace vpux {

PWLTableMap* customPWLTable_leakyRelu() {
    PWLTableType pwlType;
    pwlType.activation = "IE.LeakyRelu";
    pwlType.dtype = mlir::quant::UniformQuantizedType();

    std::vector<int> vecRange{-128, -109, -90, -72, -54, -36, -18, 0, 128};
    std::vector<int> vecShift{1, -1, 0, 0, 0, -1, -1, -4};
    std::vector<int> vecBias{-119, 44, -43, -31, -19, 18, 10, 0};
    double rangeMin = -65504.0, rangeMax = 65504.0;
    int postShift = 4;
    PWLTableEntry pwlTable = {vecRange, vecShift, vecBias, std::make_pair(rangeMin, rangeMax), postShift};
    std::vector<PWLTableEntry> vecTable;
    vecTable.push_back(pwlTable);

    PWLTableMap* pwlTables = new PWLTableMap();
    pwlTables->insert({pwlType, vecTable});
    return pwlTables;
}

}  // namespace vpux