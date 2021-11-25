#include "vpux/compiler/utils/custom_pwl_table.hpp"

namespace vpux {

PWLTableMap* customPWLTable_leakyRelu() {
    PWLTableType pwl_type;
    pwl_type.activation = "IE.LeakyRelu";
    pwl_type.dtype = mlir::quant::UniformQuantizedType();

    std::vector<int> vecRange{-128, -109, -90, -72, -54, -36, -18, 0, 128};
    std::vector<int> vecShift{1, -1, 0, 0, 0, -1, -1, -4};
    std::vector<int> vecBias{-119, 44, -43, -31, -19, 18, 10, 0};
    double range_min = -65504.0, range_max = 65504.0;
    int post_shift = 4;
    PWLTableEntry pwl_table = {vecRange, vecShift, vecBias, std::make_pair(range_min, range_max), post_shift};
    std::vector<PWLTableEntry> vecTable;
    vecTable.push_back(pwl_table);

    PWLTableMap* pwlTables = new PWLTableMap();
    pwlTables->insert({pwl_type, vecTable});
    return pwlTables;
}

}  // namespace vpux