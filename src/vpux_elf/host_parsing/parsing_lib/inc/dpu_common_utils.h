#pragma once

#include <data_types.h>
#include <nce_2p7_hw.h>

namespace parsing_lib {
unsigned int SOH_LinesPerCluster(unsigned int parentHeight, unsigned int height, unsigned int clusters);

// void Update_Invariant_SOH(DPULayerTypes opType, DPUInvariant &invariant,
//                           common_runtime::RelativeAddress &input,
//                           const common_runtime::NNRelocationData &relocationData);
bool Is_Dtype_Mix_Supported(DType inputType, DType weightsType);

} // namespace parsing_lib
