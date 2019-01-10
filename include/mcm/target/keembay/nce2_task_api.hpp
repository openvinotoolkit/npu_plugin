#ifndef MV_NCE2_TASK_API_
#define MV_NCE2_TASK_API_

#include "include/mcm/target/keembay/nce2_dma_direction.hpp"

namespace mv
{
    Data::TensorIterator createDPUTask(const std::vector<Data::TensorIterator>& inputs, const std::string& opType, const std::string& name = "");
    Data::TensorIterator createDMATask(Data::TensorIterator data0, DmaDirection direction, const std::string& name = "");
}

#endif
