#ifndef MV_NCE2_TASK_API_
#define MV_NCE2_TASK_API_

#include "include/mcm/target/keembay/types/nce2_dma_direction.hpp"
#include "include/mcm/computation/model/base_op_model.hpp"

namespace mv
{
    Data::TensorIterator createDPUTask(BaseOpModel &om, Data::OpListIterator opIt, const std::string& name = "");
    Data::TensorIterator createDPUTask(BaseOpModel& om, const std::vector<Data::TensorIterator>& inputs, const std::string& opType, std::vector<std::pair<std::string, mv::Attribute>>& list, const std::string& name = "");
    Data::TensorIterator createDMATask(BaseOpModel& om, Data::TensorIterator data0, DmaDirection direction, const std::string& name = "");
    Data::TensorIterator createBarrierTask(BaseOpModel& om, const std::string& name = "");
}

#endif
