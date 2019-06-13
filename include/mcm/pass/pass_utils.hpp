#ifndef PASS_UTILS_HPP_
#define PASS_UTILS_HPP_

#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

namespace mv
{
    std::vector<std::pair<mv::Data::OpListIterator,size_t>> getOutputDataFlow(mv::OpModel& om, mv::Data::OpListIterator &opIt, bool deleteOp = true);
    void setOutputDataFlow(mv::OpModel& om, mv::Data::TensorIterator &dpuTaskOutputTensor, const std::vector<std::pair<mv::Data::OpListIterator,size_t>>& outDataFlows);
}

#endif // PASS_UTILS_HPP_
