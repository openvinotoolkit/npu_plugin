#ifndef PASS_UTILS_HPP_
#define PASS_UTILS_HPP_

#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"

namespace mv
{
    std::vector<std::pair<mv::Data::OpListIterator,size_t>> getOutputDataFlow(mv::OpModel& om, mv::Data::OpListIterator &opIt, bool deleteOp = true);
    void setOutputDataFlow(mv::OpModel& om, mv::Data::TensorIterator &dpuTaskOutputTensor, const std::vector<std::pair<mv::Data::OpListIterator,size_t>>& outDataFlows);
    std::vector<mv::Control::OpListIterator> getInputControlFlow(mv::ControlModel& om, mv::Control::OpListIterator opIt);
    std::vector<mv::Control::OpListIterator> getOutputControlFlow(mv::ControlModel& om, mv::Control::OpListIterator opIt);
    void setInputControlFlow(mv::ControlModel& cm, mv::Control::OpListIterator op, const std::vector<mv::Control::OpListIterator>& inputControlFlows);
    void setOutputControlFlow(mv::ControlModel& cm, mv::Control::OpListIterator op, const std::vector<mv::Control::OpListIterator>& outputControlFlows);
    mv::Data::OpListIterator linkNewOperationsRemove(mv::Data::OpListIterator parentOpIt, mv::Data::TensorIterator sourceTensor, mv::OpModel om, mv::Data::OpListIterator opIt);
}
void calcZeroPointAndScalePerTensor(double outputMax,  double outputMin, double& outScale, int64_t& outZp);
void updateInfMinMaxPerTensor(mv::Data::TensorIterator tensor);
void updateInfMinMaxPerChannel(mv::Data::TensorIterator tensor);
void provideAccuracyinPPEs(mv::ComputationModel& model);

#endif // PASS_UTILS_HPP_
