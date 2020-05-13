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
    mv::Data::OpListIterator linkNewOperationsReplacement(mv::Data::OpListIterator parentOpIt, mv::Data::TensorIterator sourceTensor, mv::OpModel om, mv::Data::OpListIterator opIt);
    mv::Data::OpListIterator linkNewMultipleOperationsReplacement(mv::Data::OpListIterator parentOpIt,
                                                      std::vector<mv::Data::TensorIterator> sourceTensors, mv::OpModel om, mv::Data::OpListIterator opIt);
    std::vector<mv::Data::OpListIterator> findSinkLayers(mv::DataModel &dataModel, const mv::Data::TensorIterator &tensor);
    bool checkA0SOHSparsityBug(mv::Data::FlowListIterator flow);
}
void calcZeroPointAndScalePerTensor(double outputMax,  double outputMin, double& outScale, int64_t& outZp);
void updateInfMinMaxPerTensor(mv::Data::TensorIterator tensor);
void updateInfMinMaxPerChannel(mv::Data::TensorIterator tensor);
void provideAccuracyinPPEs(mv::ComputationModel& model);
//template <class T>
//std::vector<T> extendToK(size_t size, std::vector<T> value, std::string tensorName);
std::vector<double> extendToK(size_t size, std::vector<double> value, std::string tensorName);
std::vector<int64_t> extendToK(size_t size, std::vector<int64_t> value, std::string tensorName);

#endif // PASS_UTILS_HPP_
