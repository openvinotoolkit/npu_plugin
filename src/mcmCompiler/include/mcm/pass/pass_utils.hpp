#ifndef PASS_UTILS_HPP_
#define PASS_UTILS_HPP_

#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/target/target_descriptor.hpp"

//cause of the BASE_PTR is 9 bits, -4 for the 16 alignment according to zoran
static const std::size_t SHIFT_FOR_STORAGE_ELEMENT = 5;
static const std::vector<std::string> activationSegmentableStrategies = {"SplitOverH", "HKSwitch"};
static const std::vector<std::string> activationRepetitionStrategies = {"SplitOverK", "Clustering"};

namespace mv
{
    std::vector<std::pair<mv::Data::OpListIterator,size_t>> getOutputDataFlow(mv::OpModel& om, mv::Data::OpListIterator &opIt, bool deleteOp = true);
    void setOutputDataFlow(mv::OpModel& om, mv::Data::TensorIterator &dpuTaskOutputTensor, const std::vector<std::pair<mv::Data::OpListIterator,size_t>>& outDataFlows);
    std::vector<mv::Control::OpListIterator> getInputControlFlow(mv::ControlModel& om, mv::Control::OpListIterator opIt);
    std::vector<mv::Control::OpListIterator> getOutputControlFlow(mv::ControlModel& om, mv::Control::OpListIterator opIt);
    void setInputControlFlow(mv::ControlModel& cm, mv::Control::OpListIterator op, const std::vector<mv::Control::OpListIterator>& inputControlFlows);
    void setOutputControlFlow(mv::ControlModel& cm, mv::Control::OpListIterator op, const std::vector<mv::Control::OpListIterator>& outputControlFlows);
    void removeConstantOp(mv::OpModel & om, mv::Data::OpListIterator paramOp);
    void removeOperation(mv::Data::TensorIterator sourceTensor, mv::OpModel & om, mv::Data::OpListIterator opIt);
    mv::Data::OpListIterator linkNewOperationsRemove(mv::Data::OpListIterator parentOpIt, mv::Data::TensorIterator sourceTensor, mv::OpModel & om, mv::Data::OpListIterator opIt);
    mv::Data::OpListIterator linkNewOperationsReplacement(mv::Data::OpListIterator parentOpIt, mv::Data::TensorIterator sourceTensor, mv::OpModel & om, mv::Data::OpListIterator opIt);
    mv::Data::OpListIterator linkNewOperationsReplacementRemoveFlows(mv::Data::OpListIterator childOpIt, mv::Data::TensorIterator sourceTensor, mv::OpModel & om, mv::Data::OpListIterator opIt);
    mv::Data::OpListIterator linkNewMultipleOperationsReplacement(mv::Data::OpListIterator parentOpIt, std::vector<mv::Data::TensorIterator> sourceTensors, mv::OpModel & om, mv::Data::OpListIterator opIt);
    mv::Data::OpListIterator linkNewMultipleOperationsReplacementRemoveFlows(mv::Data::OpListIterator parentOpIt, std::vector<mv::Data::TensorIterator> sourceTensors, mv::OpModel & om, mv::Data::OpListIterator opIt);
    mv::Data::TensorIterator insertDMAReplacementRemoveFlows(mv::OpModel& om, mv::Data::OpListIterator opIt, mv::Data::TensorIterator input, mv::DmaDirection const& direction, int8_t const &port, std::vector<mv::Data::FlowListIterator> flows, std::vector<std::size_t> inSlots, std::vector<mv::Data::OpListIterator> sinks, std::string const& dmaOpName);
    std::vector<mv::Data::OpListIterator> findSinkLayers(mv::DataModel &dataModel, const mv::Data::TensorIterator &tensor);
    bool checkA0SOHSparsityBug(mv::Data::FlowListIterator flow, std::string referenceDevice, mv::Target target);

    mv::Data::TensorIterator dequantizeWeightsToFP16(
        mv::Data::TensorIterator weightsTensor,
        mv::Data::OpListIterator opIt,
        mv::OpModel &om);

    bool isVectorsEqual(const std::vector<double>& left, const std::vector<double>& right);
    bool isEqualScale(const mv::QuantizationParams& left, const mv::QuantizationParams& right);
    bool isEqual(const mv::QuantizationParams& left, const mv::QuantizationParams& right);
    bool checkPPEAccuracy(mv::ComputationModel& model);
    bool checkA0Sparsity(const mv::OpModel& model);
    std::vector<std::string>::const_iterator findIsDPUPwlPostOp(const std::vector<std::string>& postOps, const mv::TargetDescriptor& td);
    bool matchPattern(const std::vector<std::string>& pattern, mv::Data::OpListIterator it, mv::ComputationModel& model);
    bool matchPattern(const std::vector<std::string>& pattern, mv::Data::OpListIterator it, mv::Data::OpListIterator& lastIt, mv::ComputationModel& model);
}

void fuseLeakyReluAccPPEFcn(mv::Data::OpListIterator& opIt, mv::ComputationModel& model, const std::string& opType, mv::TargetDescriptor& td);

std::vector<double> extendToK(size_t size, std::vector<double> value, const std::string& tensorName);
std::vector<int64_t> extendToK(size_t size, std::vector<int64_t> value, const std::string& tensorName);

#endif // PASS_UTILS_HPP_
