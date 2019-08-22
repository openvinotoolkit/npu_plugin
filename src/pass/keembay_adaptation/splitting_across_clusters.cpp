#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/utils/warning_manager.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/target/keembay/workload_struct.hpp"
#include "include/mcm/target/keembay/workloads.hpp"
#include "include/mcm/target/keembay/rectangle.hpp"

static const std::vector<mv::DPUModeList> TENSOR_MPE {{{1,1}}, {{16,1}}, {{1,16}}};

static void SplittingTensorsAcrossClusters(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&,
                                    mv::Element&, mv::json::Object&);
static void subTensorsGen(mv::ComputationModel& model, const std::vector<mv::Data::TensorIterator> &tensors, unsigned nClusters,
                          const mv::pass::PassEntry& pass);
static void unpopulatedSplitOverH(const unsigned nWorkloads, std::vector<mv::Workload> &subTensors, mv::Workloads &Tensor,
                                  const mv::pass::PassEntry& pass, int &success);
static void populatedSplitOverH(const unsigned nClusters, std::vector<mv::Workload> &subTensors, mv::Workloads& Tensor,
                                const mv::pass::PassEntry& pass, int &success);
static std::vector<mv::Data::OpListIterator> findSinkLayers(mv::DataModel &dataModel, const mv::Data::TensorIterator& tensor);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(SplittingTensorsAcrossClusters)
        .setFunc(SplittingTensorsAcrossClusters)
        .setDescription(
            "Computing Splitting across clusters"
        );
    }
}

void SplittingTensorsAcrossClusters(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&,
                             mv::json::Object&)
{
    mv::OpModel om(model);
    auto globalParams = model.getGlobalConfigParams();
    unsigned numClusters = globalParams->get<int>("Number_of_Clusters");

    if (numClusters > 1)
    {
        std::vector <mv::Data::TensorIterator> tensors;
        for(auto layer = om.opBegin(); layer != om.opEnd(); ++layer)
        {
            std::string opType = layer->getOpType();
            if (opType == "DPUTask")
            {
                auto outputTensor = layer->getOutputTensor(0);
                tensors.push_back(outputTensor);
                for(std::size_t i = 0; i < layer->inputSlots(); ++i)
                {
                    auto inputTensor = layer->getInputTensor(i);
                    tensors.push_back(inputTensor);
                }
            }
        }
        for(auto layer = om.opBegin(); layer != om.opEnd(); ++layer)
        {
            std::string opType = layer->getOpType();
            if (opType == "DMATask")
            {
                auto outputTensor = layer->getOutputTensor(0);
                if (std::find(tensors.begin(), tensors.end(), outputTensor) == tensors.end())
                    tensors.push_back(outputTensor);
                auto inputTensor = layer->getInputTensor(0);
                if (std::find(tensors.begin(), tensors.end(), inputTensor) == tensors.end())
                    tensors.push_back(inputTensor);
            }
        }
        subTensorsGen(model, tensors, numClusters, pass);
    }
    return;
}

void subTensorsGen(mv::ComputationModel& model, const std::vector <mv::Data::TensorIterator>& tensors, unsigned nClusters,
                   const mv::pass::PassEntry& pass)
{
    mv::DataModel dm(model);
    for (auto& tensor : tensors)
    {
        int success;
        UNUSED(success);
        int nWorkloads = nClusters;
        mv::Workloads Tensor(tensor->getName(), tensor->getShape());
        std::vector<mv::Workload> subTensors;
        if (tensor->get<std::string>("splitStrategy") == "SplitOverH")
        {
            if(!tensor->isPopulated())
                unpopulatedSplitOverH(nClusters, subTensors, Tensor, pass, success);
            else
                populatedSplitOverH(nClusters, subTensors, Tensor, pass, success);
            tensor->splitAcrossClusters(subTensors, true, false);
        }
        else if (tensor->get<std::string>("splitStrategy") == "SplitOverHOverlapped")
        {
            if(!tensor->isPopulated())
            {
                unpopulatedSplitOverH(nClusters, subTensors, Tensor, pass, success);
                std::vector<mv::Data::OpListIterator> sinkOperators = findSinkLayers(dm, tensor);
                if (sinkOperators[0]->getOpType() == "DMATask")
                    sinkOperators = findSinkLayers(dm, sinkOperators[0]->getOutputTensor(0));
                //The sink ops should have the same padding
                std::array <unsigned short, 4> padding = {0, 0, sinkOperators[0]->get<std::array<unsigned short, 4>>("padding")[2],
                                                       sinkOperators[0]->get<std::array<unsigned short, 4>>("padding")[3]};
                //Rectangular Heuristc: The workload has only one rectangle in its list, itself
                subTensors = Tensor.overlap_and_clip(padding, tensor->getShape());
            }
            else
                populatedSplitOverH(nClusters, subTensors, Tensor, pass, success);
            tensor->splitAcrossClusters(subTensors, true, false);
        }
        else if (tensor->get<std::string>("splitStrategy") == "SplitOverK")
        {
            if(!tensor->isPopulated())
                success = Tensor.partitionTensorWithRectangleHeuristic(TENSOR_MPE[2], nWorkloads, false, true, true,
                        mv::WorkloadSplitMode::NC, pass);
            else
                success = Tensor.partitionTensorWithRectangleHeuristic(TENSOR_MPE[1], nWorkloads, true, false, true,
                        mv::WorkloadSplitMode::NC, pass);
            subTensors = Tensor.getWorkloads();
            tensor->splitAcrossClusters(subTensors, false, false);
        }
        else if (tensor->get<std::string>("splitStrategy") == "HKSwitch")
        {
            if(!tensor->isPopulated())
            {
                unpopulatedSplitOverH(nClusters, subTensors, Tensor, pass, success);
                tensor->splitAcrossClusters(subTensors, true, true);
            }
            else
            {
                populatedSplitOverH(nClusters, subTensors, Tensor, pass, success);
                tensor->splitAcrossClusters(subTensors, true, false);
            }
        }
    }
    return;
}

static void unpopulatedSplitOverH(const unsigned nWorkloads, std::vector<mv::Workload> &subTensors, mv::Workloads& Tensor,
                                  const mv::pass::PassEntry& pass, int &success)
{
    success = Tensor.partitionTensorWithRectangleHeuristic(TENSOR_MPE[0], nWorkloads, true, false, true,
            mv::WorkloadSplitMode::H, pass);
    subTensors = Tensor.getWorkloads();
    return;
}

static void populatedSplitOverH(const unsigned nClusters, std::vector<mv::Workload> &subTensors, mv::Workloads& Tensor,
                                          const mv::pass::PassEntry& pass, int &success)
{
    success = Tensor.partitionTensorWithRectangleHeuristic(TENSOR_MPE[0], 1, true, false, true,
            mv::WorkloadSplitMode::H, pass);
    subTensors = Tensor.getWorkloads();
    std::vector<mv::Workload> newSubTensors;
    for (unsigned int i = 0; i < nClusters; i++)
    {
        for (unsigned int j =0; j < subTensors.size(); j++)
            newSubTensors.push_back(subTensors[j]);
    }
    subTensors = newSubTensors;
    return;
}

static std::vector<mv::Data::OpListIterator> findSinkLayers(mv::DataModel &dataModel, const mv::Data::TensorIterator &tensor)
{
    std::vector<mv::Data::OpListIterator> sinkOperations;
    auto flowsNames = (tensor)->get<std::set<std::string>>("flows");
    for(auto flowName : flowsNames)
    {
        auto df = dataModel.getDataFlow(flowName);
        sinkOperations.push_back(df.sink());
    }
    return sinkOperations;
}
