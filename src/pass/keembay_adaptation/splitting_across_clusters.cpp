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
                                    mv::Element&, mv::Element&);
static void subTensorsGen(mv::ComputationModel& model, const std::vector<mv::Data::TensorIterator> &tensors, unsigned nClusters,
                          const mv::pass::PassEntry& pass);
static void unpopulatedSplitOverH(const unsigned nWorkloads, std::vector<mv::Workload> &subTensors, mv::Workloads &Tensor,
                                  const mv::pass::PassEntry& pass, int &success);
static void populatedSplitOverH(const unsigned nClusters, std::vector<mv::Workload> &subTensors, mv::Workloads& Tensor,
                                const mv::pass::PassEntry& pass, int &success);
static std::vector<mv::Data::OpListIterator> findSinkLayers(mv::DataModel &dataModel, const mv::Data::TensorIterator& tensor);
static std::vector<mv::Workload> fixRectangularHeuristicBug(std::vector<mv::Workload> subTensors, const mv::Data::TensorIterator &tensor, int nWorkloads);
static void ensureSplitStrategiesForSpilling(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);


namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(SplittingTensorsAcrossClusters)
        .setFunc(SplittingTensorsAcrossClusters)
        .setDescription(
            "Computing Splitting across clusters"
        );


        MV_REGISTER_PASS(EnsureSplitStrategiesForSpilling)
            .setFunc(ensureSplitStrategiesForSpilling)
            .setDescription(
               "Ensures Split Strategies still valid after Spilling cases");
    }
}

void SplittingTensorsAcrossClusters(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&,
                             mv::Element &)
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
        else if (tensor->get<std::string>("splitStrategy") == "Clustering")
        {
            tensor->splitAcrossClusters(subTensors, false, false, true);
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
            int16_t difference;
            std::set<int16_t> differences;
            for (unsigned idx = 0; idx < nWorkloads; idx++)
            {
                difference = subTensors[idx].MaxX - subTensors[idx].MinX;
                differences.insert(difference);
            }
            //NOTE This is coming from the fact of symmetrical split over K
            if (differences.size() > 1)
            {
                auto newSubTensors = fixRectangularHeuristicBug(subTensors, tensor, nWorkloads);
                subTensors.clear();
                subTensors = newSubTensors;
            }
                //NOTE:Permanent handle for bug in Rectangular Heuristic
            if (subTensors.size() != nWorkloads)
            {
                auto newSubTensors = fixRectangularHeuristicBug(subTensors, tensor, nWorkloads);
                subTensors.clear();
                subTensors = newSubTensors;
            }
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

static std::vector<mv::Workload> fixRectangularHeuristicBug(std::vector<mv::Workload> subTensors, const mv::Data::TensorIterator &tensor, int nWorkloads)
{

    std::vector<mv::Workload> newSubTensors;
    auto tensorNeedsAlignment = tensor->hasAttr("alignment") ? tensor->get<bool>("alignment") : false;
    size_t paddedSymetricalOutputChannels;
    if (tensorNeedsAlignment)
    {
        int numClusters = 4; //TODO
        int pad = 16 * numClusters; //todo fix it

        auto output_channels = tensor->getShape()[mv::KERNEL_INPUT_CHANNELS];
        output_channels = mv::round_up(output_channels, pad);
        paddedSymetricalOutputChannels = output_channels / numClusters ;
    }
    if (!tensor->isPopulated())
    {
        for (unsigned i = 0; i < nWorkloads; i ++)
        {
            mv::Workload subTensor;
            if (i == 0)
            {
                subTensor.MaxX = 15;
                subTensor.MaxY = subTensors[0].MaxY;
                subTensor.MaxZ = subTensors[0].MaxZ;
                subTensor.MinX = 0;
                subTensor.MinY = subTensors[0].MinY;
                subTensor.MinZ = subTensors[0].MinZ;

            }
            else if (i == 1)
            {
                subTensor.MaxX = 31;
                subTensor.MaxY = subTensors[0].MaxY;
                subTensor.MaxZ = paddedSymetricalOutputChannels;
                subTensor.MinX = 16;
                subTensor.MinY = subTensors[0].MinY;
                subTensor.MinZ = subTensors[0].MinZ;
            }
            else if (i == nWorkloads - 1)
            {
                subTensor = subTensors[subTensors.size() - 1];
            }
            else
            {
                subTensor = subTensors[1];
            }
            if (tensorNeedsAlignment)
            {
                subTensor.MinX = paddedSymetricalOutputChannels*i;
                subTensor.MaxX = subTensor.MinX + paddedSymetricalOutputChannels - 1;
            }
            newSubTensors.push_back(subTensor);
        }
    }
    else
    {
        for (unsigned i = 0; i < nWorkloads; i ++)
         {
            mv::Workload subTensor;
            if (i == 0)
            {
                subTensor.MaxX = subTensors[0].MaxX;
                subTensor.MaxY = 15;
                subTensor.MaxZ = subTensors[0].MaxZ;
                subTensor.MinX = subTensors[0].MinX;
                subTensor.MinY = 0;
                subTensor.MinZ = subTensors[0].MinZ;
            }
            else if (i == 1)
            {
                subTensor.MaxX = subTensors[0].MaxX;
                subTensor.MaxY = 31;
                subTensor.MaxZ = subTensors[0].MaxZ;
                subTensor.MinX = subTensors[0].MinX;
                subTensor.MinY = 16;
                subTensor.MinZ = subTensors[0].MinZ;
            }
            else if (i == nWorkloads - 1)
            {
                subTensor = subTensors[subTensors.size() - 1];
            }
            else
            {
                subTensor = subTensors[1];
            }
            newSubTensors.push_back(subTensor);
         }
    }

    return newSubTensors;
}

// Pass role: Splitting Strategies propagation algorithm may create an incompatibility
void ensureSplitStrategiesForSpilling(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);
    std::vector<std::pair<std::string, std::string>>incompatibleStrategies =
    {
        {"SplitOverHOverlapped", "Clustering"},
        {"SplitOverHOverlapped", "SplitOverK"},
        {"SplitOverH", "Clustering"},
        {"SplitOverH", "SplitOverK"},
        {"SplitOverK", "SplitOverH"},
        {"Clustering", "SplitOverH"},
        {"SplitOverK", "HKSwitch"},
        {"Clustering", "HKSwitch"}
    };
    auto globalParams = model.getGlobalConfigParams();
    unsigned numClusters = globalParams->get<int>("Number_of_Clusters");

    if (numClusters > 1)
    {
        for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
        {
            std::string opType = opIt->getOpType();
            if (opType == "DMATask")
            {
                auto outputTensor = opIt->getOutputTensor(0);
                auto inputTensor = opIt->getInputTensor(0);

                if (opIt->get<mv::DmaDirection>("direction") == mv::DmaDirectionEnum::DDR2CMX &&
                    !outputTensor->isPopulated())
                {
                    std::vector<mv::Data::OpListIterator> sinkOperators = findSinkLayers(dm, outputTensor);

                    //ASSUMPTION: all sink ops have the same strategy.
                    auto opStrategy = sinkOperators[0]->get<std::string>("splitStrategy");
                    auto tensorStrategy = outputTensor->get<std::string>("splitStrategy");

                    std::pair<std::string, std::string> possibleCombination(tensorStrategy, opStrategy);
                    for (auto restrictedCombination: incompatibleStrategies)
                    {
                        if (possibleCombination == restrictedCombination)
                        {
                            // Strategy have to be adjusted...
                            outputTensor->set<std::string>("splitStrategy", opStrategy);

                            // ... and splitting has to be done again!!! <- Price for efficiency
                            subTensorsGen(model, {outputTensor}, numClusters, pass);
                        }
                    }
                }
            }
        }
    }
    return;

}
