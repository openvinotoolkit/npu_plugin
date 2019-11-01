#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/utils/warning_manager.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/target/kmb/workload_struct.hpp"
#include "include/mcm/target/kmb/workloads.hpp"
#include "include/mcm/target/kmb/rectangle.hpp"

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
    mv::DataModel dm(model);
    auto globalParams = model.getGlobalConfigParams();
    unsigned int numClusters = (unsigned int)globalParams->get<int>("Number_of_Clusters");

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

                    // New weights sparsity approach: no explicit costant operation
                    // for sparsity map is present in the graph.
                    // So check for sparsity has to be done only here
                    if(inputTensor->isPopulated() && inputTensor->isSparse())
                        tensors.push_back(dm.getTensor(inputTensor->getSparsityMap()->getName()));

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
    auto globalParams = model.getGlobalConfigParams();
    unsigned nWorkloads = nClusters;

    for (auto& tensor : tensors)
    {
        int success;
        UNUSED(success);
        mv::Workloads Tensor(tensor->getName(), tensor->getShape());
        std::vector<mv::Workload> subTensors;

        if (tensor->get<std::string>("splitStrategy") == "SplitOverH")
        {
            if(!tensor->isPopulated())
            {
                unpopulatedSplitOverH(nClusters, subTensors, Tensor, pass, success);
            }
            else
                populatedSplitOverH(nClusters, subTensors, Tensor, pass, success);
            tensor->splitAcrossClusters(subTensors, true, false);
        }
        else if (tensor->get<std::string>("splitStrategy") == "Clustering")
        {
            //NOTE: Compute the same subtensors with the initial Tensor in order to do everything 1
            //function in seralization
            if (!tensor->isPopulated())
            {
                for (unsigned i = 0; i < nWorkloads; i++)
                {
                    mv::Workload subTensor;
                    subTensor.MaxX = tensor->getShape()[mv::IO_WIDTH_DIMENSION];
                    subTensor.MinX = 0;
                    subTensor.MaxZ = tensor->getShape()[mv::IO_CHANNEL_DIMENSION];
                    subTensor.MinZ = 0;
                    subTensor.MaxY = tensor->getShape()[mv::IO_HEIGHT_DIMENSION];
                    subTensor.MinY = 0;
                    subTensors.push_back(subTensor);
                }
            }
            else
            {
                for (unsigned i = 0; i < nWorkloads; i++)
                {
                    mv::Workload subTensor;
                    subTensor.MaxX = tensor->getShape()[mv::KERNEL_WIDTH];
                    subTensor.MinX = 0;
                    subTensor.MaxZ = tensor->getShape()[mv::KERNEL_OUTPUT_CHANNELS];
                    subTensor.MinZ = 0;
                    subTensor.MaxY = tensor->getShape()[mv::KERNEL_INPUT_CHANNELS];
                    subTensor.MinY = 0;
                    subTensors.push_back(subTensor);
                }
            }
            tensor->shareAcrossClusters(subTensors, nWorkloads);
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
            //NOTE:Temporary handle for bug in Rectangular Heuristic
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

    if (!tensor->isPopulated())
    {
        std::vector<size_t> z_sizes(nWorkloads);

        auto totalZ = subTensors[0].MaxX;
        for(size_t i=1; i<subTensors.size();i++)
            if (subTensors[i].MaxX > totalZ)
                totalZ = subTensors[i].MaxX;
        totalZ++;

        int t=0;
        while (totalZ > 0)
        {
            z_sizes[t] += 16;
            totalZ -= 16;
            t++;
            if (t == nWorkloads)
                t = 0;
        }
        for (int i = 0; i < nWorkloads; i++)
        {
            mv::Workload subTensor = subTensors[0];

            if (i==0)
            {
                subTensor.MinX = 0;
                subTensor.MaxX = z_sizes[i] - 1;
            }
            else
            {
                subTensor.MinX = newSubTensors[i - 1].MaxX + 1;
                subTensor.MaxX = subTensor.MinX + z_sizes[i] - 1;
            }
            newSubTensors.push_back(subTensor);
        }
    }
    else
    {
        std::vector<size_t> z_sizes(nWorkloads);

        auto totalZ = subTensors[0].MaxY;
        for(size_t i=1; i<subTensors.size();i++)
            if (subTensors[i].MaxY > totalZ)
                totalZ = subTensors[i].MaxY;
        totalZ++;

        int t=0;
        while (totalZ > 0)
        {
            z_sizes[t] += 16;
            totalZ -= 16;
            t++;
            if (t == nWorkloads)
                t = 0;
        }
        for (int i = 0; i < nWorkloads; i++)
        {
            mv::Workload subTensor = subTensors[0];

            if (i==0)
            {
                subTensor.MinY = 0;
                subTensor.MaxY = z_sizes[i] - 1;
            }
            else
            {
                subTensor.MinY = newSubTensors[i - 1].MaxY + 1;
                subTensor.MaxY = subTensor.MinY + z_sizes[i] - 1;
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

                if (opIt->get<mv::DmaDirection>("direction") == mv::DmaDirectionEnum::DDR2NNCMX &&
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
