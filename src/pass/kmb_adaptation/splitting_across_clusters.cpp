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
#include "include/mcm/tensor/tiling.hpp"

static const std::vector<mv::DPUModeList> TENSOR_MPE {{{1,1}}, {{16,1}}, {{1,16}}};

static void SplittingTensorsAcrossClusters(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&,
                                    mv::Element&, mv::Element&);
static void subTensorsGen(mv::ComputationModel& model, const std::vector<mv::Data::TensorIterator> &tensors, unsigned nClusters,
                          const mv::pass::PassEntry& pass, std::size_t id=0);
static void unpopulatedSplitOverH(const unsigned nWorkloads, std::vector<mv::Workload> &subTensors, mv::Workloads &Tensor,
                                  const mv::pass::PassEntry& pass, int &success);
static void populatedSplitOverH(const unsigned nClusters, std::vector<mv::Workload> &subTensors, mv::Workloads& Tensor,
                                const mv::pass::PassEntry& pass, int &success);
static std::vector<mv::Workload> fixRectangularHeuristicBug(std::vector<mv::Workload> subTensors, const mv::Data::TensorIterator &tensor, int nWorkloads, int outputChannels);
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

bool findSparseTensorIndex(
    mv::Data::OpListIterator layer,
    const std::string& name,
    std::size_t tensorIdx)
{
    bool found = false;
    if (layer->hasAttr(name))
    {
        auto tensorList = layer->get<std::vector<size_t>>(name);
        if (std::find(tensorList.begin(), tensorList.end(), tensorIdx) !=
            tensorList.end())
            found = true;
    }
    return found;
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
        std::set <std::string> tensorNames;
        //NOTE: special Tensors are marked the tensors that are populated that need to be handled
        //as unpopulated, cases where compiler handes the activation sparsity!!!
        //terrible...compiler concludes a solver of hardware limitations
        std::set <std::string> specialTensorNames;
        std::vector <mv::Data::TensorIterator> tensors;
        std::vector <mv::Data::TensorIterator> specialTensors;

        //Todo:: the construction and logic of this pass needs to be refactored.
        // The pass should not target specific ops to determine if it's output needs to have subtensors generated,
        // but via location. If  the outputTensor is in NNCMX, then it needs a clustering strategy, and subtensors
        // They can be also non DPUTasks like ConcatInCMX, Slice,Reshape etc...

        auto implicitConcats = om.getOps("ImplicitConcat");
        for(auto layer : implicitConcats)
        {
            auto outputTensor = layer->getOutputTensor(0);
            if(outputTensor->get<mv::Tensor::MemoryLocation>("Location") == mv::Tensor::MemoryLocation::NNCMX)
            {
                auto outputTensorName = layer->getOutputTensor(0)->getName();
                tensorNames.insert(outputTensorName);
            }
        }
        auto dpuTasks = om.getOps("DPUTask");
        for(auto layer : dpuTasks)
        {
            auto outputTensorName = layer->getOutputTensor(0)->getName();
            tensorNames.insert(outputTensorName);
            for(std::size_t i = 0; i < layer->inputSlots(); ++i)
            {
                if (findSparseTensorIndex(layer, "unpopulatedSparsityMapIndex", i) ||
                    findSparseTensorIndex(layer, "storageElementIndex", i))
                    specialTensorNames.insert(layer->getInputTensor(i)->getName());
                else
                {
                    auto inputTensorName = layer->getInputTensor(i)->getName();
                    auto inputTensor = layer->getInputTensor(i);
                    tensorNames.insert(inputTensorName);

                    // New weights sparsity approach: no explicit costant operation
                    // for sparsity map is present in the graph.
                    // So check for sparsity has to be done only here
                    if(inputTensor->isPopulated() && inputTensor->isSparse())
                        tensorNames.insert(inputTensor->getSparsityMap()->getName());
                }
            }
        }
        //Also need to generate subtensors for output tensor of input operation, and the input tensor of output operation
        auto inOutputTensor = om.getInput()->getOutputTensor(0);
        if((inOutputTensor->get<std::string>("splitStrategy") == "SplitOverH") || (inOutputTensor->get<std::string>("splitStrategy") == "SplitOverHOverlapped"))
        {
            auto inOutputTensorName = inOutputTensor->getName();
            tensorNames.insert(inOutputTensorName);
        }

        auto outInputTensor = om.getOutput()->getInputTensor(0);
        if((outInputTensor->get<std::string>("splitStrategy") == "SplitOverH") || (outInputTensor->get<std::string>("splitStrategy") == "SplitOverHOverlapped"))
        {
            auto outInputTensorName = outInputTensor->getName();
            tensorNames.insert(outInputTensorName);
            //Note: Output can't take input activation sparsity, so this code shouldn't be needed
            // if(outInputTensor->isPopulated() && outInputTensor->isSparse())
            //     tensorNames.insert(outInputTensor->getSparsityMap()->getName());
        }

        for (auto specialTensorName : specialTensorNames)
            specialTensors.push_back(dm.getTensor(specialTensorName));
        for (auto tensorName : tensorNames)
            tensors.push_back(dm.getTensor(tensorName));
        subTensorsGen(model, tensors, numClusters, pass);
        subTensorsGen(model, specialTensors, numClusters, pass, 1);
        for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
        {
            if (opIt->getOpType() == "Crop" || opIt->getOpType() == "Concat" || opIt->getOpType() == "ImplicitConcat" || opIt->getOpType() == "Slice")
            {
                auto sinkOperators = findSinkLayers(dm, opIt->getOutputTensor(0));
                if (sinkOperators[0]->getOpType() == "Align" || sinkOperators[0]->getOpType() == "Crop")
                {
                    subTensorsGen(model, {opIt->getOutputTensor(0)},numClusters, pass);
                }
            }
        }
    }
    return;
}

void subTensorsGen(mv::ComputationModel& model, const std::vector <mv::Data::TensorIterator>& tensors, unsigned nClusters,
                   const mv::pass::PassEntry& pass, std::size_t id)
{
    mv::DataModel dm(model);
    unsigned nWorkloads = nClusters;

    if (id == 0)
    {
        for (auto& tensor : tensors)
        {
            int success;
            UNUSED(success);
            auto needs_sparse = (tensor->hasAttr("needs_sparse")) ? tensor->get<bool>("needs_sparse") : false;
            auto needs_splits_aligned = (tensor->hasAttr("needs_splits_aligned")) ? tensor->get<bool>("needs_splits_aligned") : false;
            mv::Workloads Tensor(tensor->getName(), tensor->getShape(), needs_sparse | needs_splits_aligned);
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

                    //Rectangular Heuristc: The workload has only one rectangle in its list, itself
                    //SplitOverHOverlapped happens only for CM Conv in the input layer. Have to step through DMATask and Align layers till we reach the DPUTask op
                    if (sinkOperators[0]->getOpType() == "DMATask" || sinkOperators[0]->getOpType() == "Align")
                        sinkOperators = findSinkLayers(dm, sinkOperators[0]->getOutputTensor(0));
                    if (sinkOperators[0]->getOpType() == "DMATask" || sinkOperators[0]->getOpType() == "Align")
                        sinkOperators = findSinkLayers(dm, sinkOperators[0]->getOutputTensor(0));

                    //Tiling logic is used to find the workloads with the overlaps
                    //SplitOverHOverlapped - so hardCoding to 'H' for axis

                    std::vector<mv::Tiling> childtiles;
                    std::vector<size_t> heightSizes;
                    std::string axis = "H";
                    if (sinkOperators[0]->getOpType() == "DPUTask")
                    {
                        mv::Tiling masterTile(axis, int(subTensors.size()));
                        masterTile.setSize(sinkOperators[0]->getInputTensor(0)->getShape());
                        masterTile.generateSpatialTiling(sinkOperators[0]);
                        childtiles = masterTile.childTiles();

                        for (unsigned i =0; i < subTensors.size(); i++)
                        {
                            auto startCoord = childtiles[i].getStartCoord()[mv::Shape::getAxis(axis)];
                            auto endCoord = childtiles[i].getSize()[mv::Shape::getAxis(axis)] + startCoord -1;
                            subTensors[i].MinY = startCoord;
                            subTensors[i].MaxY = endCoord;
                        }
                    }
                }
                else
                {
                    populatedSplitOverH(nClusters, subTensors, Tensor, pass, success);
                }
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
                    mv::Shape tensorShape = tensor->getShape();
                    std::size_t outputChannels = tensorShape[mv::IO_CHANNEL_DIMENSION];
                    std::vector<mv::Data::OpListIterator> sinkOperators = findSinkLayers(dm, tensor);
                    if (!sinkOperators.empty())
                    {
                        while (sinkOperators[0]->getOpType() != "DPUTask" and
                                sinkOperators[0]->getOpType() != "Output"){
                            sinkOperators = findSinkLayers(dm, sinkOperators[0]->getOutputTensor(0));
                        }
                        if(sinkOperators[0]->getOpType() != "Output")
                            outputChannels = sinkOperators[0]->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION];
                    }
                    auto newSubTensors = fixRectangularHeuristicBug(subTensors, tensor, nWorkloads, outputChannels);
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
    }
    else
    {
        for (auto& tensor : tensors)
        {
            int success;
            auto needs_sparse = (tensor->hasAttr("needs_sparse")) ? tensor->get<bool>("needs_sparse") : false;
            auto needs_splits_aligned = (tensor->hasAttr("needs_splits_aligned")) ? tensor->get<bool>("needs_splits_aligned") : false;
            mv::Workloads Tensor(tensor->getName(), tensor->getShape(), needs_sparse | needs_splits_aligned);
            std::vector<mv::Workload> subTensors;

            if (tensor->get<std::string>("splitStrategy") == "SplitOverH")
            {
                unpopulatedSplitOverH(nClusters, subTensors, Tensor, pass, success);
                tensor->splitPopulatedActivationAcrossClusters(subTensors, true, false);
            }
            else if (tensor->get<std::string>("splitStrategy") == "Clustering" ||
                     tensor->get<std::string>("splitStrategy") == "SplitOverK")
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
                tensor->shareAcrossClusters(subTensors, nWorkloads);
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

//TODO re-enable this version
static std::vector<mv::Workload> fixRectangularHeuristicBug(std::vector<mv::Workload> subTensors, const mv::Data::TensorIterator &tensor, int nWorkloads, int outputChannels)
{
    std::vector<mv::Workload> newSubTensors;

    if (!tensor->isPopulated())
    {
        if (subTensors.empty())
        {
            std::size_t quantumofAlignedChannels, equalSlice, remainingSlice = 0;
            if (outputChannels % 16 == 0)
            {
                quantumofAlignedChannels = outputChannels/16;
                equalSlice = quantumofAlignedChannels/nWorkloads;
                remainingSlice = quantumofAlignedChannels%nWorkloads;
            }
            else
                throw std::string("Trying to compute SubTensors for an unaligned Tensor");
            for (int n = 0; n < nWorkloads; n++)
            {
                mv::Workload subTensor;
                if (n != nWorkloads - 1)
                {
                    subTensor.MaxX = tensor->getShape()[mv::IO_WIDTH_DIMENSION];
                    subTensor.MinX = 0;
                    subTensor.MaxZ = equalSlice * (n+1) * 16;
                    subTensor.MinZ = equalSlice * n * 16;
                    subTensor.MaxY = tensor->getShape()[mv::IO_HEIGHT_DIMENSION];
                    subTensor.MinY = 0;
                }
                else
                {
                    subTensor.MaxX = tensor->getShape()[mv::IO_WIDTH_DIMENSION];
                    subTensor.MinX = 0;
                    subTensor.MaxZ = (equalSlice + remainingSlice) * (n+1) * 16;
                    subTensor.MinZ = equalSlice * n;
                    subTensor.MaxY = tensor->getShape()[mv::IO_HEIGHT_DIMENSION];
                    subTensor.MinY = 0;
                }
                newSubTensors.push_back(subTensor);
            }
        }
        else
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
    std::pair<std::string, std::string> clusteringToSoH("Clustering", "SplitOverH");
    std::pair<std::string, std::string> SoKToSoH("SplitOverK", "SplitOverH");
    std::pair<std::string, std::string> SoHToSoK("SplitOverH", "SplitOverK");
    std::pair<std::string, std::string> SoHToClustering("SplitOverH", "Clustering");
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

                    if (opIt->getOutputTensor(0)->get<std::string>("splitStrategy") != "SplitOverHOverlapped")
                    {
                        if (sinkOperators[0]->getOpType() == "DPUTask")
                        {
                            if (sinkOperators[0]->get<std::string>("taskOp") == "ChannelMajorConvolution")
                            {
                                if (sinkOperators[0]->get<std::string>("splitStrategy") == "SplitOverH")
                                {
                                    outputTensor->setOrder(mv::Order(mv::Order::getColMajorID(4)));
                                    inputTensor->cleanSubtensors();
                                    outputTensor->cleanSubtensors();
                                    outputTensor->set<std::string>("splitStrategy", "SplitOverHOverlapped");
                                    inputTensor->set<std::string>("splitStrategy", "SplitOverHOverlapped");
                                    subTensorsGen(model, {inputTensor, outputTensor}, numClusters, pass);

                                    // support for input->slice->DMA->CMConv

                                    auto sourceOp = om.getSourceOp(inputTensor);
                                    if (sourceOp->getOpType() == "Slice")
                                    {
                                        sourceOp->set<std::string>("splitStrategy", "SplitOverHOverlapped");
                                        auto sourceOpInputTensor = sourceOp->getInputTensor(0);
                                        sourceOpInputTensor->cleanSubtensors();
                                        sourceOpInputTensor->set<std::string>("splitStrategy", "SplitOverHOverlapped");
                                        inputTensor->cleanSubtensors();
                                        inputTensor->set<std::string>("splitStrategy", "SplitOverHOverlapped");
                                        subTensorsGen(model, {inputTensor, sourceOpInputTensor}, numClusters, pass);
                                    }
                                }
                            }
                        }

                        if (sinkOperators[0]->getOpType() == "Align")
                        {
                            auto nextSinkOperators = findSinkLayers(dm, sinkOperators[0]->getOutputTensor(0));
                            if (nextSinkOperators[0]->getOpType() == "DPUTask")
                            {
                                if (nextSinkOperators[0]->get<std::string>("taskOp") == "ChannelMajorConvolution")
                                {
                                     if (nextSinkOperators[0]->get<std::string>("splitStrategy") == "SplitOverH")
                                     {
                                         outputTensor->setOrder(mv::Order(mv::Order::getColMajorID(4)));
                                         sinkOperators[0]->getOutputTensor(0)->setOrder(mv::Order(mv::Order::getColMajorID(4)));
                                         sinkOperators[0]->getInputTensor(0)->cleanSubtensors();
                                         sinkOperators[0]->getOutputTensor(0)->cleanSubtensors();
                                         sinkOperators[0]->getInputTensor(0)->set<std::string>("splitStrategy", "SplitOverHOverlapped");
                                         sinkOperators[0]->getOutputTensor(0)->set<std::string>("splitStrategy", "SplitOverHOverlapped");
                                         subTensorsGen(model, {sinkOperators[0]->getInputTensor(0), sinkOperators[0]->getOutputTensor(0)}, numClusters, pass);
                                     }
                                 }
                             }
                        }
                    }

                    //ASSUMPTION: all sink ops have the same strategy, except DMATask
                    if (sinkOperators[0]->getOpType() == "DMATask") {
                        continue;
                    }

                    auto opStrategy = sinkOperators[0]->get<std::string>("splitStrategy");
                    auto tensorStrategy = outputTensor->get<std::string>("splitStrategy");

                    std::pair<std::string, std::string> possibleCombination(tensorStrategy, opStrategy);
                    for (auto restrictedCombination: incompatibleStrategies)
                    {
                        if (possibleCombination == restrictedCombination)
                        {
                            // Strategies have to be adjusted...
                            outputTensor->set<std::string>("splitStrategy", opStrategy);
                            inputTensor->set<std::string>("splitStrategy", opStrategy);
                            inputTensor->cleanSubtensors();
                            outputTensor->cleanSubtensors();
                            if ((possibleCombination == clusteringToSoH || possibleCombination == SoKToSoH))
                                inputTensor->set<std::string>("overwriteStrategy", "ClusteringToSoH");
                            else if ((possibleCombination == SoHToClustering || possibleCombination == SoHToSoK))
                                inputTensor->set<std::string>("overwriteStrategy", "SoHToClustering");
                            // ... and splitting has to be done again!!! <- Price for efficiency
                            subTensorsGen(model, {inputTensor, outputTensor}, numClusters, pass);
                        }
                    }
                }
            }
        }
    }
    return;

}
