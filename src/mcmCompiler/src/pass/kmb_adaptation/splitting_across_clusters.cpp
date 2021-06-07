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

namespace {
    using tensorComparatorFunc =
        std::function<bool(
        mv::Data::TensorIterator, mv::Data::TensorIterator)>;

    using tensorSet = std::set<mv::Data::TensorIterator, tensorComparatorFunc>;

    bool compareTensor(mv::Data::TensorIterator t0, mv::Data::TensorIterator t1) {
        return t0->getName().compare(t1->getName()) < 0;
    }
}

static const std::vector<mv::DPUModeList> TENSOR_MPE {{{1,1}}, {{16,1}}, {{1,16}}};

static void SplittingTensorsAcrossClusters(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&,
                                    mv::Element&, mv::Element&);
static void subTensorsGen(mv::ComputationModel& model, const tensorSet &tensors, unsigned nClusters,
                          const mv::pass::PassEntry& pass, std::size_t id=0);
static void unpopulatedSplitOverH(const unsigned nWorkloads, std::vector<mv::Workload> &subTensors, mv::Workloads &Tensor,
                                  const mv::pass::PassEntry& pass, int &success);
static void populatedSplitOverH(const unsigned nClusters, std::vector<mv::Workload> &subTensors, mv::Workloads& Tensor,
                                const mv::pass::PassEntry& pass, int &success);
static std::vector<mv::Workload> fixRectangularHeuristicBug(std::vector<mv::Workload> subTensors, const mv::Data::TensorIterator &tensor, const std::size_t nWorkloads, const std::size_t outputChannels);
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
    auto insertTensor = [](tensorSet& tensors, const mv::Data::TensorIterator& tensor, const mv::Data::OpListIterator& parentOp) {
        if (tensor->hasSubTensors())
            return;

        if (!tensor->hasAttr("splitStrategy"))
        {
            if (parentOp->hasAttr("splitStrategy"))
                tensor->set<std::string>("splitStrategy", parentOp->get<std::string>("splitStrategy"));
            else if (parentOp->inputSlots() && parentOp->getInputTensor(0)->hasAttr("splitStrategy"))
                tensor->set<std::string>("splitStrategy", parentOp->getInputTensor(0)->get<std::string>("splitStrategy"));
        }
        tensors.insert(tensor);
    };

    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto globalParams = model.getGlobalConfigParams();
    unsigned int numClusters = (unsigned int)globalParams->get<int>("Number_of_Clusters");

    if (numClusters <= 1)
        return;

    tensorSet tensors(compareTensor);
    //NOTE: special Tensors are marked the tensors that are populated that need to be handled
    //as unpopulated, cases where compiler handes the activation sparsity!!!
    //terrible...compiler concludes a solver of hardware limitations
    tensorSet specialTensors(compareTensor);

    //Todo:: the construction and logic of this pass needs to be refactored.
    // The pass should not target specific ops to determine if it's output needs to have subtensors generated,
    // but via location. If  the outputTensor is in NNCMX, then it needs a clustering strategy, and subtensors
    // They can be also non DPUTasks like ConcatInCMX, Slice,Reshape etc...

    for (auto& op : om.getOps())
    {
        //Also need to generate subtensors for output tensor of input operation
        //  Note: Input can't provide output activation sparsity, so sparse subtensors
        // shouldn't be needed
        if (op->getOpType() == "Input" || op->getOpType() == "ImplicitInput")
        {
            for (const auto& tensor : op->getOutputTensor()) {
                if((tensor->get<std::string>("splitStrategy") == "SplitOverH") ||
                    (tensor->get<std::string>("splitStrategy") == "SplitOverHOverlapped"))
                    insertTensor(tensors, tensor, op);
            }
            continue;
        }

        //Also need to generate subtensors for input tensor of output operation
        //  Note: Output can't take input activation sparsity, so sparse subtensors
        // shouldn't be needed
        if (op->getOpType() == "Output" || op->getOpType() == "ImplicitOutput")
        {
            for (const auto& tensor : op->getInputTensor()) {
                if((tensor->get<std::string>("splitStrategy") == "SplitOverH") ||
                    (tensor->get<std::string>("splitStrategy") == "SplitOverHOverlapped"))
                    insertTensor(tensors, tensor, om.getSourceOp(tensor));
            }
            continue;
        }

        const auto outputTensor = op->getOutputTensor(mv::IO_TENSOR_OUTPUT);
        if (op->isImplicit())
        {
            for (const auto& tensor : op->getInputTensor()) {
                insertTensor(tensors, tensor, om.getSourceOp(tensor));
            }

            for (const auto& tensor : op->getOutputTensor()) {
                insertTensor(tensors, tensor, op);
            }

            continue;
        }

        if (op->getOpType() == "DPUTask")
        {
            insertTensor(tensors, outputTensor, op);

            for(std::size_t i = 0; i < op->inputSlots(); ++i)
            {
                const auto inputTensor = op->getInputTensor(i);
                if (findSparseTensorIndex(op, "unpopulatedSparsityMapIndex", i) ||
                    findSparseTensorIndex(op, "storageElementIndex", i))
                    insertTensor(specialTensors, inputTensor, om.getSourceOp(inputTensor));
                else
                {
                    insertTensor(tensors, inputTensor, om.getSourceOp(inputTensor));

                    // New weights sparsity approach: no explicit costant operation
                    // for sparsity map is present in the graph.
                    // So check for sparsity has to be done only here
                    if(inputTensor->isPopulated() && inputTensor->isSparse())
                    {
                        const auto sparsityMap = inputTensor->getSparsityMap();
                        const auto tensor = dm.isTensorDefined(sparsityMap) ?
                            dm.getTensor(sparsityMap->getName()) : dm.defineTensor(sparsityMap);
                        insertTensor(tensors, tensor, op);
                    }
                }
            }
        }
    }

    subTensorsGen(model, tensors, numClusters, pass);
    subTensorsGen(model, specialTensors, numClusters, pass, 1);

    tensors.clear();
    auto srcImplicitChainTypes = std::vector<std::string>({"Concat"});
    auto destImplicitChainTypes = std::vector<std::string>({"Align", "Crop", "ImplicitConcat", "Slice"});
    auto srcImplitcitOpsTypes = om.getOpsOfTypes(srcImplicitChainTypes);
    for(auto srcImplicitOpsList : srcImplitcitOpsTypes)
    {
        for(auto srcImplicitOp : srcImplicitOpsList.second)
        {
            const auto outputTensor = srcImplicitOp->getOutputTensor(mv::IO_TENSOR_OUTPUT);
            auto sinkOperators = findSinkLayers(dm, outputTensor);

            if (sinkOperators.empty())
                continue;

            if (std::find(destImplicitChainTypes.cbegin(), destImplicitChainTypes.cend(),
                sinkOperators[0]->getOpType()) != destImplicitChainTypes.cend())
                insertTensor(tensors, outputTensor, srcImplicitOp);
        }
    }
    subTensorsGen(model, tensors, numClusters, pass);

}

void subTensorsGen(mv::ComputationModel& model, const tensorSet& tensors, unsigned nClusters,
                   const mv::pass::PassEntry& pass, std::size_t id)
{
    mv::DataModel dm(model);
    unsigned nWorkloads = nClusters;

    if (id == 0)
    {
        for (auto& tensor : tensors)
        {
            int success;
            mv::Workloads Tensor(tensor->getName(), tensor->getShape());
            std::vector<mv::Workload> subTensors;

            if (tensor->get<std::string>("splitStrategy") == "SplitOverH")
            {
                if(!tensor->isPopulated())
                {
                    // special handling for tensor 1x1, in fact switch to Clustering
                    // TODO: support tensors less than 1x4
                    auto tensorShape = tensor->getShape();
                    auto W = tensorShape[mv::IO_WIDTH_DIMENSION];
                    auto H = tensorShape[mv::IO_HEIGHT_DIMENSION];
                    auto Z = tensorShape[mv::IO_CHANNEL_DIMENSION];
                    if (W == 1 && H == 1) {
                        for (unsigned i = 0; i < nWorkloads; i++)
                        {
                            mv::Workload subTensor;
                            subTensor.MaxX = W;
                            subTensor.MinX = 0;
                            subTensor.MaxZ = Z;
                            subTensor.MinZ = 0;
                            subTensor.MaxY = H;
                            subTensor.MinY = 0;
                            subTensors.push_back(subTensor);
                        }
                        tensor->shareAcrossClusters(subTensors, nWorkloads);
                        continue;
                    }
                    else {
                        unpopulatedSplitOverH(nClusters, subTensors, Tensor, pass, success);
                    }
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
                        masterTile.setSize(sinkOperators[0]->getInputTensor(mv::IO_TENSOR_INPUT)->getShape());
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
                        while (sinkOperators[0]->getOpType() != "DPUTask" &&
                                sinkOperators[0]->getOpType() != "Output"){
                            sinkOperators = findSinkLayers(dm, sinkOperators[0]->getOutputTensor(mv::IO_TENSOR_OUTPUT));
                        }
                        if(sinkOperators[0]->getOpType() != "Output")
                            outputChannels = sinkOperators[0]->getOutputTensor(mv::IO_TENSOR_OUTPUT)->getShape()[mv::IO_CHANNEL_DIMENSION];
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
            mv::Workloads Tensor(tensor->getName(), tensor->getShape());
            std::vector<mv::Workload> subTensors;

            if (std::find(activationSegmentableStrategies.cbegin(), activationSegmentableStrategies.cend(),
                tensor->get<std::string>("splitStrategy")) != activationSegmentableStrategies.cend())
            {
                unpopulatedSplitOverH(nClusters, subTensors, Tensor, pass, success);
                tensor->splitPopulatedActivationAcrossClusters(subTensors, true, false);
            }
            else
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
static std::vector<mv::Workload> fixRectangularHeuristicBug(std::vector<mv::Workload> subTensors, const mv::Data::TensorIterator &tensor, const std::size_t nWorkloads, const std::size_t outputChannels)
{
    std::vector<mv::Workload> newSubTensors;

    if (!tensor->isPopulated())
    {
        if (subTensors.empty())
        {
            const std::size_t quantumofAlignedChannels = outputChannels/16;
            const std::size_t unalignedRemainder = outputChannels % 16;
            const std::size_t equalSlice = quantumofAlignedChannels/nWorkloads;
            const std::size_t remainingSlice = quantumofAlignedChannels%nWorkloads;

            for (std::size_t n = 0; n < nWorkloads; n++)
            {
                mv::Workload subTensor;
                if (n != nWorkloads - 1)
                {
                    subTensor.MaxZ = tensor->getShape()[mv::IO_WIDTH_DIMENSION] - 1;
                    subTensor.MinZ = 0;
                    subTensor.MaxX = equalSlice * (n+1) * 16 - 1;
                    subTensor.MinX = equalSlice * n * 16;
                    subTensor.MaxY = tensor->getShape()[mv::IO_HEIGHT_DIMENSION] - 1;
                    subTensor.MinY = 0;
                }
                else
                {
                    subTensor.MaxZ = tensor->getShape()[mv::IO_WIDTH_DIMENSION] - 1;
                    subTensor.MinZ = 0;
                    subTensor.MaxX = (equalSlice * (n+1) + remainingSlice) * 16 + unalignedRemainder - 1;
                    subTensor.MinX = equalSlice * n * 16;
                    subTensor.MaxY = tensor->getShape()[mv::IO_HEIGHT_DIMENSION] - 1;
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

            std::size_t t = 0;
            while (totalZ > 0)
            {
                z_sizes[t] += 16;
                totalZ -= 16;
                t++;
                if (t == nWorkloads)
                    t = 0;
            }
            for (std::size_t i = 0; i < nWorkloads; i++)
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

        std::size_t t = 0;
        while (totalZ > 0)
        {
            z_sizes[t] += 16;
            totalZ -= 16;
            t++;
            if (t == nWorkloads)
                t = 0;
        }
        for (std::size_t i = 0; i < nWorkloads; i++)
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

void ensureSplitStrategiesForSpilling(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element&, mv::Element&)
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
    std::pair<std::string, std::string> HKSwitchToSoH("HKSwitch", "SplitOverH");
    std::pair<std::string, std::string> HKSwitchToHKSwitch("HKSwitch", "HKSwitch");
    std::pair<std::string, std::string> SoKToHKSwitch("SplitOverK", "HKSwitch");
    std::pair<std::string, std::string> ClusteringToHKSwitch("Clustering", "HKSwitch");
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
                auto outputTensor = opIt->getOutputTensor(mv::IO_TENSOR_OUTPUT);
                auto inputTensor = opIt->getInputTensor(mv::IO_TENSOR_INPUT);

                auto sourceOp = om.getSourceOp(inputTensor);

                // Ensure input and output tensors of slice have compatible splitStrategy,
                // so that the following DMA can move the tensor correctly.
                if (sourceOp->getOpType() == "Slice")
                {
                    const std::pair<std::string, std::string> inOutStrategies(
                        sourceOp->getInputTensor(mv::IO_TENSOR_INPUT)->get<std::string>("splitStrategy"),
                        sourceOp->getOutputTensor(mv::IO_TENSOR_OUTPUT)->get<std::string>("splitStrategy"));

                    if (inOutStrategies == SoKToSoH)
                    {
                        inputTensor->cleanSubtensors();
                        inputTensor->set<std::string>("splitStrategy", inOutStrategies.first);
                        outputTensor->cleanSubtensors();
                        outputTensor->set<std::string>("splitStrategy", inOutStrategies.first);

                        tensorSet tensors(compareTensor);
                        tensors.insert({inputTensor, outputTensor});
                        subTensorsGen(model, tensors, numClusters, pass);
                    }
                }

                if (opIt->get<mv::DmaDirection>("direction") == mv::DmaDirectionEnum::DDR2NNCMX &&
                    !outputTensor->isPopulated())
                {
                    std::vector<mv::Data::OpListIterator> sinkOperators = findSinkLayers(dm, outputTensor);

                    //FOR THE COPY OPERATION RESPONSIBLE IS ONLY THE PASS THAT  ASSIGNS THE STRATEGIES
                    //NOT FOR ALL THE IMPLICIT CAUSE WE MIGHT HAVE SLICE WHEN WE STREAM
                    if (sinkOperators[0]->getOpType() == "Copy" )
                        continue;

                    if (opIt->getOutputTensor(mv::IO_TENSOR_OUTPUT)->get<std::string>("splitStrategy") != "SplitOverHOverlapped")
                    {
                        if (sinkOperators[0]->getOpType() == "DPUTask")
                        {
                            if (sinkOperators[0]->get<std::string>("taskOp") == "ChannelMajorConvolution" &&
                                td.getTarget() != mv::Target::ma3720)
                            {
                                if (sinkOperators[0]->get<std::string>("splitStrategy") == "SplitOverH")
                                {
                                    outputTensor->setOrder(mv::Order(mv::Order::getColMajorID(4)));
                                    inputTensor->cleanSubtensors();
                                    outputTensor->cleanSubtensors();
                                    outputTensor->set<std::string>("splitStrategy", "SplitOverHOverlapped");
                                    inputTensor->set<std::string>("splitStrategy", "SplitOverHOverlapped");

                                    tensorSet tensors(compareTensor);
                                    tensors.insert({inputTensor, outputTensor});

                                    // support for input->slice->DMA->CMConv

                                    if (sourceOp->getOpType() == "Slice")
                                    {
                                        sourceOp->set<std::string>("splitStrategy", "SplitOverHOverlapped");
                                        auto sourceOpInputTensor = sourceOp->getInputTensor(mv::IO_TENSOR_INPUT);
                                        sourceOpInputTensor->cleanSubtensors();
                                        sourceOpInputTensor->set<std::string>("splitStrategy", "SplitOverHOverlapped");
                                        inputTensor->cleanSubtensors();
                                        inputTensor->set<std::string>("splitStrategy", "SplitOverHOverlapped");

                                        tensors.insert(sourceOpInputTensor);
                                    }

                                    subTensorsGen(model, tensors, numClusters, pass);
                                }
                            }
                        }

                        if (sinkOperators[0]->getOpType() == "Align")
                        {

                            auto sinkOutTensor = sinkOperators[0]->getOutputTensor(mv::IO_TENSOR_OUTPUT);
                            auto nextSinkOperators = findSinkLayers(dm, sinkOutTensor);
                            if (nextSinkOperators[0]->getOpType() == "DPUTask")
                            {
                                if (nextSinkOperators[0]->get<std::string>("taskOp") == "ChannelMajorConvolution" &&
                                    td.getTarget() != mv::Target::ma3720)
                                {
                                    if (nextSinkOperators[0]->get<std::string>("splitStrategy") == "SplitOverH")
                                    {
                                        outputTensor->setOrder(mv::Order(mv::Order::getColMajorID(4)));
                                        auto sinkInTensor = sinkOperators[0]->getInputTensor(mv::IO_TENSOR_INPUT);

                                        sinkOutTensor->setOrder(mv::Order(mv::Order::getColMajorID(4)));
                                        sinkInTensor->cleanSubtensors();
                                        sinkOutTensor->cleanSubtensors();
                                        sinkInTensor->set<std::string>("splitStrategy", "SplitOverHOverlapped");
                                        sinkOutTensor->set<std::string>("splitStrategy", "SplitOverHOverlapped");

                                        tensorSet tensors(compareTensor);
                                        tensors.insert({sinkInTensor, sinkOutTensor});
                                        subTensorsGen(model, tensors, numClusters, pass);
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

                            // Store SoH splits before cleaning subtensors
                            auto splitShapes = std::vector<mv::Shape>();
                            if (possibleCombination == SoHToClustering || possibleCombination == SoHToSoK)
                            {
                                for (unsigned i=0; i<numClusters; ++i)
                                {
                                    splitShapes.push_back(inputTensor->getSubTensor(i).getShape());
                                }
                            }

                            tensorSet setSubs(compareTensor);
                            outputTensor->cleanSubtensors();
                            if (possibleCombination == clusteringToSoH || possibleCombination == HKSwitchToHKSwitch ||
                                 possibleCombination == SoKToSoH || possibleCombination == SoKToHKSwitch ||
                                 possibleCombination == ClusteringToHKSwitch || possibleCombination == HKSwitchToSoH)
                            {
                                inputTensor->cleanSubtensors();
                                inputTensor->set<std::string>("overwriteStrategy", "ClusteringToSoH");
                                setSubs = {inputTensor, outputTensor};
                            }
                            else if ((possibleCombination == SoHToClustering || possibleCombination == SoHToSoK))
                            {
                                inputTensor->set<std::string>("overwriteStrategy", "SoHToClustering");
                                setSubs = {outputTensor};
                            }

                            // ... and splitting has to be done again!!! <- Price for efficiency
                            subTensorsGen(model, setSubs, numClusters, pass);

                        }
                    }
                }
            }
        }
    }
    return;

}
