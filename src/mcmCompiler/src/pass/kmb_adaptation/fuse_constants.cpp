#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/kmb/ppe_task.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void fuseConstantsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void markDMATasksToIgnoreFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void setNotFusedDataIndexesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void repopulateFusedConstantsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(FuseConstants)
            .setFunc(fuseConstantsFcn)
            .setDescription(
                "The idea of this pass is the following: Gather all the dpuTasks which has more than one populated input and fuse them.");

        MV_REGISTER_PASS(MarkDMATasksToIgnore)
            .setFunc(markDMATasksToIgnoreFcn)
            .setDescription(
                "The idea of this pass is the following: Gather the dmas that do not need to be scheduled.");

        MV_REGISTER_PASS(SetNotFusedDataIndexes)
            .setFunc(setNotFusedDataIndexesFcn)
            .setDescription(
                "The idea of this pass is the following: Compute addresses for the non-scheduled dmas.");

        MV_REGISTER_PASS(RepopulateFusedConstants)
            .setFunc(repopulateFusedConstantsFcn)
            .setDescription(
                "The idea of this pass is the following: RepopulateFusedConstants after population of other constants.");
    }
}

bool hasPopulatedInputs(const std::string& taskOp)
{
    return (taskOp == "DepthwiseConv" || taskOp == "MaxPool" || taskOp == "Conv" || taskOp == "ChannelMajorConvolution");
}

bool isWeightsSparse(const mv::Data::OpListIterator& task)
{
    return task->hasAttr("weightsSparsity") && task->get<bool>("weightsSparsity");
}

std::vector<std::size_t> setFusionOrder(mv::Data::OpListIterator& opIt)
{
    std::vector<std::size_t> fusionOrder;
    // fusion order = weight table, fake sparsity, weight sparsity, weights
    if (opIt->hasAttr("weightsTableIndex"))
        fusionOrder.push_back(opIt->get<std::size_t>("weightsTableIndex"));
    if (opIt->hasAttr("fakeSparsityIndex"))
        fusionOrder.push_back(opIt->get<std::size_t>("fakeSparsityIndex"));
    // NOTE: weight sparsity has no index as part of weights
    if (opIt->hasWeights())
        fusionOrder.push_back(mv::IO_TENSOR_WEIGHTS_SET);
    // store attribute for future iterations
    opIt->set<std::vector<std::size_t>>("fusionOrder", fusionOrder);
    return fusionOrder;
}

std::size_t shiftForU8Conversion(std::size_t offset)
{
    return ((8 * offset) & 0xFF);
}

bool opIsSharingWeights(mv::Data::OpListIterator& opIt)
{
    // skip operations sharing weights
    return (opIt->hasAttr("shareWeights") && opIt->get<bool>("shareWeights"))
        || (opIt->hasAttr("DilatedSubConv") && opIt->get<bool>("DilatedSubConv"))
        || (opIt->hasAttr("slicingV2") && opIt->get<bool>("slicingV2"));
}

void fuseConstantsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto globalParams = model.getGlobalConfigParams();

    // Skip pass if Tensor Fusion isn't enabled
    if (!(globalParams->hasAttr("enableTensorFusion") && globalParams->get<bool>("enableTensorFusion")))
        return;

    std::vector<int64_t> zp = { 0 };
    std::vector<double> min = { 1 };
    std::vector<double> max = { 1 };
    std::vector<double> scale = { 1 };
    mv::QuantizationParams neutralQuantization(zp, scale, min, max);

    for (auto& opIt : om.getOps("DPUTask"))
    {
        if (hasPopulatedInputs(opIt->get<std::string>("taskOp")) 
            && !opIsSharingWeights(opIt))
        {
            auto inputTensors = opIt->getInputTensor();
            if (inputTensors.size() < 2) { continue; }
            std::vector<std::size_t> fusionOrder = setFusionOrder(opIt);
            if (fusionOrder.empty()) { continue; }
            std::vector<mv::DataElement> fusedData = {};
            std::size_t totalShape = 0;
            std::string splitStrategy = opIt->get<std::string>("splitStrategy");
            bool isSparseWeights = isWeightsSparse(opIt);

            for (auto tensorIndex = fusionOrder.begin(); tensorIndex != fusionOrder.end(); ++tensorIndex)
            {
                auto inputTensor = opIt->getInputTensor(*tensorIndex);
                std::vector<mv::DataElement> populatedData = {};

                if (isSparseWeights && *tensorIndex == mv::IO_TENSOR_WEIGHTS_SET)
                {
                    auto weightsTensor = inputTensors[mv::IO_TENSOR_WEIGHTS_SET];
                    auto sparsityMap = dm.getTensor(weightsTensor->getSparsityMap()->getName());
                    auto smData = sparsityMap->getData();
                    fusedData.insert(fusedData.end(), smData.begin(), smData.end());
                    totalShape += sparsityMap->getShape().totalSize() *
                            (sparsityMap->getDType().getSizeInBits() / 8);
                    std::vector<int64_t> populatedDataPacked = weightsTensor->getDataPacked();
                    for (std::size_t i = 0; i < populatedDataPacked.size(); ++i)
                    {
                        mv::DataElement de(false, populatedDataPacked.at(i));
                        populatedData.push_back(de);
                    }
                }
                else
                {
                    populatedData = inputTensor->getData();
                }

                // convert to U8 representation
                int64_t populatedDataU8Size = populatedData.size() * (inputTensor->getDType().getSizeInBits()/8);
                mv::DataElement zeroPoint(false, (int64_t) 0);
                std::vector<mv::DataElement> populatedU8Data (populatedDataU8Size, zeroPoint);
                for (std::size_t i = 0; i < populatedData.size(); ++i)
                {
                    for (auto dt_idx = 0UL; dt_idx < inputTensor->getDType().getSizeInBits()/8; ++dt_idx)
                    {
                        populatedU8Data.at(dt_idx + i * inputTensor->getDType().getSizeInBits()/8) = 
                            (populatedData.at(i) >> shiftForU8Conversion(dt_idx));
                    }
                }
                fusedData.insert(fusedData.end(), populatedU8Data.begin(), populatedU8Data.end());
                totalShape += populatedDataU8Size;
            }

            auto fusedConstant = om.constantDataElement(opIt->getName() + "_fusedConstants", fusedData,
                                        {1, 1, 1, totalShape},
                                        mv::DType("UInt8"), mv::Order::getZMajorID(4));
            fusedConstant->setQuantParams(neutralQuantization);
            auto fusedOp = om.getSourceOp(fusedConstant);
            fusedOp->set<std::string>("splitStrategy", splitStrategy);
            fusedConstant->set<std::string>("splitStrategy", splitStrategy);
            fusedConstant->set<bool>("fusedTensor", true);
            if (opIt->hasAttr("opId"))
            {
                unsigned currentOpId = opIt->get<unsigned>("opId");
                fusedOp->set<unsigned>("opId", currentOpId);
            }
            auto newInputsSize = opIt->addInputTensor(fusedConstant);
            om.defineFlow(fusedConstant, opIt, newInputsSize - 1);
            opIt->set<size_t>("fusedConstantIndex", newInputsSize - 1);
        }
    }
}

void markDMATasksToIgnoreFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    // The DMAs are marked as ignored rather than being removed as the entire process of fusion does not 
    // modify the current process of handling/populating DPU Constants which can have different dTypes. 
    // Instead, initially a fused tensor represents the space needed to be allocated and will be scheduled 
    // accordingly. Later the data from the tensors which are fused is moved over to the fused tensor 
    // and stored as U8 - making the process of fusion easier.
    
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    for (auto& opIt : om.getOps("DPUTask"))
    {
        if (opIt->hasAttr("fusedConstantIndex") && opIt->hasAttr("fusionOrder"))
        {
            std::vector<std::size_t> fusionOrder = opIt->get<std::vector<std::size_t>>("fusionOrder");
            auto inputTensors = opIt->getInputTensor();

            for (auto tensorIndex = fusionOrder.begin(); tensorIndex != fusionOrder.end(); ++tensorIndex)
            {
                auto inputTensor = inputTensors[*tensorIndex];
                auto dmaTask = om.getSourceOp(inputTensor);
                dmaTask->set<bool>("toIgnore", true);
                dmaTask->getInputTensor(0)->set<bool>("toIgnore", true);
            }
        }
    }
}

void setNotFusedDataIndexesFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    for (auto& opIt : om.getOps("DPUTask"))
    {
        if (opIt->hasAttr("fusedConstantIndex") && opIt->hasAttr("fusionOrder"))
        {
            std::vector<std::size_t> fusionOrder = opIt->get<std::vector<std::size_t>>("fusionOrder");
            std::size_t fusedConstantIndex = opIt->get<std::size_t>("fusedConstantIndex");
            bool sparseWeights = isWeightsSparse(opIt);
            auto inputTensors = opIt->getInputTensor();
            auto baseAddress = inputTensors[fusedConstantIndex]->getAddress();
            auto strategy = opIt->get<std::string>("splitStrategy");
            int64_t tempAddress = baseAddress;

            for (auto tensorIndex = fusionOrder.begin(); tensorIndex != fusionOrder.end(); ++tensorIndex)
            {
                auto inputTensor = inputTensors[*tensorIndex];
                if (sparseWeights && *tensorIndex == mv::IO_TENSOR_WEIGHTS_SET)
                {
                    inputTensor = dm.getTensor(inputTensor->getSparsityMap()->getName());
                }

                inputTensor->set<std::size_t>("fusedOffset", tempAddress - baseAddress);
                inputTensor->setAddress(tempAddress);
                std::size_t offset = inputTensor->getShape().totalSize() * (inputTensor->getDType().getSizeInBits() / 8);
                tempAddress += offset;

                if (strategy == "SplitOverK")
                {
                    auto fusedConstantData = inputTensors[fusedConstantIndex];
                    std::vector<std::size_t> fusedClusterOffsets = inputTensor->get<std::vector<std::size_t>>("fusedClusterOffsets");
                    for (auto sub_idx = 0UL; sub_idx < inputTensor->numSubTensors(); ++sub_idx)
                    {
                        auto fusedSubAddress = fusedConstantData->getSubTensor(sub_idx).getAddress();
                        inputTensor->getSubTensor(sub_idx).set<std::size_t>("fusedOffset", fusedClusterOffsets.at(sub_idx));
                        inputTensor->getSubTensor(sub_idx).setAddress(fusedSubAddress + fusedClusterOffsets.at(sub_idx));
                        if (sparseWeights && *tensorIndex == mv::IO_TENSOR_WEIGHTS_SET)
                        {
                            inputTensors[mv::IO_TENSOR_WEIGHTS_SET]->getSubTensor(sub_idx).set<std::size_t>("sparsityMapAddress", fusedSubAddress + fusedClusterOffsets.at(sub_idx));
                        }
                    }
                }

                if (sparseWeights && *tensorIndex == mv::IO_TENSOR_WEIGHTS_SET)
                {
                    sparseWeights = false;
                    --tensorIndex;
                }
            }
        }
    }
}

void populateAsU8ConstantSubTensors(mv::Data::TensorIterator& populatedTensor, mv::Data::TensorIterator& fusedConstantData)
{
    std::vector<std::size_t> fusedClusterOffsets = populatedTensor->get<std::vector<std::size_t>>("fusedClusterOffsets");
    for (auto sub_idx = 0UL; sub_idx < populatedTensor->numSubTensors(); ++sub_idx)
    {
        auto fusedSubAddress = fusedConstantData->getSubTensor(sub_idx).getAddress();
        populatedTensor->getSubTensor(sub_idx).set<std::size_t>("address", fusedSubAddress + fusedClusterOffsets.at(sub_idx));
        for (auto p_t_idx = 0UL; p_t_idx < populatedTensor->getSubTensor(sub_idx).size(); ++p_t_idx)
        {
            for (auto d_t_indx = 0UL; d_t_indx < populatedTensor->getDType().getSizeInBits()/8; ++d_t_indx)
            {
                fusedConstantData->getSubTensor(sub_idx).at(d_t_indx + p_t_idx * populatedTensor->getDType().getSizeInBits()/8 + fusedClusterOffsets.at(sub_idx)) =
                    (populatedTensor->getSubTensor(sub_idx).at(p_t_idx) >> shiftForU8Conversion(d_t_indx));
            }
        }
    }
}

void repopulateFusedConstantsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto dpuTasks = om.getOps("DPUTask");
    for (auto& opIt : dpuTasks)
    {
        if (opIt->hasAttr("fusedConstantIndex") && opIt->hasAttr("fusionOrder") 
            && hasPopulatedInputs(opIt->get<std::string>("taskOp")))
        {
            std::vector<std::size_t> fusionOrder = opIt->get<std::vector<std::size_t>>("fusionOrder");
            std::size_t fusedConstantIndex = opIt->get<std::size_t>("fusedConstantIndex");
            auto strategy = opIt->get<std::string>("splitStrategy");
            bool sparseWeights = isWeightsSparse(opIt);
            auto inputTensors = opIt->getInputTensor();
            auto fusedConstantData = opIt->getInputTensor()[fusedConstantIndex];

            for (auto tensorIndex = fusionOrder.begin(); tensorIndex != fusionOrder.end(); ++tensorIndex)
            {
                // 1. Weight Table populated need to be copied over
                if (strategy != "SplitOverK")
                {
                    if (*tensorIndex == opIt->get<std::size_t>("weightsTableIndex"))
                    {
                        auto weightsTable = inputTensors[*tensorIndex];
                        for (auto i = 0UL; i < weightsTable->size(); i++)
                        {
                            for (auto dt_idx = 0UL; dt_idx < weightsTable->getDType().getSizeInBits()/8; ++dt_idx)
                            {
                                //NOTE: the idea is that i shift so many times in order to go the useful field of weights
                                //table in the last 8 significant bits and then keep them
                                fusedConstantData->at(i * weightsTable->getDType().getSizeInBits()/8 + dt_idx) =
                                        (weightsTable->at(i) >> shiftForU8Conversion(dt_idx));
                            }
                        }
                    }
                }
                // 2. SOK: subtensors need population
                else
                {
                    std::vector<std::size_t> fusedClusterOffsets = inputTensors[*tensorIndex]->get<std::vector<std::size_t>>("fusedClusterOffsets");
                    if (sparseWeights && *tensorIndex == mv::IO_TENSOR_WEIGHTS_SET)
                    {
                        // NOTE: Special case for sparse weights
                        auto weightTensor = inputTensors[*tensorIndex];
                        auto weightSparsityTensor = dm.getTensor(weightTensor->getSparsityMap()->getName());
                        std::vector<std::size_t> fusedSparseOffsets = weightSparsityTensor->get<std::vector<std::size_t>>("fusedClusterOffsets");
                        populateAsU8ConstantSubTensors(weightSparsityTensor, fusedConstantData);
                        for (auto sub_idx = 0UL; sub_idx < weightTensor->numSubTensors(); sub_idx++)
                        {
                            auto fusedSubAddress = fusedConstantData->getSubTensor(sub_idx).getAddress();
                            weightTensor->getSubTensor(sub_idx).set<std::size_t>("address", fusedSubAddress + fusedClusterOffsets.at(sub_idx));
                            std::vector<mv::DataElement> populatedData;
                            std::vector<int64_t> populatedDataPacked = weightTensor->getSubTensor(sub_idx).getDataPacked();
                            for (std::size_t i = 0; i < populatedDataPacked.size(); ++i)
                            {
                                mv::DataElement de(false, populatedDataPacked.at(i));
                                populatedData.push_back(de);
                            }
                            for (std::size_t weight_idx = 0; weight_idx < populatedData.size(); ++weight_idx)
                            {
                                for (auto dt_idx = 0UL; dt_idx < weightTensor->getDType().getSizeInBits()/8; ++dt_idx)
                                {
                                    fusedConstantData->getSubTensor(sub_idx).at(dt_idx + weight_idx * weightTensor->getDType().getSizeInBits()/8 + fusedClusterOffsets.at(sub_idx)) = 
                                        (populatedData.at(weight_idx) >> shiftForU8Conversion(dt_idx));
                                }
                            }
                        }
                    }
                    else
                    {
                        populateAsU8ConstantSubTensors(inputTensors[*tensorIndex], fusedConstantData);
                    }
                }
            }
        }
    }
}
