#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/target/kmb/ppe_task.hpp"
#include <cmath>
#include <math.h>

static const std::size_t WT_ELEMENTS_PER_CHANNEL = 4;
static const std::size_t ALU_HALT_OPCODE = 6;
static const std::size_t ALU_LOAD = 2;
static void generateWeightsTablesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void generateInstructionListTablesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void populateWeightsTablesPointersFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void populateWeightsTablesQuantizationFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void removeBiasTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void populateStorageElementPointersFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void populateInstructionListTablesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(GenerateWeightsTables)
        .setFunc(generateWeightsTablesFcn)
        .setDescription(
            "Generates weights tables for the Tasks that need them"
        );
        MV_REGISTER_PASS(GenerateInstructionListTables)
        .setFunc(generateInstructionListTablesFcn)
        .setDescription(
            "Generates instruction list tables for the Tasks that need them"
        );
        MV_REGISTER_PASS(PopulateWeightsTablesPointers)
        .setFunc(populateWeightsTablesPointersFcn)
        .setDescription(
            "Populate WeightsTables"
        );
        MV_REGISTER_PASS(PopulateWeightsTablesQuantization)
        .setFunc(populateWeightsTablesQuantizationFcn)
        .setDescription(
            "Populate WeightsTables"
        );
        MV_REGISTER_PASS(RemoveBiasTensors)
        .setFunc(removeBiasTensorsFcn)
        .setDescription(
            "remove bias tensors after been adding to all weight tables"
        );
        MV_REGISTER_PASS(PopulateStorageElementPointers)
        .setFunc(populateStorageElementPointersFcn)
        .setDescription(
            "Populate storage element maps for activations"
        );
        MV_REGISTER_PASS(PopulateInstructionListTables)
        .setFunc(populateInstructionListTablesFcn)
        .setDescription(
            "Populate instruction list tables for PWL solution"
        );
    }
}

void populatePointerMultiCluster(mv::Data::TensorIterator weightsTableData, mv::Data::OpListIterator dpuTaskOp, const std::vector<int64_t>& increments, long int offset, std::size_t addingIndex, mv::ComputationModel &model, mv::Data::TensorIterator tensor)
{
    //std::cout << "Populating " << std::to_string(addingIndex) << " pointer of weights table for op " << dpuTaskOp->getName() << std::endl;
    if (dpuTaskOp->get<std::string>("splitStrategy") != "SplitOverK")
    {
        for (size_t i = 0, k = 0; i < weightsTableData->size(); i+=WT_ELEMENTS_PER_CHANNEL, ++k)
            weightsTableData->at(i + addingIndex) = offset + increments[k];
    }
    else
    {
        // Saving a backup copy of the offset
        long int new_offset = offset;

        unsigned numClusters =  model.getGlobalConfigParams()->get<int>("Number_of_Clusters");
        std::size_t sizeToIterate = 0;
        std::size_t totalSizeToIterate = 0;
        std::size_t k = 0;
        bool isSparse = tensor->isSparse();
        for (unsigned i = 0; i < numClusters; i++)
        {
            // Resetting offset at the beginning of the cluster
            offset = new_offset;
            // Resetting k index only when weights are not sparse
            if(!isSparse)
                k = 0;

            // Filling cluster
            for (size_t j = 0; j < weightsTableData->getSubTensor(i).size(); j+=WT_ELEMENTS_PER_CHANNEL)
                weightsTableData->at(j + addingIndex + totalSizeToIterate) = offset + increments[k++];

            // Preparing for next iteration
            sizeToIterate = tensor->getSubTensor(i).getShape()[mv::KERNEL_OUTPUT_CHANNELS] * WT_ELEMENTS_PER_CHANNEL;
            totalSizeToIterate += sizeToIterate;
        }
    }
    return;
}

void populateWeightsTablesDataPointers(mv::Data::TensorIterator weightsTableData, mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
{
    auto taskOp = dpuTaskOp->get<std::string>("taskOp");

    // Max pooling does not need DataPointer
    // Eltwise doesn't have weights table at all
    if(taskOp == "Conv" || taskOp == "ChannelMajorConvolution" ||
       taskOp == "DepthwiseConv")
    {
        auto weights = dpuTaskOp->getInputTensor(1);

        // NOTE: Output channels taken from the weights, hence already padded
        auto outputChannels = weights->getShape()[mv::KERNEL_OUTPUT_CHANNELS];
        auto strategy = dpuTaskOp->get<std::string>("splitStrategy");

        long int offset = weights->getAddress();
        std::vector<int64_t> increments;

        if(weights->isSparse())
        {
            if(strategy != "SplitOverK")
                increments = weights->getKernelDataOffsets();
            else
            {
                unsigned numClusters =  model.getGlobalConfigParams()->get<int>("Number_of_Clusters");
                for(std::size_t i = 0; i < numClusters; ++i)
                {
                    auto kernelOffsets = weights->getSubTensor(i).getKernelDataOffsets();
                    increments.insert(increments.end(), kernelOffsets.begin(), kernelOffsets.end());
                }
            }
        }
        else
        {
            long int increment = weights->getShape()[mv::KERNEL_WEIGHT_SETS] * (weights->getDType().getSizeInBits() / 8);
            increments = std::vector<int64_t>(outputChannels, 0);
            for(unsigned i = 1; i < outputChannels; ++i)
                increments[i] = increments[i-1] + increment;
        }
        populatePointerMultiCluster(weightsTableData, dpuTaskOp, increments, offset, 0, model, weights);
    }
}



void populateWeightsTablesSparsityPointers(mv::Data::TensorIterator weightsTableData, mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model, mv::TargetDescriptor& td)
{
    mv::DataModel dm(model);

    auto taskOp = dpuTaskOp->get<std::string>("taskOp");
    if(taskOp == "Conv")
    {
        auto weights = dpuTaskOp->getInputTensor(1);

        // NOTE: Output channels taken from the weights, hence already padded
        auto outputChannels = weights->getShape()[mv::KERNEL_OUTPUT_CHANNELS];
        if(weights->isSparse())
        {
            auto weightsSparsityMap = dm.getTensor(weights->getSparsityMap()->getName());
            long int offset = weightsSparsityMap->getAddress();
            auto sparsityMapSizeInWords = weightsSparsityMap->getShape().totalSize();
            auto sparsityMapSizeInBytes = sparsityMapSizeInWords * weightsSparsityMap->getDType().getSizeInBits() / 8;
            long int increment = sparsityMapSizeInBytes / outputChannels;

            std::vector<int64_t> increments = std::vector<int64_t>(outputChannels, 0);
            for(unsigned i = 1; i < outputChannels; ++i)
                increments[i] = increments[i-1] + increment;

            populatePointerMultiCluster(weightsTableData, dpuTaskOp, increments, offset, 1, model, weightsSparsityMap);
        }
        else
        {
            // Dense ZMajor Convolution case
            // Not using the generic function because it's a super simple case
            int64_t offset = 0xFFFFFF; // NOTE: Implementation defined
            for (size_t i = 0; i < weightsTableData->size(); i+=WT_ELEMENTS_PER_CHANNEL)
                  weightsTableData->at(i+1) = offset;
        }
    }
    else if(taskOp == "DepthwiseConv"  ||
            (taskOp == "ChannelMajorConvolution" && td.getTarget() != mv::Target::ma3720) ||
            taskOp == "MaxPool")
    {
        // We have fake sparsity here! Yuppi!
        auto activationWindow = dpuTaskOp->getInputTensor(dpuTaskOp->get<std::size_t>("fakeSparsityIndex"));
        auto activationWindowSizeInWords = activationWindow->getShape().totalSize();
        auto activationWindowSizeInBytes = activationWindowSizeInWords * activationWindow->getDType().getSizeInBits() / 8;

        auto outputChannels = dpuTaskOp->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION];

        auto paddedOutputChannels = outputChannels;
        auto globalParams = model.getGlobalConfigParams();
        int pad = globalParams->hasAttr("VPU2ChannelPadding") ? globalParams->get<int>("VPU2ChannelPadding") : 16;

        paddedOutputChannels = mv::round_up(outputChannels, pad);
        auto increment = activationWindowSizeInBytes / paddedOutputChannels;

        long int offset = activationWindow->getAddress();
        std::vector<int64_t> increments = std::vector<int64_t>(paddedOutputChannels, 0);
        for(unsigned i = 1; i < paddedOutputChannels; ++i)
            increments[i] = increments[i-1] + increment;

        populatePointerMultiCluster(weightsTableData, dpuTaskOp, increments, offset, 1, model, activationWindow);
    }
}
#include "mcm/utils/custom_math.hpp"


void populateWeightsTablesActivationAndBias(mv::Data::TensorIterator weightsTableData, mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model,
        mv::TargetDescriptor& td)
{
    mv::DataModel dm(model);
    mv::QuantizationParams quantParams = {{},{},{},{}};
    auto output = dpuTaskOp->getOutputTensor(0);
    auto outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];
    auto paddedOutputChannels = outputChannels;
    auto globalParams = model.getGlobalConfigParams();
    int pad = globalParams->hasAttr("VPU2ChannelPadding") ? globalParams->get<int>("VPU2ChannelPadding") : 16;

    paddedOutputChannels = mv::round_up(outputChannels, pad);

    std::vector<int32_t> mScaled(paddedOutputChannels, 0);
    std::vector<int32_t> mShift(paddedOutputChannels, 0);
    if(output->hasAttr("quantParams"))
    {
        if (dpuTaskOp->hasAttr("pwlQuantParams"))
            quantParams = dpuTaskOp->get<mv::QuantizationParams>("pwlQuantParams");
        else
            quantParams = dpuTaskOp->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams");
        if (!quantParams.isEmpty())
        {
            auto mult = quantParams.getMult();
            auto shift = quantParams.getShift();
            if (output->hasAttr("preAdjustedShift") && output->hasAttr("preAdjustedMult"))
            {
                shift = output->get<std::vector<unsigned>>("preAdjustedShift");
                mult = output->get<std::vector<unsigned>>("preAdjustedMult");
            }
            std::transform(mScaled.begin(), mScaled.end(), mult.begin(), mScaled.begin(), std::plus<int32_t>());
            std::transform(mShift.begin(), mShift.end(), shift.begin(), mShift.begin(), std::plus<int32_t>());
            for (size_t idx = outputChannels; idx < paddedOutputChannels; idx++)
            {
                mScaled[idx] = mScaled[0];
                mShift[idx] = mShift[0];
            }
        }
    }
    std::vector<mv::DataElement> biasData;
    bool hasBias = dpuTaskOp->hasAttr("bias");
    bool hasPPETask = dpuTaskOp->hasAttr("PPETask");

    mv::Data::TensorIterator bias;
    if (hasBias)
    {
        bias = dm.getTensor(dpuTaskOp->get<std::string>("bias"));
        biasData = bias->getData(); //Bias has the type Int32 in both cases above
    }

    bool floatScaleTable = false;
    if (dpuTaskOp->hasAttr("floatScale"))
        floatScaleTable = true;

    if (floatScaleTable)
    {
        auto mScale = dpuTaskOp->get<std::vector<float>>("floatScale");
        for (size_t i = 0; i < weightsTableData->size(); i+=WT_ELEMENTS_PER_CHANNEL)
        {
            weightsTableData->at(i+2) = static_cast<int64_t>(mv::float_as_int(mScale[i/WT_ELEMENTS_PER_CHANNEL]));
            if (hasBias)
                weightsTableData->at(i+3) = static_cast<int64_t>(mv::float_as_int(biasData[i/WT_ELEMENTS_PER_CHANNEL]));
        }
    }
    else
    {
        unsigned round_mode = 1;
        std::vector<int32_t> round32(outputChannels, round_mode);
        std::vector<int32_t> reluMultData(outputChannels, 0);
        if (hasPPETask && td.getTarget() != mv::Target::ma3720)
        {
            auto ppeFF = dpuTaskOp->get<mv::PPETask>("PPETask").getFixedFunction();
            auto& ppeLayers = ppeFF.getLayers();
            auto isLRelu = std::find(ppeLayers.begin(), ppeLayers.end(), mv::PPELayerTypeEnum::PPELayerType_LPRELU) != ppeLayers.end();
            if (isLRelu)
                std::fill(reluMultData.begin(), reluMultData.end(), dpuTaskOp->get<mv::PPETask>("PPETask").getFixedFunction().getLReluMult());
        }

        for (size_t i = 0; i < weightsTableData->size(); i+=WT_ELEMENTS_PER_CHANNEL)
        {
            weightsTableData->at(i+2) = static_cast<int64_t>((mScaled[i/WT_ELEMENTS_PER_CHANNEL] << 16) | (round32[i/WT_ELEMENTS_PER_CHANNEL] << 14) | (mShift[i/WT_ELEMENTS_PER_CHANNEL]) << 8) | reluMultData[i/WT_ELEMENTS_PER_CHANNEL];
            if (hasBias)
                weightsTableData->at(i+3) = biasData[i/WT_ELEMENTS_PER_CHANNEL];
        }
    }
}
static void populateWeightsTablesQuantizationFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    for(auto dpuTaskOp = om.opBegin(); dpuTaskOp != om.opEnd(); ++dpuTaskOp)
    {
        if(dpuTaskOp->getOpType() == "DPUTask")
        {
            auto taskOp = dpuTaskOp->get<std::string>("taskOp");
            if(taskOp == "Conv" ||
               taskOp == "ChannelMajorConvolution" ||
               taskOp == "MaxPool" ||
               taskOp == "DepthwiseConv")
            {
                auto weightsTable = dpuTaskOp->getInputTensor(dpuTaskOp->get<std::size_t>("weightsTableIndex"));
                populateWeightsTablesActivationAndBias(weightsTable, dpuTaskOp, model, td);
            }
        }
    }
}
static void removeBiasTensorsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);
    std::set<std::string> biasNamesToRemove;
    for(auto dpuTaskOp = om.opBegin(); dpuTaskOp != om.opEnd(); ++dpuTaskOp)
    {
        if(dpuTaskOp->getOpType() == "DPUTask")
        {
            auto taskOp = dpuTaskOp->get<std::string>("taskOp");
            if(taskOp == "Conv" ||
               taskOp == "ChannelMajorConvolution" ||
               taskOp == "MaxPool" ||
               taskOp == "DepthwiseConv")
            {
                bool hasBias = dpuTaskOp->hasAttr("bias");
                if (hasBias)
                {
                    auto bias = dm.getTensor(dpuTaskOp->get<std::string>("bias"));
                    biasNamesToRemove.insert(bias->getName());
                    dpuTaskOp->erase("bias");
                }
            }
        }
    }
    for(auto biasName = biasNamesToRemove.begin(); biasName != biasNamesToRemove.end(); ++biasName)
    {
        auto bias = dm.getTensor(*biasName);
        dm.undefineTensor(bias);
    }
}

static void populateWeightsTablesPointersFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    for(auto dpuTaskOp = om.opBegin(); dpuTaskOp != om.opEnd(); ++dpuTaskOp)
    {
        if(dpuTaskOp->getOpType() == "DPUTask")
        {
            auto taskOp = dpuTaskOp->get<std::string>("taskOp");
            if(taskOp == "Conv" ||
               taskOp == "ChannelMajorConvolution" ||
               taskOp == "MaxPool" ||
               taskOp == "DepthwiseConv")
            {
                auto weightsTable = dpuTaskOp->getInputTensor(dpuTaskOp->get<std::size_t>("weightsTableIndex"));
                populateWeightsTablesDataPointers(weightsTable, dpuTaskOp, model);
                populateWeightsTablesSparsityPointers(weightsTable, dpuTaskOp, model, td);
            }
        }
    }
}

void populateActivationStorageElementMap(
    mv::Data::OpListIterator op,
    mv::ComputationModel& model)
{

    using clusterSolverFunc =
        std::function<mv::Tensor*(
        mv::Data::OpListIterator, size_t, size_t)>;

    using displacementCalcFunc =
        std::function<std::pair<long int, long int>(
        mv::Data::OpListIterator, size_t, clusterSolverFunc, size_t)>;

    const std::vector<clusterSolverFunc> clusterSolversFunctors = {
        [](mv::Data::OpListIterator op1, size_t tidx, size_t)
        {
            return &*op1->getInputTensor(tidx);
        },
        [](mv::Data::OpListIterator op1, size_t tidx, size_t clidx)
        {
            return &op1->getInputTensor(tidx)->getSubTensor(clidx);
        }
    };

    const std::unordered_map<std::string, displacementCalcFunc> displacementFunctors =
    {
        {
            "Conv",
            [](mv::Data::OpListIterator op1,
                size_t inputTensorIdx,
                clusterSolverFunc clSolver,
                size_t clIdx){
                std::vector<std::pair<long int, long int>> displacements;
                auto offset = 0;
                auto increment =
                    clSolver(op1, inputTensorIdx, clIdx)->getShape()[mv::IO_CHANNEL_DIMENSION] *
                    (clSolver(op1, inputTensorIdx, clIdx)->getDType().getSizeInBytes());
                return std::make_pair(offset, increment);
            }
        },
        {
            "Eltwise",
            [](mv::Data::OpListIterator op1,
                size_t inputTensorIdx,
                clusterSolverFunc clSolver,
                size_t clIdx){
                auto in0 = clSolver(op1, 0, clIdx);
                auto in1 = clSolver(op1, 1, clIdx);
                auto in0_addr = in0->hasAttr("address") ? in0->getAddress() : in0->get<std::size_t>("sliceAddress");
                auto in1_addr = in1->hasAttr("address") ? in1->getAddress() : in1->get<std::size_t>("sliceAddress");
                auto base_addr =std::min(
                    in0_addr,
                    in1_addr);
                auto in_tensor = inputTensorIdx == 0 ? in0 : in1;
                auto in_addr = inputTensorIdx == 0 ? in0_addr : in1_addr;
                auto offset = in_addr - base_addr;
                auto increment =
                    in_tensor->getShape()[mv::IO_CHANNEL_DIMENSION] *
                    (in_tensor->getDType().getSizeInBytes());
                return std::make_pair(offset, increment);
            }
        }
    };

    auto dispFunctor = displacementFunctors.find(op->get<std::string>("taskOp"));
    if (dispFunctor == displacementFunctors.cend())
        throw mv::RuntimeError(model, op->getName() +
            ": Op marked sparsity solving yet no displacement " +
            "solver registered for op type " +
            op->get<std::string>("taskOp"));


    auto inputTensorIdx = 0;
    for (auto tidx : op->get<std::vector<std::size_t>>("storageElementIndex"))
    {
        auto storageElementTable = op->getInputTensor(tidx);
        std::vector<int64_t> table_offsets(storageElementTable->getShape().totalSize(), 0);

        if (std::find(activationSegmentableStrategies.cbegin(), activationSegmentableStrategies.cend(),
            op->get<std::string>("splitStrategy")) == activationSegmentableStrategies.cend()) {
                auto disp = dispFunctor->second(op, inputTensorIdx, clusterSolversFunctors[0], 0);
                for (size_t i = 0; i < table_offsets.size(); ++i)
                    table_offsets[i] =
                        ((disp.first + i * disp.second) <<
                        SHIFT_FOR_STORAGE_ELEMENT);
        }
        else
        {
            auto numClusters =
                model.getGlobalConfigParams()->get<int>("Number_of_Clusters");
            auto running_index = 0;

            for (int cl = 0; cl < numClusters; cl++) {
                auto disp = dispFunctor->second(op, inputTensorIdx, clusterSolversFunctors[1], cl);
                auto clTotalSize =
                    storageElementTable->getSubTensor(cl).getShape().totalSize();
                for (size_t i = 0; i < clTotalSize; ++i)
                    table_offsets[running_index + i] =
                        ((disp.first + i * disp.second) <<
                        SHIFT_FOR_STORAGE_ELEMENT) + cl;
                running_index += clTotalSize;
            }
        }
        storageElementTable->populate(table_offsets, mv::Order("NHWC"));
        inputTensorIdx++;
    }
}

// Sub function to generate storage element pointer for dilated convolution

void populateActivationStorageElementMapForDilatedConvolution(mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel&)
{
    auto input = dpuTaskOp->getInputTensor(0);
    auto subConvIndex = dpuTaskOp->get<unsigned>("subConvIndex");
    auto activationStorageElement = dpuTaskOp->getInputTensor(dpuTaskOp->
                                            get<std::vector<std::size_t>>("storageElementIndex")[0]);
    auto dilationFactor = dpuTaskOp->get<unsigned>("originalDilationFactor");
    auto originalWidth = dpuTaskOp->get<mv::Shape>("originalShape")[mv::IO_WIDTH_DIMENSION];
    auto inputChannels = input->getShape()[mv::IO_CHANNEL_DIMENSION];
    auto width = activationStorageElement->getShape()[mv::IO_WIDTH_DIMENSION];
    auto height = activationStorageElement->getShape()[mv::IO_HEIGHT_DIMENSION];

    std::vector<int64_t> unpopulated_offsets(width*height, 0);
    unsigned subConvRowIdx = subConvIndex/dilationFactor;
    unsigned subConvColIdx = subConvIndex%dilationFactor;
    long int increment = inputChannels * (input->getDType().getSizeInBits() / 8) ;

    long int subConvElementIncrement = increment * dilationFactor;
    long int subConvRowIncrement = increment * originalWidth * dilationFactor;
    long int subConvOffset = increment * subConvColIdx + subConvRowIdx*originalWidth*increment;

    unsigned i = 0;
    unsigned rowOffset = subConvOffset;
    for(unsigned h = 0; h < height; ++h)
    {
        for(unsigned w = 0; w < width; ++w)
        {
            unpopulated_offsets[i++] = ((rowOffset + w * subConvElementIncrement ) << SHIFT_FOR_STORAGE_ELEMENT);
        }
        rowOffset += subConvRowIncrement;
    }
    activationStorageElement->populate(unpopulated_offsets, mv::Order("NHWC"));
}

int64_t getSmallestInputAddress(mv::Data::OpListIterator implicitJoin)
{
    auto numberInputs = implicitJoin.inputsSize();
    auto minBaseAddress = implicitJoin->getInputTensor(0)->getAddress();
    for (size_t i=1; i < numberInputs; i++)
    {
        auto address = implicitJoin->getInputTensor(i)->getAddress();
        if (address < minBaseAddress)
            minBaseAddress = address;
    }

    //std::cout << " minBaseAddress " << std::hex << minBaseAddress << std::endl;
    return minBaseAddress;
}

void populateActivationStorageElementMapForLayerAfterDilatedConvolution(mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
{
    mv::OpModel om(model);

    auto input = dpuTaskOp->getInputTensor()[0];
    auto parentImplicitOp = om.getSourceOp(input);
    while (parentImplicitOp->getOpType() != "ImplicitJoin")
    {
        parentImplicitOp = om.getSourceOp(parentImplicitOp->getInputTensor()[0]);
    }
    std::size_t numberSubConvs = 0;
    int64_t inputBaseAddress = 0;
    auto activationStorageElement = dpuTaskOp->getInputTensor(dpuTaskOp->get<std::vector<std::size_t>>("storageElementIndex")[0]);
    auto width = activationStorageElement->getShape()[mv::IO_WIDTH_DIMENSION];
    auto height = activationStorageElement->getShape()[mv::IO_HEIGHT_DIMENSION];
    std::vector<int64_t> unpopulated_offsets(width*height, 0);
    auto inputChannels = input->getShape()[mv::IO_CHANNEL_DIMENSION];
    long int increment = inputChannels * (input->getDType().getSizeInBits() / 8) ;

    //NOTE: The code referring to the previous operation if concat
    //is redundant as the final implementation was not to use an adittional operation
    //but resolve the unshuffling of the tensor through the 3D-DMAs, so I am leaving to comments


//    if (parentImplicitOp->getOpType() == "ImplicitJoin")
//    {
    numberSubConvs = parentImplicitOp.inputsSize();
    inputBaseAddress = getSmallestInputAddress(parentImplicitOp);
    //Original DF factor is sqrt() of inputs to ImplicitJoin
    unsigned int originalDilationFactor = std::sqrt(numberSubConvs);

    //for simplicity we pick base address as the smallest of all subconvs output addresses (to avoid negatives)
    unsigned i = 0;
    for(unsigned h = 0; h < height; ++h)
    {
        unsigned subConvRowIdx = (h%originalDilationFactor)*originalDilationFactor;
        for(unsigned w = 0; w < width; ++w)
        {
            //get base address based on subConvIdx
            unsigned subConvIdx = subConvRowIdx + w%originalDilationFactor;
            auto subConvBaseAddressOffset = parentImplicitOp->getInputTensor(subConvIdx)->getAddress() - inputBaseAddress;
            auto subConvWidth = parentImplicitOp->getInputTensor(subConvIdx)->getShape()[mv::IO_WIDTH_DIMENSION];
            //calc offset from start of subconv
            unsigned subConvElementIdx = (h/originalDilationFactor)*subConvWidth + (w/originalDilationFactor);
            unsigned subConvElementOffset = subConvElementIdx * increment;

            unpopulated_offsets[i++] = ((subConvBaseAddressOffset + subConvElementOffset) << SHIFT_FOR_STORAGE_ELEMENT);
            //std::cout << " row " << h << " col " << w << " address "  <<  std::hex << unpopulated_offsets[i-1] << " not shifted " << (subConvBaseAddressOffset + subConvElementOffset) << std::endl;
        }
    }
//    }
//    else if (parentImplicitOp->getOpType() == "DMATask" &&
//             om.getSourceOp(parentImplicitOp->getInputTensor()[0])->getOpType() == "ImplicitConcat" &&
//             om.getSourceOp(parentImplicitOp->getInputTensor()[0])->get<bool>("joinSimulation"))
//    {
//        numberSubConvs = om.getSourceOp(parentImplicitOp->getInputTensor()[0])->get<size_t>("dilationSubConvs");
//        unsigned int originalDilationFactor = std::sqrt(numberSubConvs);
//        unsigned i = 0;
//        unsigned subConvHeight = ceil((double)height / originalDilationFactor); //height of bigger subconvs
//        unsigned subConvWidth = ceil((double)width / originalDilationFactor); //width of bigger subconvs
//        for(unsigned h = 0; h < height; ++h)
//        {
//            for(unsigned w = 0; w < width; ++w)
//            {
//                unsigned totalNumberOfRows=0;
//                unsigned totalNumberOfCols=0;

//                //calc number of rows
//                if((height % originalDilationFactor) == 0 || (h % originalDilationFactor)  < (height % originalDilationFactor))  // all the sub conv to the left are of full width
//                {
//                    totalNumberOfRows = h%originalDilationFactor * subConvHeight;
//                }
//                else
//                {
//                    //add height of subconvRows of full height first and then add remaining of smaller height
//                    totalNumberOfRows = (height % originalDilationFactor) * subConvHeight + (h%originalDilationFactor - height%originalDilationFactor)*(subConvHeight - 1);
//                }
//                totalNumberOfRows += h / originalDilationFactor;
//                //calc number of cols
//                if((width % originalDilationFactor) == 0 || (w % originalDilationFactor)  < (width % originalDilationFactor))  // all the sub conv to the left are of full width
//                {
//                    totalNumberOfCols = w%originalDilationFactor * subConvWidth;
//                }
//                else
//                {
//                    //add width*subConvWidth for of full subConvWidth  + (subConvWidth-1) for the rows of smaller subconvs
//                    totalNumberOfCols = (width % originalDilationFactor) * subConvWidth + (w%originalDilationFactor - width%originalDilationFactor)*(subConvWidth - 1);
//                }
//                totalNumberOfCols += w / originalDilationFactor;
//                unsigned subConvElementIdx = (totalNumberOfCols + totalNumberOfRows*width);

//                unsigned subConvElementOffset = subConvElementIdx * increment;

//                unpopulated_offsets[i++] = (subConvElementOffset << SHIFT_FOR_STORAGE_ELEMENT);
//            }
//        }
//    }

    activationStorageElement->populate(unpopulated_offsets, mv::Order("NHWC"));
}


//NOTE: The whole idea of the pwl is that we are going to use a linear function that represents leaky Relu.
//This comes through the equation and idea of Alessandro https://colab.research.google.com/drive/1xTQyJtZiPtMw-r1jUGks-aspbrpuEdKR#scrollTo=biQruEJ7olzD.
//Idea: We use the equation: ((x << m) + b) >> s, and train its variables in order to find a close solution that always satisfies the
//leaky relu. After we generate the instruction list table and we save the values of the registers inside.
//The map of the bits per instruction are described here: https://docs.google.com/spreadsheets/d/1RcD1FYGiKCTCRTDsU-J4r_FaQyAQbzMLyRu7WkeNoOY/edit#gid=0.

void populateInstructionListMap(const std::string& pwlType,
                                mv::Data::TensorIterator instructionListTable,
                                const mv::QuantizationParams& outQuantParams)
{
    //NOTE : The instruction list has 5 bits of addresses so the biggest count of instructions is 11111 = 27
    //27 of course will be aligned to 32 and will contain NOPS inside
    auto instructionListShape = instructionListTable->getShape();
    std::vector<uint32_t> template_table(instructionListShape.totalSize(), 0);

    //NOTE: first 2 are hardware reserved areas
    std::size_t ADDR_OF_RESERVED = 6;
    std::size_t ADDR_OF_ADDR_FLEX = 11;
    std::size_t ADDR_OF_FIRST2_BITS = 9;
    std::size_t ADDR_OF_REST_BITS = 16;
    std::size_t ADDR_OF_VALUE = 19;
    std::size_t MASK_FIRST2_BITS = 3;
    std::size_t first2_bits, last3_bits;
    std::vector<int> range_vector;
    std::vector<int> shift_vector;
    std::vector<int> bias_vector;

    if (pwlType == "LeakyRelu") {
        range_vector = {-128, -109, -90, -72, -54, -36, -18, 0, 128};
        shift_vector = {1, -1, 0, 0, 0, -1, -1, -4};
        bias_vector = {-119, 44, -43, -31, -19, 18, 10, 0};
    } else if (pwlType == "Mish") {
        range_vector = {-128, -109, -90, -72, -54, -36, -18, 0, 128};
        shift_vector = {-12, -12, -12, -12, -12, -12, -12, 0};
        bias_vector = {1, 1, 1, 1, 1, 1, 1, 0};
        struct mish_params_t {
            std::vector<int> _range_vector;
            std::vector<int> _shift_vector;
            std::vector<int> _bias_vector;
        };
        const std::map<int32_t, mish_params_t> MISH_PARAMS = {
            {65000, {
                { -128, -104, -77, -52, -33, -9, 6, 49, 127},
                { 4, 5, 4, 4, 4, 0, -1, -1,},
                { 8, 2, 1, -6, -9, 1, -4, 0,},
            }},
            {66757, {
                { -128, -101, -74, -50, -31, -9, 6, 47, 127},
                { 1, 1, 1, 1, 0, -3, -4, -4,},
                { 64, 35, 5, -55, -65, 8, -32, 0,},
            }},
            {73281, {
                { -128, -90, -65, -47, -33, -9, 6, 36, 127},
                { 3, 2, 2, 1, 2, -2, -3, -3,},
                { 16, 15, 1, -8, -29, 4, -16, 0,},
            }},
            {74804, {
                { -128, -90, -65, -47, -33, -9, 6, 36, 127},
                { 3, 2, 2, 1, 2, -2, -3, -3,},
                { 16, 15, 1, -8, -29, 4, -16, 0,},
            }},
            {74805, {
                { -128, -90, -65, -47, -33, -9, 6, 36, 127},
                { 3, 2, 2, 1, 2, -2, -3, -3,},
                { 16, 15, 1, -8, -29, 4, -16, 0,},
            }},
            {76484, {
                { -128, -85, -61, -44, -30, -8, 6, 35, 127},
                { 3, 2, 2, 1, 2, -2, -3, -3,},
                { 16, 14, 0, -10, -29, 4, -16, 0,},
            }},
            {76718, {
                { -128, -85, -61, -44, -30, -8, 6, 35, 127},
                { 3, 2, 2, 1, 2, -2, -3, -3,},
                { 16, 14, 0, -10, -29, 4, -16, 0,},
            }},
            {81015, {
                { -128, -79, -57, -45, -30, -8, 6, 32, 127},
                { 3, 2, 1, 1, 2, -2, -3, -3,},
                { 16, 12, 13, -1, -29, 4, -16, 0,},
            }},
            {81484, {
                { -128, -79, -56, -44, -30, -8, 6, 31, 127},
                { 3, 2, 1, 1, 2, -2, -3, -3,},
                { 16, 12, 12, -2, -29, 4, -16, 0,},
            }},
            {81875, {
                { -128, -78, -56, -44, -30, -8, 6, 31, 127},
                { 3, 2, 1, 1, 1, -2, -3, -3,},
                { 16, 12, 12, -2, -25, 4, -16, 0,},
            }},
            {82656, {
                { -128, -77, -55, -43, -29, -8, 6, 31, 127},
                { 2, 1, 0, 0, 0, -3, -4, -4,},
                { 32, 23, 23, -5, -51, 8, -32, 0,},
            }},
            {82812, {
                { -128, -77, -55, -43, -29, -8, 6, 31, 127},
                { 2, 1, 0, 0, 0, -3, -4, -4,},
                { 32, 23, 23, -5, -51, 8, -32, 0,},
            }},
            {82813, {
                { -128, -77, -55, -43, -29, -8, 6, 31, 127},
                { 2, 1, 0, 0, 0, -3, -4, -4,},
                { 32, 23, 23, -5, -51, 8, -32, 0,},
            }},
            {86953, {
                { -128, -73, -51, -36, -19, -8, 6, 29, 127},
                { 3, 2, 1, 2, 0, -2, -3, -3,},
                { 16, 11, 10, -23, -17, 4, -16, 0,},
            }},
            {88437, {
                { -128, -71, -50, -39, -30, -5, 6, 29, 127},
                { 4, 3, 2, 2, 3, -1, -2, -2,},
                { 8, 5, 5, -2, -11, 2, -8, 0,},
            }},
            {89219, {
                { -128, -70, -49, -38, -29, -5, 6, 28, 127},
                { 4, 3, 2, 2, 3, -1, -2, -2,},
                { 8, 5, 5, -2, -11, 2, -8, 0,},
            }},
            {90000, {
                { -128, -69, -49, -38, -29, -5, 6, 27, 127},
                { 4, 3, 2, 2, 3, -1, -2, -2,},
                { 8, 5, 5, -2, -11, 2, -8, 0,},
            }},
            {91484, {
                { -128, -68, -48, -37, -28, -5, 6, 27, 127},
                { 4, 3, 2, 2, 3, -1, -2, -2,},
                { 8, 5, 4, -2, -11, 2, -8, 0,},
            }},
            {91875, {
                { -128, -68, -47, -36, -27, -5, 6, 27, 127},
                { 4, 3, 2, 2, 3, -1, -2, -2,},
                { 8, 5, 4, -3, -11, 2, -8, 0,},
            }},
            {92343, {
                { -128, -67, -47, -36, -27, -5, 6, 26, 127},
                { 4, 3, 2, 2, 3, -1, -2, -2,},
                { 8, 5, 4, -3, -11, 2, -8, 0,},
            }},
            {92344, {
                { -128, -67, -47, -36, -27, -5, 6, 26, 127},
                { 4, 3, 2, 2, 3, -1, -2, -2,},
                { 8, 5, 4, -3, -11, 2, -8, 0,},
            }},
            {93515, {
                { -128, -66, -46, -35, -26, -5, 6, 26, 127},
                { 4, 3, 2, 2, 3, -1, -2, -2,},
                { 8, 5, 4, -3, -11, 2, -8, 0,},
            }},
            {93516, {
                { -128, -66, -46, -35, -26, -5, 6, 26, 127},
                { 4, 3, 2, 2, 3, -1, -2, -2,},
                { 8, 5, 4, -3, -11, 2, -8, 0,},
            }},
            {94687, {
                { -128, -65, -45, -35, -26, -5, 6, 25, 127},
                { 4, 3, 2, 2, 3, -1, -2, -2,},
                { 8, 5, 4, -3, -11, 2, -8, 0,},
            }},
            {94688, {
                { -128, -65, -45, -35, -26, -5, 6, 25, 127},
                { 4, 3, 2, 2, 3, -1, -2, -2,},
                { 8, 5, 4, -3, -11, 2, -8, 0,},
            }},
            {95781, {
                { -128, -64, -45, -34, -25, -5, 6, 25, 127},
                { 4, 3, 2, 2, 3, -1, -2, -2,},
                { 8, 4, 4, -3, -11, 2, -8, 0,},
            }},
            {96171, {
                { -128, -64, -44, -34, -25, -5, 6, 25, 127},
                { 2, 1, 0, 0, 0, -3, -4, -4,},
                { 32, 16, 12, -14, -39, 8, -32, 0,},
            }},
            {96172, {
                { -128, -64, -44, -34, -25, -5, 6, 25, 127},
                { 2, 1, 0, 0, 0, -3, -4, -4,},
                { 32, 16, 12, -14, -39, 8, -32, 0,},
            }},
            {96640, {
                { -128, -63, -44, -33, -24, -5, 6, 25, 127},
                { 3, 1, 0, 0, 0, -3, -4, -4,},
                { 16, 16, 12, -15, -39, 8, -32, 0,},
            }},
            {96641, {
                { -128, -63, -44, -33, -24, -5, 6, 25, 127},
                { 3, 1, 0, 0, 0, -3, -4, -4,},
                { 16, 16, 12, -15, -39, 8, -32, 0,},
            }},
            {97031, {
                { -128, -63, -44, -33, -24, -5, 6, 25, 127},
                { 3, 1, 0, 0, 0, -3, -4, -4,},
                { 16, 16, 12, -15, -39, 8, -32, 0,},
            }},
            {98281, {
                { -128, -62, -43, -32, -23, -5, 6, 24, 127},
                { 3, 1, 0, 0, 0, -3, -4, -4,},
                { 16, 15, 11, -16, -39, 8, -32, 0,},
            }},
            {98437, {
                { -128, -62, -43, -32, -23, -5, 6, 24, 127},
                { 3, 1, 0, 0, 0, -3, -4, -4,},
                { 16, 15, 11, -16, -39, 8, -32, 0,},
            }},
            {98438, {
                { -128, -62, -43, -32, -23, -5, 6, 24, 127},
                { 3, 1, 0, 0, 0, -3, -4, -4,},
                { 16, 15, 11, -16, -39, 8, -32, 0,},
            }},
            {98984, {
                { -128, -62, -42, -32, -23, -5, 6, 24, 127},
                { 3, 1, 0, 0, 0, -3, -4, -4,},
                { 16, 15, 10, -16, -39, 8, -32, 0,},
            }},
            {99140, {
                { -128, -61, -42, -32, -23, -5, 6, 24, 127},
                { 3, 1, 0, 0, 0, -3, -4, -4,},
                { 16, 15, 10, -16, -39, 8, -32, 0,},
            }},
            {99141, {
                { -128, -61, -42, -32, -23, -5, 6, 24, 127},
                { 3, 1, 0, 0, 0, -3, -4, -4,},
                { 16, 15, 10, -16, -39, 8, -32, 0,},
            }},
            {99843, {
                { -128, -61, -42, -32, -22, -5, 6, 24, 127},
                { 3, 1, 0, 0, 0, -3, -4, -4,},
                { 16, 15, 10, -16, -39, 8, -32, 0,},
            }},
            {99844, {
                { -128, -61, -42, -32, -22, -5, 6, 24, 127},
                { 3, 1, 0, 0, 0, -3, -4, -4,},
                { 16, 15, 10, -16, -39, 8, -32, 0,},
            }},
            {101641, {
                { -128, -60, -41, -31, -21, -5, 6, 23, 127},
                { 3, 1, 0, 0, 0, -3, -4, -4,},
                { 16, 14, 9, -17, -39, 8, -32, 0,},
            }},
            {101875, {
                { -128, -59, -41, -30, -21, -5, 6, 23, 127},
                { 3, 1, 0, 0, 0, -3, -4, -4,},
                { 16, 14, 9, -18, -39, 8, -32, 0,},
            }},
            {102578, {
                { -128, -59, -40, -30, -21, -5, 6, 23, 127},
                { 3, 1, 0, 0, 0, -3, -4, -4,},
                { 16, 14, 8, -18, -39, 8, -32, 0,},
            }},
            {103280, {
                { -128, -58, -40, -30, -20, -5, 6, 22, 127},
                { 3, 1, 0, 0, 0, -3, -4, -4,},
                { 16, 13, 8, -18, -39, 8, -32, 0,},
            }},
            {107422, {
                { -128, -55, -38, -28, -18, -5, 6, 20, 127},
                { 5, 3, 2, 2, 1, -1, -2, -2,},
                { 4, 3, 2, -5, -7, 2, -8, 0,},
            }},
            {109453, {
                { -128, -54, -37, -27, -17, -4, 6, 20, 127},
                { 4, 2, 1, 1, 0, -2, -3, -3,},
                { 8, 6, 3, -10, -13, 4, -8, 0,},
            }},
            {112266, {
                { -128, -52, -35, -26, -2, 6, 12, 25, 127},
                { 2, 0, -1, 0, -4, -5, -5, -5,},
                { 32, 20, 6, -59, 16, -64, -32, 0,},
            }},
            {112500, {
                { -128, -52, -35, -25, -2, 6, 12, 25, 127},
                { 2, 0, -1, 0, -4, -5, -5, -5,},
                { 32, 20, 6, -59, 16, -64, -32, 0,},
            }},
            {113047, {
                { -128, -52, -35, -25, -2, 6, 12, 25, 127},
                { 2, 0, -1, 0, -4, -5, -5, -5,},
                { 32, 20, 6, -59, 16, -64, -32, 0,},
            }},
            {114375, {
                { -128, -51, -34, -25, -2, 6, 12, 25, 127},
                { 2, 0, -1, 0, -4, -5, -5, -5,},
                { 32, 19, 4, -59, 16, -64, -32, 0,},
            }},
            {116641, {
                { -128, -50, -33, -24, -2, 5, 10, 24, 127},
                { 2, 0, -1, 0, -4, -4, -5, -5,},
                { 32, 18, 2, -59, 16, 80, -32, 0,},
            }},
            {122655, {
                { -128, -47, -31, -21, -5, 0, 3, 23, 127},
                { 6, 4, 3, 4, 1, 0, -1, -1,},
                { 2, 1, 0, -4, -1, 1, -2, 0,},
            }},
            {124375, {
                { -128, -46, -30, -21, -5, 0, 3, 22, 127},
                { 4, 1, 1, 1, -1, -2, -3, -3,},
                { 8, 15, -1, -13, -4, 4, -8, 0,},
            }},
            {124766, {
                { -128, -46, -30, -21, -5, 0, 3, 22, 127},
                { 4, 1, 1, 1, -1, -2, -3, -3,},
                { 8, 15, -1, -13, -4, 4, -8, 0,},
            }},
            {131719, {
                { -128, -43, -28, -18, -5, 0, 3, 20, 127},
                { 4, 1, 1, 1, -1, -2, -3, -3,},
                { 8, 14, -2, -13, -4, 4, -8, 0,},
            }},
            {131797, {
                { -128, -43, -28, -18, -5, 0, 3, 20, 127},
                { 4, 1, 1, 1, -1, -2, -3, -3,},
                { 8, 14, -2, -13, -4, 4, -8, 0,},
            }},
            {132266, {
                { -128, -42, -27, -18, -5, 0, 3, 20, 127},
                { 4, 1, 1, 1, -1, -2, -3, -3,},
                { 8, 13, -2, -13, -4, 4, -8, 0,},
            }},
            {133281, {
                { -128, -42, -27, -18, -5, 0, 3, 20, 127},
                { 4, 1, 1, 1, -1, -2, -3, -3,},
                { 8, 13, -2, -13, -2, 4, -8, 0,},
            }},
            {150234, {
                { -128, -36, -22, -13, -5, 0, 3, 17, 127},
                { 4, 1, 1, 0, -1, -2, -3, -3,},
                { 8, 10, -5, -9, -2, 4, -8, 0,},
            }},
            {153047, {
                { -128, -35, -22, -12, -5, 0, 3, 17, 127},
                { 4, 1, 1, 0, -1, -2, -3, -3,},
                { 8, 10, -5, -9, -2, 4, -8, 0,},
            }},
            {153828, {
                { -128, -35, -21, -11, -5, 0, 3, 17, 127},
                { 5, 2, 2, 0, 0, -1, -2, -2,},
                { 4, 5, -2, 0, -1, 2, -4, 0,},
            }},
            {160938, {
                { -128, -33, -20, -11, -5, 0, 3, 16, 127},
                { 5, 2, 2, 1, 0, -1, -2, -2,},
                { 4, 5, -3, -2, -1, 2, -4, 0,},
            }},
            {161719, {
                { -128, -32, -20, -11, -5, 0, 3, 15, 127},
                { 5, 2, 2, 1, 0, -1, -2, -2,},
                { 4, 4, -3, -2, -1, 2, -4, 0,},
            }},
            {161875, {
                { -128, -32, -20, -11, -5, 0, 3, 15, 127},
                { 5, 2, 2, 1, 0, -1, -2, -2,},
                { 4, 4, -3, -2, -1, 2, -4, 0,},
            }},
            {164375, {
                { -128, -32, -19, -11, -5, 0, 3, 15, 127},
                { 5, 2, 2, 1, 0, -1, -2, -2,},
                { 4, 4, -3, -2, -1, 2, -4, 0,},
            }},
            {169531, {
                { -128, -30, -18, -11, -5, 0, 3, 14, 127},
                { 5, 2, 1, 1, 0, -1, -2, -2,},
                { 4, 4, 1, -2, -1, 2, -4, 0,},
            }},
            {178438, {
                { -128, -28, -17, -11, -5, 0, 3, 13, 127},
                { 5, 2, 1, 1, 0, -1, -2, -2,},
                { 4, 3, 1, -2, -1, 2, -4, 0,},
            }},
            {189061, {
                { -128, -26, -15, -11, -5, 0, 3, 12, 127},
                { 5, 2, 0, 1, 0, -1, -2, -2,},
                { 4, 3, 7, -2, -1, 2, -4, 0,},
            }},
            {192188, {
                { -128, -26, -14, -11, -5, 0, 3, 12, 127},
                { 5, 2, 0, 1, 0, -1, -2, -2,},
                { 4, 3, 6, -2, -1, 2, -4, 0,},
            }},
            {192656, {
                { -128, -26, -14, -11, -5, 0, 3, 12, 127},
                { 5, 2, 0, 1, 0, -1, -2, -2,},
                { 4, 3, 6, -2, -1, 2, -4, 0,},
            }},
            {193594, {
                { -128, -25, -14, -11, -5, 0, 3, 12, 127},
                { 5, 2, 0, 1, 0, -1, -2, -2,},
                { 4, 3, 6, -2, -1, 2, -4, 0,},
            }},
            {198281, {
                { -128, -25, -13, -11, -5, 0, 3, 12, 127},
                { 5, 2, -1, 1, 0, -1, -2, -2,},
                { 4, 3, 18, -2, -1, 2, -4, 0,},
            }},
            {198282, {
                { -128, -25, -13, -11, -5, 0, 3, 12, 127},
                { 5, 2, -1, 1, 0, -1, -2, -2,},
                { 4, 3, 18, -2, -1, 2, -4, 0,},
            }},
            {215156, {
                {-128, -22, -11, -2, 0, 2, 10, 124, 127},
                {4, 1, 0, -2, -3, -3, -3, -3},
                {8, 3, -5, 0, 0, -8, 0, 0},
            }},
            {237500, {
                {-128, -19, -9, -2, 0, 2, 9, 124, 127},
                {6, 3, 2, 0, -1, -1, -1, -1},
                {2, 1, -1, 0, 0, -2, 0, 0},
            }},
            {238438, {
                {-128, -19, -9, -2, 0, 2, 9, 124, 127},
                {6, 3, 2, 0, -1, -1, -1, -1},
                {2, 1, -1, 0, 0, -2, 0, 0},
            }},
            {254844, {
                {-128, -17, -7, -2, 0, 2, 8, 124, 127},
                {7, 5, 2, 1, 0, 0, 0, 0},
                {1, 0, 0, 0, 0, -1, 0, 0},
            }},
            {269531, {
                {-128, -16, -7, -2, 0, 2, 7, 124, 127},
                {7, 4, 3, 1, 0, 0, 0, 0},
                {1, 0, 0, 0, 0, -1, 0, 0},
            }},
            {307500, {
                {-128, -13, -7, -2, 0, 2, 6, 124, 127},
                {7, 4, 3, 1, 0, 0, 0, 0},
                {1, 0, 0, 0, 0, -1, 0, 0},
            }},
            {355312, {
                {-128, -11, -7, 0, 3, 4, 6, 124, 127},
                {7, 4, 3, 0, -5, 0, 0, 0},
                {1, 0, 0, 0, -94, 0, 0, 0},
            }},
            {355313, {
                {-128, -11, -7, 0, 3, 4, 6, 124, 127},
                {7, 4, 3, 0, -5, 0, 0, 0},
                {1, 0, 0, 0, -94, 0, 0, 0},
            }},
            {388125, {
                {-128, -9, -7, 0, 3, 4, 6, 124, 127},
                {7, 4, 3, 0, -5, 0, 0, 0},
                {1, 0, 0, 0, -93, 0, 0, 0},
            }},

            {112253, {
                { -128, -52, -35, -26, -2, 6, 12, 25, 127},
                { 2, 0, -1, 0, -4, -5, -5, -5,},
                { 32, 20, 6, -59, 16, -64, -32, 0,},
            }},
            {113965, {
                { -128, -51, -34, -25, -2, 6, 12, 25, 127},
                { 2, 0, -1, 0, -4, -5, -5, -5,},
                { 32, 19, 4, -59, 16, -64, -32, 0,},
            }},
            {91865, {
                { -128, -68, -47, -36, -27, -5, 6, 27, 127},
                { 4, 3, 2, 2, 3, -1, -2, -2,},
                { 8, 5, 4, -3, -11, 2, -8, 0,},
            }},
            {88441, {
                { -128, -71, -50, -39, -30, -5, 6, 29, 127},
                { 4, 3, 2, 2, 3, -1, -2, -2,},
                { 8, 5, 5, -2, -11, 2, -8, 0,},
            }},
            {133264, {
                { -128, -42, -27, -18, -5, 0, 3, 20, 127},
                { 4, 1, 1, 1, -1, -2, -3, -3,},
                { 8, 13, -2, -13, -2, 4, -8, 0,},
            }},
            {261082, {
                {-128, -17, -7, -2, 0, 2, 8, 124, 127},
                {7, 5, 2, 1, 0, 0, 0, 0},
                {1, 0, 0, 0, 0, -1, 0, 0},
            }},
            {228631, {
                {-128, -19, -9, -2, 0, 2, 9, 124, 127},
                {6, 3, 2, 0, -1, -1, -1, -1},
                {2, 1, -1, 0, 0, -2, 0, 0},
            }},
            {216725, {
                {-128, -22, -11, -2, 0, 2, 10, 124, 127},
                {4, 1, 0, -2, -3, -3, -3, -3},
                {8, 3, -5, 0, 0, -8, 0, 0},
            }},
            {101787, {
                { -128, -60, -41, -31, -21, -5, 6, 23, 127},
                { 3, 1, 0, 0, 0, -3, -4, -4,},
                { 16, 14, 9, -17, -39, 8, -32, 0,},
            }},
            {97935, {
                { -128, -63, -44, -33, -24, -5, 6, 25, 127},
                { 3, 1, 0, 0, 0, -3, -4, -4,},
                { 16, 16, 12, -15, -39, 8, -32, 0,},
            }},
            {94666, {
                { -128, -65, -45, -35, -26, -5, 6, 25, 127},
                { 4, 3, 2, 2, 3, -1, -2, -2,},
                { 8, 5, 4, -3, -11, 2, -8, 0,},
            }},
            {161863, {
                { -128, -32, -20, -11, -5, 0, 3, 15, 127},
                { 5, 2, 2, 1, 0, -1, -2, -2,},
                { 4, 4, -3, -2, -1, 2, -4, 0,},
            }},
            {285828, {
                {-128, -16, -7, -2, 0, 2, 7, 124, 127},
                {7, 4, 3, 1, 0, 0, 0, 0},
                {1, 0, 0, 0, 0, -1, 0, 0},
            }},
            {206375, {
                { -128, -25, -13, -11, -5, 0, 3, 12, 127},
                { 5, 2, -1, 1, 0, -1, -2, -2,},
                { 4, 3, 18, -2, -1, 2, -4, 0,},
            }},
            {205441, {
                { -128, -25, -13, -11, -5, 0, 3, 12, 127},
                { 5, 2, -1, 1, 0, -1, -2, -2,},
                { 4, 3, 18, -2, -1, 2, -4, 0,},
            }},
            {170423, {
                { -128, -30, -18, -11, -5, 0, 3, 14, 127},
                { 5, 2, 1, 1, 0, -1, -2, -2,},
                { 4, 4, 1, -2, -1, 2, -4, 0,},
            }},
            {137817, {
                { -128, -42, -27, -18, -5, 0, 3, 20, 127},
                { 4, 1, 1, 1, -1, -2, -3, -3,},
                { 8, 13, -2, -13, -2, 4, -8, 0,},
            }},
            {107429, {
                { -128, -55, -38, -28, -18, -5, 6, 20, 127},
                { 5, 3, 2, 2, 1, -1, -2, -2,},
                { 4, 3, 2, -5, -7, 2, -8, 0,},
            }},
            {130938, {
                { -128, -43, -28, -18, -5, 0, 3, 20, 127},
                { 4, 1, 1, 1, -1, -2, -3, -3,},
                { 8, 14, -2, -13, -4, 4, -8, 0,},
            }},
            {269564, {
                {-128, -16, -7, -2, 0, 2, 7, 124, 127},
                {7, 4, 3, 1, 0, 0, 0, 0},
                {1, 0, 0, 0, 0, -1, 0, 0},
            }},
            {102020, {
                { -128, -59, -41, -30, -21, -5, 6, 23, 127},
                { 3, 1, 0, 0, 0, -3, -4, -4,},
                { 16, 14, 9, -18, -39, 8, -32, 0,},
            }},
            {254856, {
                {-128, -17, -7, -2, 0, 2, 8, 124, 127},
                {7, 5, 2, 1, 0, 0, 0, 0},
                {1, 0, 0, 0, 0, -1, 0, 0},
            }},
            {232989, {
                {-128, -19, -9, -2, 0, 2, 9, 124, 127},
                {6, 3, 2, 0, -1, -1, -1, -1},
                {2, 1, -1, 0, 0, -2, 0, 0},
            }},
            {103265, {
                { -128, -59, -41, -30, -21, -5, 6, 23, 127},
                { 3, 1, 0, 0, 0, -3, -4, -4,},
                { 16, 14, 9, -18, -39, 8, -32, 0,},
            }},
            {192601, {
                { -128, -26, -14, -11, -5, 0, 3, 12, 127},
                { 5, 2, 0, 1, 0, -1, -2, -2,},
                { 4, 3, 6, -2, -1, 2, -4, 0,},
            }},
            {207231, {
                { -128, -25, -13, -11, -5, 0, 3, 12, 127},
                { 5, 2, -1, 1, 0, -1, -2, -2,},
                { 4, 3, 18, -2, -1, 2, -4, 0,},
            }},
            {123109, {
                { -128, -47, -31, -21, -5, 0, 3, 23, 127},
                { 6, 4, 3, 4, 1, 0, -1, -1,},
                { 2, 1, 0, -4, -1, 1, -2, 0,},
            }},
        };
        int32_t max_quant = std::round(outQuantParams.getMax().at(0) * 10000.f);
        if (MISH_PARAMS.count(max_quant) > 0) {
            const auto params = MISH_PARAMS.at(max_quant);
            range_vector = params._range_vector;
            shift_vector = params._shift_vector;
            bias_vector = params._bias_vector;
        } else {
            throw std::runtime_error("weights_tables: Couldn't find max_quant: " + std::to_string(max_quant));
        }
    }

    std::size_t k = 0;
    for (std::size_t j = 0; j < 32; j++)
    {
        first2_bits = j & MASK_FIRST2_BITS;
        last3_bits = j >> 2;

        if (j == 15)
            template_table[j] = (ALU_HALT_OPCODE);
        else if (j > 25)
            template_table[j] = (ALU_HALT_OPCODE);
        else
        {
            if (j < range_vector.size())
            {
                template_table[j] = ((range_vector[j] << ADDR_OF_VALUE) | (last3_bits << ADDR_OF_REST_BITS)
                    | (8 << ADDR_OF_ADDR_FLEX)
                    | (first2_bits << ADDR_OF_FIRST2_BITS) | (0 << ADDR_OF_RESERVED) | ALU_LOAD);
            }
            else if (j < range_vector.size() + shift_vector.size() + 1)
            {
                if (j < 16)
                    template_table[j] = ((shift_vector[j - range_vector.size()] << ADDR_OF_VALUE)
                        | (last3_bits << ADDR_OF_REST_BITS) | (8 << ADDR_OF_ADDR_FLEX)
                        | (first2_bits << ADDR_OF_FIRST2_BITS) | (0 << ADDR_OF_RESERVED)  | ALU_LOAD);
                 else
                {
                    k = j - 1;
                    first2_bits = k & MASK_FIRST2_BITS;
                    last3_bits = k >> 2;
                    template_table[j] = ((shift_vector[k - range_vector.size()] << ADDR_OF_VALUE)
                        | (last3_bits << ADDR_OF_REST_BITS) | (8 << ADDR_OF_ADDR_FLEX)
                        | (first2_bits << ADDR_OF_FIRST2_BITS) | (0 << ADDR_OF_RESERVED)  | ALU_LOAD);
                }
            }
            else if (j < range_vector.size() + shift_vector.size() + bias_vector.size() + 1)
            {
                k = j - 1;
                first2_bits = k & MASK_FIRST2_BITS;
                last3_bits = k >> 2;
                template_table[j] = ((bias_vector[k - range_vector.size() - shift_vector.size()] << ADDR_OF_VALUE)
                        | (last3_bits << ADDR_OF_REST_BITS) | (8 << ADDR_OF_ADDR_FLEX)
                        | (first2_bits << ADDR_OF_FIRST2_BITS) | (0 << ADDR_OF_RESERVED)  | ALU_LOAD);
            }
        }
    }

    std::vector<int64_t> template_table_appropriate_type(template_table.begin(), template_table.end());

    instructionListTable->populate(template_table_appropriate_type, mv::Order("NHWC"));
}

static void populateStorageElementPointersFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    for(auto op : om.getOps("DPUTask"))
    {
        auto taskOp = op->getOpType();
        if (taskOp == "DPUTask")
        {
            if(op->hasAttr("activationSparsityCompilerSolving") &&
                op->get<bool>("activationSparsityCompilerSolving"))
                populateActivationStorageElementMap(op, model);

            // New logic for generating SEP for dilated convolution
            if(op->hasAttr("activationSparsityCompilerSolvingForDilatedConv")
                    && op->get<bool>("activationSparsityCompilerSolvingForDilatedConv"))
            {
                populateActivationStorageElementMapForDilatedConvolution(op, model);
            }
        }
    }
}

static void populateInstructionListTablesFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    const std::string dpuPWLTag = "WithDPUPWL";

    for(auto dpuTaskOp = om.opBegin(); dpuTaskOp != om.opEnd(); ++dpuTaskOp)
    {
        auto taskOp = dpuTaskOp->getOpType();
        if (taskOp == "DPUTask")
        {
            if (dpuTaskOp->hasAttr(dpuPWLTag) && dpuTaskOp->get<bool>(dpuPWLTag))
            {
                auto instructionListTable
                        = dpuTaskOp->getInputTensor(dpuTaskOp->get<std::size_t>("instructionListTableIndex"));

                auto attrs = dpuTaskOp->getAttrs({dpuPWLTag});
                for (auto && attr : attrs) {
                    if (attr.first.find("With") == 0) {
                        populateInstructionListMap(attr.first.substr(4),
                                                   instructionListTable,
                                                   dpuTaskOp->getOutputTensor(0)->getQuantParams());
                    }
                }
            }
        }
    }
}

static void generateWeightsTablesFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    for(auto dpuTaskOp = om.opBegin(); dpuTaskOp != om.opEnd(); ++dpuTaskOp)
    {
        if(dpuTaskOp->getOpType() == "DPUTask")
        {
            auto taskOp = dpuTaskOp->get<std::string>("taskOp");
            if(taskOp == "Conv" ||
               taskOp == "ChannelMajorConvolution" ||
               taskOp == "MaxPool" ||
               taskOp == "DepthwiseConv")
            {
                std::string opName = dpuTaskOp->getName();
                std::string kernelWeightsTableName(mv::createWeightTableName(opName));

                auto outputChannels = dpuTaskOp->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION];
                outputChannels = mv::round_up(dpuTaskOp->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION], 16);

                // per channel layout:
                // 3 -> bias
                // 2 -> mult << 16 | round << 14 |  shift << 8 | prelu
                // 1 -> SP_PTR
                // 0 -> DATA_PTR
                mv::Shape shape({WT_ELEMENTS_PER_CHANNEL, 1, 1, outputChannels});
                std::vector<int64_t> weightsTableData(shape.totalSize(), 0);
                auto weightTable = om.constantInt(kernelWeightsTableName, weightsTableData, shape, mv::DType("Int32"), mv::Order("NHWC"));
                weightTable->set<bool>("weightTable", true);
                om.getSourceOp(weightTable)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
                unsigned newSize = dpuTaskOp->addInputTensor(weightTable);
                om.defineFlow(weightTable, dpuTaskOp, newSize - 1);
                dpuTaskOp->set<size_t>("weightsTableIndex", newSize - 1);
            }
        }
    }
}

static void generateInstructionListTablesFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    for(auto dpuTaskOp = om.opBegin(); dpuTaskOp != om.opEnd(); ++dpuTaskOp)
    {
        if(dpuTaskOp->getOpType() == "DPUTask")
        {
            auto taskOpType = dpuTaskOp->get<std::string>("taskOp");
            if (dpuTaskOp->hasAttr("postOpTypes") && dpuTaskOp->hasAttr("WithDPUPWL") && dpuTaskOp->get<bool>("WithDPUPWL"))
            {
                auto postOps = dpuTaskOp->get<std::vector<std::string>>("postOpTypes");
                //"FLEXARB"
                auto ppeIterator = std::find_if(postOps.begin(), postOps.end(), mv::ControlModel::isDpuPwl);
                if ( ppeIterator != dpuTaskOp->get<std::vector<std::string>>("postOpTypes").end())
                {
                    std::string opName = dpuTaskOp->getName();
                    std::string instructionListTableName(mv::createInstructionListTableName(opName));
                    std::size_t numberOfInstructions = 25;
                    std::size_t alignedInstructions = mv::round_up(numberOfInstructions, 16);
                    mv::Shape shape({alignedInstructions, 1, 1, 1});
                    std::vector<int64_t> instructionListTableData(shape.totalSize(), 0);
                    auto instructionListTable = om.constantInt(instructionListTableName, instructionListTableData, shape, mv::DType("Int32"), mv::Order("NHWC"));
                    instructionListTable->set<bool>("instructionListTable", true);
                    om.getSourceOp(instructionListTable)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
                    unsigned newSize = dpuTaskOp->addInputTensor(instructionListTable);
                    om.defineFlow(instructionListTable, dpuTaskOp, newSize - 1);
                    dpuTaskOp->set<size_t>("instructionListTableIndex", newSize - 1);
                }
            }
        }
    }
}
