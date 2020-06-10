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



void populateWeightsTablesSparsityPointers(mv::Data::TensorIterator weightsTableData, mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
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
            long int offset = 0xFFFFFF; // NOTE: Implementation defined
            for (size_t i = 0; i < weightsTableData->size(); i+=WT_ELEMENTS_PER_CHANNEL)
                  weightsTableData->at(i+1) = offset;
        }
    }
    else if(taskOp == "DepthwiseConv"  ||
            taskOp == "ChannelMajorConvolution" ||
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


void populateWeightsTablesActivationAndBias(mv::Data::TensorIterator weightsTableData, mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
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
        quantParams = dpuTaskOp->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams");
        if (!quantParams.isEmpty())
        {
            auto mult = quantParams.getMult();
            auto shift = quantParams.getShift();
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

    unsigned round_mode = 1;
    std::vector<int32_t> round32(outputChannels, round_mode);
    std::vector<int32_t> reluMultData(outputChannels, 0);
    if (hasPPETask)
    {
        auto ppeFF = dpuTaskOp->get<mv::PPETask>("PPETask").getFixedFunction();
        auto& ppeLayers = ppeFF.getLayers();
        auto isLRelu = std::find(ppeLayers.begin(), ppeLayers.end(), mv::PPELayerTypeEnum::PPELayerType_LPRELU) != ppeLayers.end();
        if (isLRelu)
            std::fill(reluMultData.begin(), reluMultData.end(), dpuTaskOp->get<mv::PPETask>("PPETask").getFixedFunction().getLReluMult());
    }

    for (size_t i = 0; i < weightsTableData->size(); i+=WT_ELEMENTS_PER_CHANNEL)
    {
        weightsTableData->at(i+2) = static_cast<long int>((mScaled[i/WT_ELEMENTS_PER_CHANNEL] << 16) | (round32[i/WT_ELEMENTS_PER_CHANNEL] << 14) | (mShift[i/WT_ELEMENTS_PER_CHANNEL]) << 8) | reluMultData[i/WT_ELEMENTS_PER_CHANNEL];
        if (hasBias)
            weightsTableData->at(i+3) = biasData[i/WT_ELEMENTS_PER_CHANNEL];
    }
}
static void populateWeightsTablesQuantizationFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
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
                populateWeightsTablesActivationAndBias(weightsTable, dpuTaskOp, model);
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

static void populateWeightsTablesPointersFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
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
                populateWeightsTablesSparsityPointers(weightsTable, dpuTaskOp, model);
            }
        }
    }
}

void populateActivationStorageElementMap(mv::Data::TensorIterator activationStorageElement, mv::Data::OpListIterator dpuTaskOp)
{
    auto input = dpuTaskOp->getInputTensor(0);
    auto inputChannels = input->getShape()[mv::IO_CHANNEL_DIMENSION];
    auto height_width = activationStorageElement->getShape().totalSize();

    std::vector<int64_t> unpopulated_offsets(height_width, 0);

    long int increment = inputChannels * (input->getDType().getSizeInBits() / 8);
    for(unsigned i = 0; i < height_width; ++i)
        unpopulated_offsets[i] = (i * increment << SHIFT_FOR_STORAGE_ELEMENT);
    activationStorageElement->populate(unpopulated_offsets, mv::Order("NHWC"));
}

//NOTE: The whole idea of the pwl is that we are going to use a linear function that represents leaky Relu.
//This comes through the equation and idea of Alessandro https://colab.research.google.com/drive/1xTQyJtZiPtMw-r1jUGks-aspbrpuEdKR#scrollTo=biQruEJ7olzD.
//Idea: We use the equation: ((x << m) + b) >> s, and train its variables in order to find a close solution that always satisfies the
//leaky relu. After we generate the instruction list table and we save the values of the registers inside.
//The map of the bits per instruction are described here: https://docs.google.com/spreadsheets/d/1RcD1FYGiKCTCRTDsU-J4r_FaQyAQbzMLyRu7WkeNoOY/edit#gid=0.
void populateInstructionListMap(mv::Data::TensorIterator instructionListTable)
{
    //NOTE : The instruction list has 5 bits of addresses so the biggest count of instructions is 11111 = 27
    //27 of course will be aligned to 32 and will containt NOPS inside
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
    std::vector<int> range_vector = {-128, -109, -90, -72, -54, -36, -18, 0, 128};
    std::vector<int> shift_vector = {1, -1, 0, 0, 0, -1, -1, -4};
    std::vector<int> bias_vector = {-119, 44, -43, -31, -19, 18, 10, 0};
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
    for(auto dpuTaskOp = om.opBegin(); dpuTaskOp != om.opEnd(); ++dpuTaskOp)
    {
        auto taskOp = dpuTaskOp->getOpType();
        if (taskOp == "DPUTask")
        {
            if(dpuTaskOp->hasAttr("activationSparsityCompilerSolving")
                    && dpuTaskOp->get<bool>("activationSparsityCompilerSolving"))
            {
                auto activationStorageElement
                        = dpuTaskOp->getInputTensor(dpuTaskOp->get<std::size_t>("storageElementIndex"));
                auto activationSparsityMap
                        = dpuTaskOp->getInputTensor(dpuTaskOp->get<std::size_t>("unpopulatedSparsityMapIndex"));
                dpuTaskOp->getInputTensor(0)->set<bool>("activationSparsityCompilerSolving", true);
                dpuTaskOp->getInputTensor(0)->set<std::size_t>("storageElementAddress", activationStorageElement->getAddress());
                dpuTaskOp->getInputTensor(0)->set<std::size_t>("unpopulatedSparsityMapIndex", activationSparsityMap->getAddress());
                populateActivationStorageElementMap(activationStorageElement, dpuTaskOp);
            }
        }
    }
}

static void populateInstructionListTablesFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    for(auto dpuTaskOp = om.opBegin(); dpuTaskOp != om.opEnd(); ++dpuTaskOp)
    {
        auto taskOp = dpuTaskOp->getOpType();
        if (taskOp == "DPUTask")
        {
            if(dpuTaskOp->hasAttr("firstConvWithLRelu")
                    && dpuTaskOp->get<bool>("firstConvWithLRelu"))
            {
                auto instructionListTable
                        = dpuTaskOp->getInputTensor(dpuTaskOp->get<std::size_t>("instructionListTableIndex"));
                populateInstructionListMap(instructionListTable);
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
                mv::QuantizationParams quantParams = {{},{},{},{}};
                auto weightTable = om.constantInt(weightsTableData, shape, mv::DType("Int32"), mv::Order("NHWC"), quantParams, kernelWeightsTableName);
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
            if(taskOpType == "Conv" && dpuTaskOp->hasAttr("postOpTypes") && dpuTaskOp->hasAttr("firstConvWithLRelu")
                    && dpuTaskOp->get<bool>("firstConvWithLRelu"))
            {
                auto ppeIterator = std::find(dpuTaskOp->get<std::vector<std::string>>("postOpTypes").begin(),
                                             dpuTaskOp->get<std::vector<std::string>>("postOpTypes").end(),
                                             "FLEXARB");
                if ( ppeIterator != dpuTaskOp->get<std::vector<std::string>>("postOpTypes").end())
                {
                    std::string opName = dpuTaskOp->getName();
                    std::string instructionListTableName(mv::createInstructionListTableName(opName));
                    std::size_t numberOfInstructions = 25;
                    std::size_t alignedInstructions = mv::round_up(numberOfInstructions, 16);
                    mv::Shape shape({alignedInstructions, 1, 1, 1});
                    std::vector<int64_t> instructionListTableData(shape.totalSize(), 0);
                    mv::QuantizationParams quantParams = {{},{},{},{}};
                    auto instructionListTable = om.constantInt(instructionListTableData, shape, mv::DType("Int32"), mv::Order("NHWC"), quantParams, instructionListTableName);
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
