#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/target/kmb/ppe_task.hpp"
#include <math.h>

static const std::size_t WT_ELEMENTS_PER_CHANNEL = 4;
//cause of the BASE_PTR is 9 bits, -4 for the 16 alignment according to zoran
static const std::size_t SHIFT_FOR_STORAGE_ELEMENT = 5;
static void generateWeightsTablesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void populateWeightsTablesPointersFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void populateWeightsTablesQuantizationFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void removeBiasTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void populateStorageElementPointersFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(GenerateWeightsTables)
        .setFunc(generateWeightsTablesFcn)
        .setDescription(
            "Generates weights tables for the Tasks that need them"
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
//    //31 bit is always 1 for storage element
//    std::int64_t constant_bias = std::pow(2, 31);

    std::vector<int64_t> unpopulated_offsets(height_width, 0);

    long int increment = inputChannels * (input->getDType().getSizeInBits() / 8);
    for(unsigned i = 0; i < height_width; ++i)
        unpopulated_offsets[i] = (i * increment << SHIFT_FOR_STORAGE_ELEMENT);
    activationStorageElement->populate(unpopulated_offsets, mv::Order("NHWC"));
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
