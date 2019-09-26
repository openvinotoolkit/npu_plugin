#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/tensor/shape.hpp"
#include <math.h>

static const std::size_t BYTES_PER_WT_ELEMENT = 4;
static void generateWeightsTablesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void populateWeightsTablesPointersFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void populateWeightsTablesQuantizationFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void removeBiasTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
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
    }
}
void populatePointerMultiCluster(mv::Data::TensorIterator weightsTableData, mv::Data::OpListIterator dpuTaskOp, const std::vector<int64_t>& increments, long int offset, std::size_t addingIndex, mv::ComputationModel &model, mv::Data::TensorIterator tensor)
{
    //std::cout << "Populating " << std::to_string(addingIndex) << " pointer of weights table for op " << dpuTaskOp->getName() << std::endl;
    if (dpuTaskOp->get<std::string>("splitStrategy") != "SplitOverK")
    {
        for (size_t i = 0, k = 0; i < weightsTableData->size(); i+=BYTES_PER_WT_ELEMENT, ++k)
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
            for (size_t j = 0; j < weightsTableData->getSubTensor(i).size(); j+=BYTES_PER_WT_ELEMENT)
                weightsTableData->at(j + addingIndex + totalSizeToIterate) = offset + increments[k++];

            // Preparing for next iteration
            sizeToIterate = tensor->getSubTensor(i).getShape()[mv::KERNEL_OUTPUT_CHANNELS] * BYTES_PER_WT_ELEMENT;
            totalSizeToIterate += sizeToIterate;
        }
    }
    return;
}

void populateWeightsTablesDataPointers(mv::Data::TensorIterator weightsTableData, mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
{
    // Max pooling does not need DataPointer
    // Eltwise doesn't have weights table at all
    if(dpuTaskOp->get<std::string>("taskOp") == "Conv" ||
       dpuTaskOp->get<std::string>("taskOp") == "ChannelMajorConvolution" ||
       dpuTaskOp->get<std::string>("taskOp") == "DepthwiseConv")
    {
        auto weights = dpuTaskOp->getInputTensor(1);
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

    if(dpuTaskOp->get<std::string>("taskOp") == "Conv")
    {
        auto weights = dpuTaskOp->getInputTensor(1);
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
            for (size_t i = 0; i < weightsTableData->size(); i+=BYTES_PER_WT_ELEMENT)
                  weightsTableData->at(i+1) = offset;
        }
    }
    else if(dpuTaskOp->get<std::string>("taskOp") == "ChannelMajorConvolution" ||
            dpuTaskOp->get<std::string>("taskOp") == "DepthwiseConv"  ||
            dpuTaskOp->get<std::string>("taskOp") == "MaxPool")
    {
        // We have fake sparsity here! Yuppi!
        auto activationWindow = dpuTaskOp->getInputTensor(dpuTaskOp->get<std::size_t>("fakeSparsityIndex"));
        auto activationWindowSizeInWords = activationWindow->getShape().totalSize();
        auto activationWindowSizeInBytes = activationWindowSizeInWords * activationWindow->getDType().getSizeInBits() / 8;

        // NOTE: Possible alignment issues here
        auto outputChannels = dpuTaskOp->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION];
        auto increment = activationWindowSizeInBytes / outputChannels;

        long int offset = activationWindow->getAddress();
        std::vector<int64_t> increments = std::vector<int64_t>(outputChannels, 0);
        for(unsigned i = 1; i < outputChannels; ++i)
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
    std::vector<int32_t> mScaled(outputChannels, 0);
    std::vector<int32_t> mShift(outputChannels, 0);
    if(output->hasAttr("quantParams"))
    {
        quantParams = dpuTaskOp->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams");
        if (!quantParams.isEmpty())
        {
            auto mult = quantParams.getMult();
            auto shift = quantParams.getShift();
            std::transform(mScaled.begin(), mScaled.end(), mult.begin(), mScaled.begin(), std::plus<int32_t>());
            std::transform(mShift.begin(), mShift.end(), shift.begin(), mShift.begin(), std::plus<int32_t>());
        }
    }
    std::vector<mv::DataElement> biasData;
    bool hasBias = dpuTaskOp->hasAttr("bias");
    mv::Data::TensorIterator bias;
    if (hasBias)
    {
        bias = dm.getTensor(dpuTaskOp->get<std::string>("bias"));
        biasData = bias->getData(); //Bias has the type Int32 in both cases above
    }
    // per channel layout:
    // 3 -> bias
    // 2 -> mult << 16 | round << 14 |  shift << 8 | prelu
    // 1 -> SP_PTR
    // 0 -> DATA_PTR
    // TODO mult & prelu are currently not implemented
    unsigned round_mode = 1;
    std::vector<int32_t> round32(outputChannels, round_mode);
    for (size_t i = 0; i < weightsTableData->size(); i+=BYTES_PER_WT_ELEMENT)
    {
        weightsTableData->at(i+2) = static_cast<long int>((mScaled[i/BYTES_PER_WT_ELEMENT] << 16) | (round32[i/BYTES_PER_WT_ELEMENT] << 14) | (mShift[i/BYTES_PER_WT_ELEMENT]) << 8);
        if (hasBias)
            weightsTableData->at(i+3) = biasData[i/BYTES_PER_WT_ELEMENT];
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
            if((dpuTaskOp->get<std::string>("taskOp") == "Conv") ||
               (dpuTaskOp->get<std::string>("taskOp") == "ChannelMajorConvolution") ||
               (dpuTaskOp->get<std::string>("taskOp") == "MaxPool") ||
               (dpuTaskOp->get<std::string>("taskOp") == "DepthwiseConv"))
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
            if((dpuTaskOp->get<std::string>("taskOp") == "Conv") ||
               (dpuTaskOp->get<std::string>("taskOp") == "ChannelMajorConvolution") ||
               (dpuTaskOp->get<std::string>("taskOp") == "MaxPool") ||
               (dpuTaskOp->get<std::string>("taskOp") == "DepthwiseConv"))
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

void debugPrint(mv::Data::TensorIterator weightsTableData)
{
    // Data pointer
    std::cout << weightsTableData->getName() + "Data pointer" << std::endl;
    for (size_t i = 0; i < weightsTableData->size(); i+=BYTES_PER_WT_ELEMENT)
        std::cout << static_cast<int64_t>(weightsTableData->at(i)) << std::endl;

    // Sparsity pointer
    std::cout << weightsTableData->getName() + "Sparsity pointer" << std::endl;
    for (size_t i = 0; i < weightsTableData->size(); i+=BYTES_PER_WT_ELEMENT)
        std::cout << static_cast<int64_t>(weightsTableData->at(i+1)) << std::endl;
}

static void populateWeightsTablesPointersFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    for(auto dpuTaskOp = om.opBegin(); dpuTaskOp != om.opEnd(); ++dpuTaskOp)
    {
        if(dpuTaskOp->getOpType() == "DPUTask")
        {
            if((dpuTaskOp->get<std::string>("taskOp") == "Conv") ||
               (dpuTaskOp->get<std::string>("taskOp") == "ChannelMajorConvolution") ||
               (dpuTaskOp->get<std::string>("taskOp") == "MaxPool") ||
               (dpuTaskOp->get<std::string>("taskOp") == "DepthwiseConv"))
            {
                auto weightsTable = dpuTaskOp->getInputTensor(dpuTaskOp->get<std::size_t>("weightsTableIndex"));
                populateWeightsTablesDataPointers(weightsTable, dpuTaskOp, model);
                populateWeightsTablesSparsityPointers(weightsTable, dpuTaskOp, model);

                debugPrint(weightsTable);
            }
        }
    }
}

static void generateWeightsTablesFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);
    mv::ControlModel cm(model);
    for(auto dpuTaskOp = om.opBegin(); dpuTaskOp != om.opEnd(); ++dpuTaskOp)
    {
        if(dpuTaskOp->getOpType() == "DPUTask")
        {
            std::string taskOp = dpuTaskOp->get<std::string>("taskOp");
            if(taskOp == "Conv" ||
               taskOp == "ChannelMajorConvolution" ||
               taskOp == "MaxPool" ||
               taskOp == "DepthwiseConv")
            {
                std::string opName = dpuTaskOp->getName();
                std::string kernelWeightsTableName(mv::createWeightTableName(opName));
                auto outputChannels = mv::round_up(dpuTaskOp->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION], 16);
                // per channel layout:
                // 3 -> bias
                // 2 -> mult << 16 | round << 14 |  shift << 8 | prelu
                // 1 -> SP_PTR
                // 0 -> DATA_PTR
                mv::Shape shape({BYTES_PER_WT_ELEMENT, 1, 1, outputChannels});
                std::vector<int64_t> weightsTableData(shape.totalSize(), 0);
                mv::QuantizationParams quantParams = {{},{},{},{}};
                auto weightTable = om.constantInt(weightsTableData, shape, mv::DType("Int32"), mv::Order("NWCH"), quantParams, kernelWeightsTableName);
                om.getSourceOp(weightTable)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
                om.getSourceOp(weightTable)->set<std::string>("populatedTensorType", "weightsTable");
                unsigned newSize = dpuTaskOp->addInputTensor(weightTable);
                om.defineFlow(weightTable, dpuTaskOp, newSize - 1);
                dpuTaskOp->set<size_t>("weightsTableIndex", newSize - 1);
            }
        }
    }
}
