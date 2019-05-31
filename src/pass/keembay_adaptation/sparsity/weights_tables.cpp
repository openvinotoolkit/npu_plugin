#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include <math.h>

static void generateWeightsTablesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void populateWeightsTablesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(GenerateWeightsTables)
        .setFunc(generateWeightsTablesFcn)
        .setDescription(
            "Generates weights tables for the Tasks that need them"
        );

        MV_REGISTER_PASS(PopulateWeightsTables)
        .setFunc(populateWeightsTablesFcn)
        .setDescription(
            "Populate WeightsTables"
        );

    }
}

void populateWeightsTablesDataPointers(mv::Tensor& weightsTableData, mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
{
    mv::OpModel om(model);

    if(dpuTaskOp->get<std::string>("taskOp") == "Conv")
    {
        auto weights = dpuTaskOp->getInputTensor(1);
        if(weights->isSparse())
        {
            // TODO: Not handling at the moment
        }
        else
        {
            long int offset = weights->getAddress();
            long int increment = weights->getShape()[0];
            for (size_t i = 0; i < weightsTableData.size(); i+=4, offset +=increment)
                  weightsTableData(i) = offset;
        }
    }
    else if(dpuTaskOp->get<std::string>("taskOp") == "ChannelMajorConvolution" || dpuTaskOp->get<std::string>("taskOp") == "DepthwiseConv")
    {
        auto weights = dpuTaskOp->getInputTensor(1);
        long int offset = weights->getAddress(); // NOTE: Implementation defined
        long int increment = weights->getShape()[0]; //WS dimension
        for (size_t i = 0; i < weightsTableData.size(); i+=4, offset +=increment)
              weightsTableData(i) = offset;
    }
    // Max pooling does not need DataPointer

}

void populateWeightsTablesSparsityPointers(mv::Tensor& weightsTableData, mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
{
    mv::OpModel om(model);

    auto output = dpuTaskOp->getOutputTensor(0);
    auto input = dpuTaskOp->getInputTensor(0);
    unsigned outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];

    if(dpuTaskOp->get<std::string>("taskOp") == "Conv")
    {
        auto weights = dpuTaskOp->getInputTensor(1);
        if(weights->isSparse())
        {
            // TODO: Not handling at the moment
        }
        // Nothing to do here if is a dense ZMajor convolution
        else
        {
            long int offset = 16777215; // NOTE: Implementation defined
            for (size_t i = 0; i < weightsTableData.size(); i+=4)
                  weightsTableData(i+1) = offset;
        }
    }
    else if(dpuTaskOp->get<std::string>("taskOp") == "ChannelMajorConvolution" ||
            dpuTaskOp->get<std::string>("taskOp") == "DepthwiseConv"  ||
            dpuTaskOp->get<std::string>("taskOp") == "MaxPool")
    {
        // We have fake sparsity here! Yuppi!
        auto activationWindow = dpuTaskOp->getInputTensor(dpuTaskOp->inputSlots() - 2);
        auto activationWindowSizeInWords = activationWindow->getShape().totalSize();
        auto activationWindowSizeInBytes = activationWindowSizeInWords * activationWindow->getDType().getSizeInBits() / 8;
        auto activationWindowBytesPerOutputChannel = activationWindowSizeInBytes / outputChannels;
        long int offset = activationWindow->getAddress(); // NOTE: Implementation defined
        long int increment = activationWindowBytesPerOutputChannel;
        for (size_t i = 0; i < weightsTableData.size(); i+=4, offset +=increment)
              weightsTableData(i+1) = offset;
    }
}

void populateWeightsTablesActivationAndBias(mv::Tensor& weightsTableData, mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
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

    for (size_t i = 0; i < weightsTableData.size(); i+=4)
    {
        weightsTableData(i+2) = static_cast<long int>((mScaled[i/4] << 16) | (round32[i/4] << 14) | (mShift[i/4]) << 8);

        if (hasBias)
            weightsTableData(i+3) = biasData[i/4];
    }

    if (hasBias)
    {
        dm.undefineTensor(bias);
        dpuTaskOp->erase("bias");
    }
}

static void populateWeightsTablesFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
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
                // Necessary hack since data is copied with DMA and we are not using a shared_ptr
                auto weightsTable = dpuTaskOp->getInputTensor(dpuTaskOp->inputSlots()-1);
                auto weightsTableOp = om.getSourceOp(weightsTable);
                weightsTableOp = weightsTableOp.leftmostParent();
                weightsTable = weightsTableOp->getOutputTensor(0);

                populateWeightsTablesDataPointers(*weightsTable, dpuTaskOp, model);
                populateWeightsTablesSparsityPointers(*weightsTable, dpuTaskOp, model);
                populateWeightsTablesActivationAndBias(*weightsTable, dpuTaskOp, model);
            }
        }
    }
}


static void generateWeightsTablesFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

    for(auto dpuTaskOp = om.opBegin(); dpuTaskOp != om.opEnd(); ++dpuTaskOp)
    {
        if(dpuTaskOp->getOpType() == "DPUTask")
        {
            if((dpuTaskOp->get<std::string>("taskOp") == "Conv") ||
               (dpuTaskOp->get<std::string>("taskOp") == "ChannelMajorConvolution") ||
               (dpuTaskOp->get<std::string>("taskOp") == "MaxPool") ||
               (dpuTaskOp->get<std::string>("taskOp") == "DepthwiseConv"))
            {
                std::string opName = dpuTaskOp->getName();

                std::string kernelWeightsTableName(mv::createWeightTableName(opName));
                auto output = dpuTaskOp->getOutputTensor(0);
                auto outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];

                // per channel layout:
                // 3 -> bias
                // 2 -> mult << 16 | round << 14 |  shift << 8 | prelu
                // 1 -> SP_PTR
                // 0 -> DATA_PTR
                mv::Shape shape({4, 1, 1, outputChannels});

                std::vector<int64_t> weightsTableData(shape.totalSize(), 0);
                mv::QuantizationParams quantParams = {{},{},{},{}};

                auto weightTable = om.constantInt(weightsTableData, shape, mv::DType("Int32"), mv::Order("NWCH"), quantParams, kernelWeightsTableName);
                om.getSourceOp(weightTable)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
                unsigned newSize = dpuTaskOp->addInputTensor(weightTable);
                om.defineFlow(weightTable, dpuTaskOp, newSize - 1);
            }
        }
    }
}
