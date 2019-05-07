#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include <math.h>

static void generateWeightsTablesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(GenerateWeightsTables)
        .setFunc(generateWeightsTablesFcn)
        .setDescription(
            "Generates weights tables for the Tasks that need them"
        );
    }
}

void populateWeightsTablesDataPointers(std::vector<int64_t>& weightsTableData, mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
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
            unsigned offset = 1024; // NOTE: Implementation defined
            unsigned increment = weights->getShape()[0];
            for (size_t i = 0; i < weightsTableData.size(); i+=4, offset +=increment)
                  weightsTableData[i] = offset;
        }
    }
    else if(dpuTaskOp->get<std::string>("taskOp") == "ChannelMajorConvolution" || dpuTaskOp->get<std::string>("taskOp") == "DepthwiseConv")
    {
        auto weights = dpuTaskOp->getInputTensor(1);
        unsigned offset = 1024; // NOTE: Implementation defined
        unsigned increment = weights->getShape()[0]; //WS dimension
        for (size_t i = 0; i < weightsTableData.size(); i+=4, offset +=increment)
              weightsTableData[i] = offset;
    }
    // Max pooling does not need DataPointer, neither does element wise

}

void populateWeightsTablesSparsityPointers(std::vector<int64_t>& weightsTableData, mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
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
            unsigned offset = 16777215; // NOTE: Implementation defined
            for (size_t i = 0; i < weightsTableData.size(); i+=4)
                  weightsTableData[i+1] = offset;
        }
    }
    else if(dpuTaskOp->get<std::string>("taskOp") == "ChannelMajorConvolution" || dpuTaskOp->get<std::string>("taskOp") == "DepthwiseConv"  || dpuTaskOp->get<std::string>("taskOp") == "MaxPool")
    {
        // We have fake sparsity here! Yuppi!
        auto activationWindow = dpuTaskOp->getInputTensor(dpuTaskOp->inputSlots() - 1);
        auto activationWindowSizeInWords = activationWindow->getShape().totalSize();
        auto activationWindowSizeInBytes = activationWindowSizeInWords * activationWindow->getDType().getSizeInBits() / 8;
        auto activationWindowBytesPerOutputChannel = activationWindowSizeInBytes / outputChannels;
        unsigned offset = 0; // NOTE: Implementation defined
        unsigned increment = activationWindowBytesPerOutputChannel;
        for (size_t i = 0; i < weightsTableData.size(); i+=4, offset +=increment)
              weightsTableData[i+1] = offset;
    }
    //Nothing to do for element wise

}

static void populateWeightsTablesActivationAndBias(std::vector<int64_t>& weightsTableData, mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    std::string taskOp = dpuTaskOp->get<std::string>("taskOp");
    bool isElementWise = (taskOp == "Add" || taskOp == "Subtract" || taskOp == "Multiply");

    if (isElementWise)
        return;

    mv::QuantizationParams quantParams = {{},{},{},{}};
    auto output = dpuTaskOp->getOutputTensor(0);
    auto outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];
    std::vector<int32_t> mScaled(outputChannels, 0);
    std::vector<int32_t> mShift(outputChannels, 0);
    if(output->hasAttr("quantParams"))
    {
        quantParams = dpuTaskOp->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams");
        if (quantParams.isEmpty() == false){
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
    for (size_t i = 0; i < weightsTableData.size(); i+=4)
    {
        weightsTableData[i+2] = (mScaled[i/4] << 16) | (mShift[i/4]) << 8;

        if (hasBias)
            weightsTableData[i+3] = biasData[i/4];
    }

    if (hasBias)
    {
        dm.undefineTensor(bias);
        dpuTaskOp->erase("bias");
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

                std::string kernelWeightsTableName(opName + "WeightsTable");
                auto output = dpuTaskOp->getOutputTensor(0);
                auto outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];

                // per channel layout:
                // 3 -> bias
                // 2 -> mult << 16 | round << 14 |  shift << 8 | prelu
                // 1 -> SP_PTR
                // 0 -> DATA_PTR
                mv::Shape shape({4, 1, 1, outputChannels});

                std::vector<int64_t> weightsTableData(shape.totalSize(), 0);

                populateWeightsTablesDataPointers(weightsTableData, dpuTaskOp, om);
                populateWeightsTablesSparsityPointers(weightsTableData, dpuTaskOp, om);
                populateWeightsTablesActivationAndBias(weightsTableData, dpuTaskOp, om);
                mv::QuantizationParams quantParams = {{},{},{},{}};

                auto weightTable = om.constantInt(weightsTableData, shape, mv::DType("Int32"), mv::Order("NWCH"), quantParams, kernelWeightsTableName);
                om.getSourceOp(weightTable)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
                unsigned newSize = dpuTaskOp->addInputTensor(weightTable);
                om.defineFlow(weightTable, dpuTaskOp, newSize - 1);
            }
        }
    }
}
