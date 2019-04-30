#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include <math.h>

static void generateWeightsTablesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void populateWeightsTablesDataPointersFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void populateWeightsTablesSparsityPointersFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void populateWeightsTablesActivationBiasFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

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

    auto output = dpuTaskOp->getOutputTensor(0);
    auto input = dpuTaskOp->getInputTensor(0);

    if(dpuTaskOp->get<std::string>("taskOp") == "Conv")
    {
        // TODO
        // for (size_t i = 0; i < size; i+=4)
        //      weightsTableData[i] = ...;
    }
    else if(dpuTaskOp->get<std::string>("taskOp") == "ChannelMajorConvolution" || dpuTaskOp->get<std::string>("taskOp") == "DepthwiseConv")
    {
        // TODO
        // for (size_t i = 0; i < size; i+=4)
        //      weightsTableData[i] = ...;
    }
    // Max pooling does not need DataPointer

}

void populateWeightsTablesSparsityPointers(std::vector<int64_t>& weightsTableData, mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
{
    mv::OpModel om(model);

    auto output = dpuTaskOp->getOutputTensor(0);
    auto input = dpuTaskOp->getInputTensor(0);

    if(dpuTaskOp->get<std::string>("taskOp") == "Conv")
    {
        // TODO
        // for (size_t i = 0; i < size; i+=4)
        //      weightsTableData[i+1] = ...;
    }
    else if(dpuTaskOp->get<std::string>("taskOp") == "ChannelMajorConvolution" || dpuTaskOp->get<std::string>("taskOp") == "DepthwiseConv")
    {
        // TODO
        // for (size_t i = 0; i < size; i+=4)
        //      weightsTableData[i+1] = ...;
    }

}

static void populateWeightsTablesActivationAndBias(std::vector<int64_t>& weightsTableData, mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

     std::string opType = dpuTaskOp->get<std::string>("taskOp");

     auto output = dpuTaskOp->getOutputTensor(0);
     auto input = dpuTaskOp->getInputTensor(0);
     size_t outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];
     std::vector<int32_t> outputZeroPoint, inputZeroPoint, resultZeroPoint;
     std::vector<double> macScale, division, mantissa_v, mScaled, zeroPointScaled;
     std::vector<int> exponent_v, bits(outputChannels, 15), shift;
     double mantissa;
     int exponent;
     //Output without input impossible
    if (output->hasAttr("quantParams"))
    {
        auto outputQuantization = output->get<mv::QuantizationParams>("quantParams");
        outputQuantization.extendParamsToOutputChannelSize(outputChannels);
        outputZeroPoint = std::vector<int32_t>(outputQuantization.getZeroPoint().begin(), outputQuantization.getZeroPoint().end());
//                output->set<mv::QuantizationParams>("quantParams", outputQuantization);
        auto inputQuantization = input->get<mv::QuantizationParams>("quantParams");
        inputQuantization.extendParamsPartialToOutputChannelSize(outputChannels);
        if (opType == "MaxPool")
        {
            inputQuantization.extendParamsToOutputChannelSize(outputChannels);
            inputZeroPoint = std::vector<int32_t>(inputQuantization.getZeroPoint().begin(), inputQuantization.getZeroPoint().end());
            std::transform(outputZeroPoint.begin(), outputZeroPoint.end(), inputZeroPoint.begin(), std::back_inserter(outputZeroPoint)
                           , std::minus<int32_t>());
        }
//                input->set<mv::QuantizationParams>("quantParams", inputQuantization);
        //WEIGHTS
        std::vector<double> weightTensorScale(outputChannels, 1);
        if (dpuTaskOp->getInputTensor().size() > 1)
        {
            auto weights = dpuTaskOp->getInputTensor(1);
            if (weights->hasAttr("quantParams"))
            {
                auto weightQuantization = weights->get<mv::QuantizationParams>("quantParams");
                weightQuantization.extendParamsPartialToOutputChannelSize(outputChannels);
                //S1
                weightTensorScale = weightQuantization.getScale();
//                        weights->set<mv::QuantizationParams>("quantParams", weightQuantization);
            }
        }
        std::transform(weightTensorScale.begin(), weightTensorScale.end(), inputQuantization.getScale().begin(), std::back_inserter(macScale),
                                                          std::multiplies<double>());
        std::transform(macScale.begin(), macScale.end(), outputQuantization.getScale().begin(), std::back_inserter(division), std::divides<double>());

        for (auto it = division.begin(); it != division.end(); ++it)
        {
            mantissa = std::frexp(*it, &exponent);
            mantissa_v.push_back(mantissa);
            exponent_v.push_back(exponent);
        }

        std::transform(bits.begin(), bits.end(), exponent_v.begin(), std::back_inserter(shift), std::minus<int>());
        std::vector<double> power_v(outputChannels, pow(2.0, bits[0]));
        std::transform(mantissa_v.begin(), mantissa_v.end(), power_v.begin(), std::back_inserter(mScaled),
                                                          std::multiplies<double>());
        std::vector<uint16_t> mScaled_conv = std::vector<uint16_t>(mScaled.begin(), mScaled.end());
        std::transform(outputZeroPoint.begin(), outputZeroPoint.end(), division.begin(), std::back_inserter(zeroPointScaled), std::divides<double>());
        std::vector<int32_t> zeroPointScaled_conv = std::vector<int32_t>(zeroPointScaled.begin(), zeroPointScaled.end());
        std::vector <uint8_t> ser_shift = std::vector<uint8_t>(shift.begin(), shift.end());
        std::vector <uint16_t> ser_scale = std::vector<uint16_t>(mScaled_conv.begin(), mScaled_conv.end());
        outputQuantization.quantize(ser_shift, ser_scale);

        if (dpuTaskOp->hasAttr("bias"))
        {
            auto biasTensor = dm.getTensor(dpuTaskOp->get<std::string>("bias"));
            auto data = biasTensor->getIntData();
            std::transform(data.begin(), data.end(), zeroPointScaled_conv.begin(), data.begin(), std::plus<int32_t>());
            biasTensor->setDType(mv::DType("Int32"));
            biasTensor->populate(data);
        }
        else
        {
            mv::Order order(mv::Order::getColMajorID(1));
            const std::string biasTensorName = dpuTaskOp->getName() + "_bias";
            mv::Shape shape({outputChannels});
            std::vector<int64_t> calling_tensor = std::vector<int64_t>(zeroPointScaled_conv.begin(), zeroPointScaled_conv.end());
            auto biasTensor = dm.defineTensor(biasTensorName, shape, mv::DType("Int32"), order, calling_tensor);
            om.addAttr(dpuTaskOp, "bias", biasTensor->getName());
        }
    }

     std::vector<mv::DataElement> biasData;
     mv::Data::TensorIterator bias;

    if (dpuTaskOp->hasAttr("bias"))
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
        weightsTableData[i+2] = ((int32_t)mScaled[i/4] << 16) | ((int32_t)shift[i/4]) << 8;
        if (dpuTaskOp->hasAttr("bias"))
            weightsTableData[i+3] = biasData[i/4];
    }

    if (dpuTaskOp->hasAttr("bias"))
    {
        dm.undefineTensor(bias);
        dpuTaskOp->erase("bias");
    }
}

static void generateWeightsTablesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
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
                mv::Shape shape({outputChannels, 1, 1, 4});

                std::vector<int64_t> weightsTableData(shape.totalSize(), 0);

                populateWeightsTablesDataPointers(weightsTableData, dpuTaskOp, om);
                populateWeightsTablesSparsityPointers(weightsTableData, dpuTaskOp, om);
                populateWeightsTablesActivationAndBias(weightsTableData, dpuTaskOp, om);

                auto weightTable = om.weightsTable(weightsTableData, shape, mv::DType("Int32"), mv::Order("WHCN"), {{},{},{},{}}, kernelWeightsTableName);
                om.getSourceOp(weightTable)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
                unsigned newSize = dpuTaskOp->addInputTensor(weightTable);
                om.defineFlow(weightTable, dpuTaskOp, newSize - 1);
            }
        }
    }
}
