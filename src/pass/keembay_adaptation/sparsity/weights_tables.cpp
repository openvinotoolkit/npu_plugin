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
template <class T>
std::vector<T> extendToK(size_t size, std::vector<T> value)
{
    if (value.size() == 1)
        return mv::utils::generateSequence<T>(size, static_cast<T>(value[0]) , 0);

    if (value.size() == size)
        return value;

    throw mv::ArgumentError("QuantizationPass", "extendToK", "parameters dimensions doesn't match size of output_channels or 1",
                std::to_string(value.size()));
}

static void populateWeightsTablesActivationAndBias(std::vector<int64_t>& weightsTableData, mv::Data::OpListIterator dpuTaskOp, mv::ComputationModel& model)
{
    auto output = dpuTaskOp->getOutputTensor(0);
    auto input = dpuTaskOp->getInputTensor(0);
    auto outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];
    std::vector<int> shift(outputChannels, 0);
    std::vector<int16_t> mScaled(outputChannels, 0);

    mv::DataModel dm(model);
    mv::OpModel om(model);

    if (output->hasAttr("quantParams") && input->hasAttr("quantParams") &&
        output->isQuantized() && input->isQuantized())
    {
        // Quantization for Gemmlowp output
        // S1 = weight scale
        // S2 = input activation scale
        // S3 = output activation scale
        // m  = (S1 * S2)/S3, scale for MAC output
        // zeroPointScaled = output zero point scaled to MAC output precision
        // biasScaled = bias scaled to MAC output precision

        auto inputQuantization = input->get<mv::QuantizationParams>("quantParams");
        //inputQuantization.extendParamsToOutputChannelSize(outputChannels);

        auto scale = extendToK(outputChannels, inputQuantization.getScale());
        std::vector<float> S2(scale.begin(), scale.end());

        auto outputQuantization = output->get<mv::QuantizationParams>("quantParams");
        scale = extendToK(outputChannels, outputQuantization.getScale());
        std::vector<float> S3(scale.begin(), scale.end());

        auto zeroPointU =  extendToK(outputChannels, outputQuantization.getZeroPoint());
        std::vector<int32_t> zeroPoint(zeroPointU.begin(), zeroPointU.end());

        std::string taskOp = dpuTaskOp->get<std::string>("taskOp");
        bool isPooling = taskOp == "MaxPool" || taskOp == "AvgPool";
        //Workaround for HW bug #227
        if (isPooling)
        {
            auto inZP = extendToK(outputChannels, inputQuantization.getZeroPoint());
            std::vector<int32_t> inputZeroPoint(inZP.begin(), inZP.end());
            std::transform(zeroPoint.begin(), zeroPoint.end(), inputZeroPoint.begin(), zeroPoint.begin(), std::minus<int32_t>());
        }

        auto m = S2;
        if (dpuTaskOp->inputSlots() > 1)
        {
            auto weights = dpuTaskOp->getInputTensor(1);
            auto weightsQuantization = weights->get<mv::QuantizationParams>("quantParams");
            scale = extendToK(outputChannels, weightsQuantization.getScale());
            std::vector<float> S1(scale.begin(), scale.end());
            //S1*S2
            std::transform(m.begin(), m.end(), S1.begin(), m.begin(), std::multiplies<float>());
        }

        // Fuse ReLU into quantization (i.e. make ReLU == saturation), will be done using a separate pass

        // m / S3
        std::transform(m.begin(), m.end(), S3.begin(), m.begin(), std::divides<float>());

        //TODO need to handle 16bits case - per Alessandro bias need to be converted to int32
        auto bits = 15;
        auto mSize = m.size();
        int exponent;
        double mantissa;

        for (size_t i = 0; i < mSize; i++)
        {
            mantissa = std::frexp(m[i], &exponent);
            shift[i] = bits - exponent;
            mScaled[i] = (mantissa * pow(2, bits));
        }
        std::vector<int32_t> zeroPointScaled(m.size());
        std::transform(zeroPoint.begin(), zeroPoint.end() , m.begin(), zeroPointScaled.begin(), std::divides<float>());

        std::vector <uint8_t> ser_shift = std::vector<uint8_t>(shift.begin(), shift.end());
        std::vector <uint16_t> ser_scale = std::vector<uint16_t>(mScaled.begin(), mScaled.end());
        //outputQuantization.quantize(ser_shift, ser_scale);

        if (dpuTaskOp->hasAttr("bias"))
        {
            auto bias = dm.getTensor(dpuTaskOp->get<std::string>("bias"));
            auto data = bias->getData();
            //auto biasQuantization = bias->get<mv::QuantizationParams>("quantParams");
            //auto Z_bias = biasQuantization.getZeroPoint();
            //auto S_bias = biasQuantization.getScale();
            std::transform(data.begin(), data.end(), zeroPointScaled.begin(), data.begin(), std::plus<int64_t>());
            bias->setDType(mv::DType("Int32"));
            bias->populate(data);

        }
        else
        {
            mv::Order order(mv::Order::getColMajorID(1));
            const std::string biasTensorName = dpuTaskOp->getName() + "_bias";
            mv::Shape shape({outputChannels});
            std::vector<int64_t> zeroPointScaled64(zeroPointScaled.begin(), zeroPointScaled.end());

            auto biasTensor = dm.defineTensor(biasTensorName, shape, mv::DType("Int32"), order, zeroPointScaled64);
            om.addAttr(dpuTaskOp, "bias", biasTensor->getName());
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
        weightsTableData[i+2] = ((int32_t)mScaled[i/4] << 16) | ((int32_t)shift[i/4]) << 8;
        if (hasBias)
            weightsTableData[i+3] = biasData[i/4];
    }

    if (hasBias)
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
