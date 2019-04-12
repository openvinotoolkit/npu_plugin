#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include <math.h>

static void generateSparsityMapsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void generateWeightsTablesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(GenerateSparsityMaps)
        .setFunc(generateSparsityMapsFcn)
        .setDescription(
            "Generates sparsity maps for the Tasks that need them"
        );

        MV_REGISTER_PASS(GenerateWeightsTables)
        .setFunc(generateWeightsTablesFcn)
        .setDescription(
            "Generates weights tables for the Tasks that need them"
        );
    }
}

mv::Data::TensorIterator createFakeSparsityMap(mv::OpModel om, mv::Data::OpListIterator dpuTaskOp, const std::string& sparsityMapName, const mv::Shape& sparsityShape, const std::vector<int64_t>& sparsityMapData)
{
    auto sparsityMap = om.constantInt(sparsityMapData, sparsityShape, mv::DType("UInt8"), mv::Order("NCHW"), sparsityMapName);
    om.getSourceOp(sparsityMap)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
    unsigned newSize = dpuTaskOp->addInputTensor(sparsityMap);
    om.defineFlow(sparsityMap, dpuTaskOp, newSize - 1);

    return sparsityMap;
}


//mv::Data::TensorIterator createSparsityMap(mv::OpModel om, mv::Data::OpListIterator dpuTaskOp, mv::Data::TensorIterator tensor)
//{
//    mv::ControlModel cm(om);

//    auto tensorShape = tensor->getShape();
//    std::string tensorName = tensor->getName();
//    tensorName.pop_back(); tensorName.pop_back(); //Necessary for .dot files

//    std::string sparsityMapName(tensorName+"SparsityMap");
//    auto sparsityMap = om.sparsityMap(mv::Shape({tensorShape[0], tensorShape[1], tensorShape[-1]}), mv::DType("Int32"), mv::Order(mv::Order::getRowMajorID(3)), sparsityMapName);
//    tensor->set<bool>("sparse", true);

//    if(tensor->isPopulated())
//    {
//        auto data = tensor->getData();
//        auto nonZeroElements = 0;
//        //default all zeropoints to zero
//        std::vector<unsigned> zeroPoint = tensor->getZeroPointsPerChannel();
//        std::vector<double> sparsityMapData(sparsityMap->getShape().totalSize());
//        std::vector<size_t> sub(tensorShape.ndims());
//        uint8_t map;

//        auto internalOrder = tensor->getInternalOrder();
//        auto order = tensor->getOrder();

//        for (size_t t = 0; t < tensorShape.totalSize(); t += 8)
//        {
//            map = 0;
//            for (size_t i = 0; i < 8; i++)
//            {
//                sub = order.indToSub(tensorShape, t+i);
//                if (sub[2] < tensorShape[2] && data[internalOrder.subToInd(tensorShape, sub)] != zeroPoint[sub[3]])
//                {
//                    map += 1 << i;
//                    nonZeroElements++;
//                }
//            }
//            sparsityMapData[t/8] = map;
//        }
//        sparsityMap->populate(sparsityMapData);

//        //BUG: Why does this give segfault?
//        sparsityMap = om.dMATask(sparsityMap, mv::DmaDirectionEnum::DDR2CMX);
//    }

//    dpuTaskOp->addInputTensor(sparsityMap);
//    om.defineFlow(sparsityMap, dpuTaskOp, dpuTaskOp->inputSlots());
//    std::string deallocationTaskName = sparsityMapName+"Deallocate";
//    om.deallocate(sparsityMap,deallocationTaskName);
//    cm.defineFlow(dpuTaskOp, om.getOp(deallocationTaskName));

//    return sparsityMap;
//}

void addWeightsTable(mv::ComputationModel& model, mv::OpModel om, mv::Data::OpListIterator dpuTaskOp, const std::string& kernelWeightsTableName)
{
    auto output = dpuTaskOp->getOutputTensor(0);
    auto input = dpuTaskOp->getInputTensor(0);
    auto outputChannels = output->getShape()[2];
    std::vector<int> shift(outputChannels, 0);
    std::vector<int16_t> mScaled(outputChannels, 0);

    mv::DataModel dm(model);

    if (output->hasAttr("quantizationParams") && input->hasAttr("quantizationParams") &&
        output->isQuantized() && input->isQuantized())
    {
        // Quantization for Gemmlowp output
        // S1 = weight scale
        // S2 = input activation scale
        // S3 = output activation scale
        // m  = (S1 * S2)/S3, scale for MAC output
        // zeroPointScaled = output zero point scaled to MAC output precision
        // biasScaled = bias scaled to MAC output precision

        auto inputQuantization = input->get<mv::QuantizationParams>("quantizationParams");
        auto scale = inputQuantization.getScale();
        std::vector<float> S2(scale.begin(), scale.end());

        auto outputQuantization = output->get<mv::QuantizationParams>("quantizationParams");
        scale = outputQuantization.getScale();
        std::vector<float> S3(scale.begin(), scale.end());

        auto zeroPointU =  outputQuantization.getZeroPoint();
        std::vector<int32_t> zeroPoint(zeroPointU.begin(), zeroPointU.end());

        std::string taskOp = dpuTaskOp->get<std::string>("taskOp");
        bool isPooling = taskOp == "MaxPool" || taskOp == "AvgPool";
        //Workaround for HW bug #227
        if (isPooling)
        {
            auto inZP = inputQuantization.getZeroPoint();
            std::vector<int32_t> inputZeroPoint(inZP.begin(), inZP.end());
            std::transform(zeroPoint.begin(), zeroPoint.end(), inputZeroPoint.begin(), zeroPoint.begin(), std::minus<int32_t>());
        }

        auto m = S2;
        if (dpuTaskOp->inputSlots() > 1)
        {
            auto weights = dpuTaskOp->getInputTensor(1);
            auto weightsQuantization = weights->get<mv::QuantizationParams>("quantizationParams");
            scale = weightsQuantization.getScale();
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

        if (dpuTaskOp->hasAttr("bias"))
        {
            auto bias = dm.getTensor(dpuTaskOp->get<std::string>("bias"));
            auto data = bias->getData();
            //auto biasQuantization = bias->get<mv::QuantizationParams>("quantizationParams");
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
    mv::Shape shape({outputChannels, 1, 1, 4});
    std::vector<mv::DataElement> biasData;
    bool hasBias = dpuTaskOp->hasAttr("bias");
    mv::Data::TensorIterator bias;
    if (hasBias)
    {
        bias = dm.getTensor(dpuTaskOp->get<std::string>("bias"));
        biasData = bias->getData(); //Bias has the type Int32 in both cases above
    }

    std::vector<int64_t> weightsTableData(shape.totalSize(), 0);
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

    auto weightTable = om.constantInt(weightsTableData, {outputChannels, 1, 1, 4}, mv::DType("UInt32"), mv::Order("WHCN"), kernelWeightsTableName);
    om.getSourceOp(weightTable)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
    unsigned newSize = dpuTaskOp->addInputTensor(weightTable);
    om.defineFlow(weightTable, dpuTaskOp, newSize - 1);

    return;
}


static void generateWeightsTablesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);

    for(auto dpuTask = om.opBegin(); dpuTask != om.opEnd(); ++dpuTask)
    {
        if(dpuTask->getOpType() == "DPUTask")
        {
            if((dpuTask->get<std::string>("taskOp") == "Conv") ||
               (dpuTask->get<std::string>("taskOp") == "ChannelMajorConvolution") ||
               (dpuTask->get<std::string>("taskOp") == "MaxPool") ||
               (dpuTask->get<std::string>("taskOp") == "DepthwiseConv"))
            {
                std::string opName = dpuTask->getName();

                std::string kernelWeightsTableName(opName + "WeightsTable");
                addWeightsTable(model, om, dpuTask, kernelWeightsTableName);
            }
        }
    }
}

uint16_t getWindowSize(uint16_t kx, uint16_t sx)
{
    //Find max mpe where if calc window <= 32
    //return window size for the max mpe
    uint16_t windowSize, maxMpeWindowSize = 64;
    int mpe = 1;

    //mpe in [1,2,4,8,16]
    while(mpe <= 16)
    {
        if (sx <= kx)
            windowSize = kx + sx * (mpe - 1);
        else
            windowSize = kx * mpe;

        if (windowSize <= 32)
            maxMpeWindowSize = windowSize;

        mpe *= 2;
    }

    return maxMpeWindowSize;
}

std::vector<int8_t> createBitPattern(uint16_t kernelW, uint16_t kernelH, uint16_t windowsSize, uint16_t inputChannels)
{
    std::vector<int8_t> bitpattern;
    bitpattern.reserve(windowsSize*kernelH*inputChannels);
    for(size_t c = 0; c < inputChannels; c++)
        for(size_t y = 0; y < kernelH; y++)
            for(size_t x = 0; x < windowsSize; x++)
                if (x < kernelW)
                    bitpattern.emplace_back(1);
                else
                    bitpattern.emplace_back(0);
    return bitpattern;
}

static void generateSparsityMapsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);


    for(auto dpuTask = om.opBegin(); dpuTask != om.opEnd(); ++dpuTask)
    {
        bool fakeSparsity = false;
        if(dpuTask->getOpType() == "DPUTask")
        {
            std::string taskOp = dpuTask->get<std::string>("taskOp");
            pass.log(mv::Logger::MessageType::Debug, " taskOp "  + dpuTask->get<std::string>("taskOp"));
            bool isConv = taskOp == "ChannelMajorConvolution";
            bool isPooling = taskOp == "MaxPool";
            bool isDepthWiseConv = taskOp == "DepthwiseConv";

            //for max pooling and deptwise convolution and channel-major convolution we need to generate sparsity data
            //even if those layers does not support sparsity.
            if (isPooling || isDepthWiseConv || isConv)
            {
                fakeSparsity = true;
                uint16_t kernelW, kernelH;

                auto strides = dpuTask->get<std::array<unsigned short, 2>>("stride");
                auto inputChannels = dpuTask->getInputTensor(0)->getShape()[2];
                auto outputChannels = dpuTask->getOutputTensor(0)->getShape()[2];

                // Using the check in this way, instead of on the operation type
                // makes this pass work on both aligned and unaligned convolutions.
                if (dpuTask->hasAttr("kSize"))
                {
                    auto kernelShape = dpuTask->get<std::array<unsigned short, 2>>("kSize");
                    kernelW = kernelShape[0];
                    kernelH = kernelShape[1];
                }
                else
                {
                    auto weightsShape = dpuTask->getInputTensor(1)->getShape();
                    kernelW = weightsShape[0];
                    kernelH = weightsShape[1];
                }

                auto windowsSize = getWindowSize(kernelW, strides[0]);

                pass.log(mv::Logger::MessageType::Debug, "windowSize " + std::to_string(windowsSize));
                pass.log(mv::Logger::MessageType::Debug, "OutputChannels " + std::to_string(outputChannels));

                std::vector<int8_t> bitpattern;
                std::vector<uint8_t> perChannelSparsity;
                std::vector<std::size_t> ndims(4);
                if (isPooling || isDepthWiseConv)
                {
                    bitpattern = std::move(createBitPattern(kernelW, kernelH, windowsSize, 1));
                    perChannelSparsity.resize(static_cast<std::size_t>(std::ceil(bitpattern.size() / 128.0)) * 16);//allocate once
                    ndims = {16, static_cast<std::size_t>(std::ceil(bitpattern.size() / 128.0)), 1, inputChannels};
                }
                else
                {
                    bitpattern = std::move(createBitPattern(kernelW, kernelH, windowsSize, inputChannels));
                    auto windowSparsitySize = static_cast<std::size_t>(std::ceil(windowsSize/8.0)); //how many bytes we need per window
                    auto NumberOfRowsSparistyBytes = static_cast<std::size_t>(std::ceil((kernelH * inputChannels * windowSparsitySize) / 16.0 ));
                    perChannelSparsity.resize(NumberOfRowsSparistyBytes * 16);//allocate once
                    ndims = {16, NumberOfRowsSparistyBytes, 1, outputChannels};
                }

                //populate sparsity
                pass.log(mv::Logger::MessageType::Debug, " perChannelSize = " + std::to_string(perChannelSparsity.size()) );
                for (size_t i = 0; i < bitpattern.size(); i++)
                {
                    //pass.log(mv::Logger::MessageType::Debug, " i = " + std::to_string(i));
                    //pass.log(mv::Logger::MessageType::Debug, " perChannelIDx = " + std::to_string(((i/128)*16 + (i%128)/8)) );
                    perChannelSparsity.at((i/128)*16 + (i%128)/8) |= bitpattern[i] << (i%8); //use at to check boundaries - just in case
                }

                //Create Tensor with Sparsity Data
                mv::Shape sparsityShape(ndims);
                std::vector<int64_t> data(sparsityShape.totalSize(), 0);
                //mv::Tensor sparsityTensor("backup", sparsityShape, mv::DType("UInt8"), mv::Order("WHCN"), data);
                auto sparsityTensor = mv::Tensor(dpuTask->getName() + "_sparse_dw", sparsityShape, mv::DType("UInt8"), mv::Order("NCHW"), data);

                for(unsigned kx = 0; kx < sparsityShape[0]; ++kx)
                    for(unsigned ky = 0; ky < sparsityShape[1]; ++ky)
                        for(unsigned ic = 0; ic < sparsityShape[2]; ++ic)
                            for(unsigned oc = 0; oc < sparsityShape[3]; ++oc)
                                sparsityTensor.at({kx, ky, ic, oc}) = static_cast<int64_t>(perChannelSparsity[ky*sparsityShape[0] + kx]);

                std::string opName = dpuTask->getName();

                std::string sparsityMapName(opName + "SparsityMap");
                auto fakeSparsityMap = createFakeSparsityMap(om, dpuTask, sparsityMapName, sparsityShape, sparsityTensor.getIntData());
                fakeSparsityMap->set<int>("channelLength", perChannelSparsity.size());

                dpuTask->set<bool>("fakeSparsity", true);
            }

        }
        if (!fakeSparsity)
        {
            if (dpuTask->getOpType() == "DPUTask" &&
                dpuTask->inputSlots() > 1 &&
                dpuTask->getInputTensor(0)->getOrder().isZMajor())
            {
                auto weights = dpuTask->getInputTensor(1);
                weights->setOrder(mv::Order("NHWC"));
                weights->setSparse();
                if (weights->isPopulated()) //that's always true
                {
                    //SparsityMap will be saved as attribute
                    auto smInternalTensor = weights->getSparsityMap();
                    auto sparsityMap = om.constantInt(smInternalTensor->getIntData(), smInternalTensor->getShape(), smInternalTensor->getDType(),
                        smInternalTensor->getOrder(), smInternalTensor->getName());
                    om.getSourceOp(sparsityMap)->set<unsigned>("opId", dpuTask->get<unsigned>("opId"));
                    unsigned newSize = dpuTask->addInputTensor(sparsityMap);
                    om.defineFlow(sparsityMap, dpuTask, newSize - 1);
                    dpuTask->set<std::string>("sparsityMap", sparsityMap->getName());
                }
            }
            unsigned n = dpuTask->getInputTensor().size();
            for (unsigned i = 0; i < n; ++i)
                if (dpuTask->getInputTensor(i)->getOrder().isZMajor() &&
                    !dpuTask->getInputTensor(i)->isPopulated()) //only weights are popualted, and we dont want to cover them here
                        dpuTask->getInputTensor(i)->setSparse();

            n = dpuTask->getOutputTensor().size();
            for (unsigned i = 0; i < n; ++i)
                if (dpuTask->getOutputTensor(i)->getOrder().isZMajor())
                    dpuTask->getOutputTensor(i)->setSparse();
        }
    }
}
