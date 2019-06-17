#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include <math.h>

static void generateSparsityMapsPopulatedTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(GenerateSparsityMapsPopulatedTensors)
        .setFunc(generateSparsityMapsPopulatedTensorsFcn)
        .setDescription(
            "Generates sparsity maps for populated tensors."
        );

    }
}

mv::Data::TensorIterator createFakeSparsityMap(mv::OpModel om, mv::Data::OpListIterator dpuTaskOp, const std::string& sparsityMapName, const mv::Shape& sparsityShape, const std::vector<int64_t>& sparsityMapData)
{
    auto sparsityMap = om.constantInt(sparsityMapData, sparsityShape, mv::DType("UInt8"), mv::Order("NCHW"), {{},{},{},{}},sparsityMapName);
    om.getSourceOp(sparsityMap)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
    unsigned newSize = dpuTaskOp->addInputTensor(sparsityMap);
    om.defineFlow(sparsityMap, dpuTaskOp, newSize - 1);

    return sparsityMap;
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

// The sparsity maps relative to populated tensors have to be generated BEFORE the dma passes.
// As they have to be DMAed into CMX.
static void generateSparsityMapsPopulatedTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::json::Object&)
{
    mv::OpModel om(model);

    auto globalConfigParams = model.getGlobalConfigParams();

    bool sparsity = globalConfigParams->hasAttr("Sparsity") ? globalConfigParams->get<bool>("Sparsity") : false;

    for(auto dpuTask = om.opBegin(); dpuTask != om.opEnd(); ++dpuTask)
    {
        if(dpuTask->getOpType() == "DPUTask")
        {
            std::string taskOp = dpuTask->get<std::string>("taskOp");
            pass.log(mv::Logger::MessageType::Debug, " taskOp "  + dpuTask->get<std::string>("taskOp"));
            bool isChannelMajorConv = taskOp == "ChannelMajorConvolution";
            bool isPooling = taskOp == "MaxPool";
            bool isDepthWiseConv = taskOp == "DepthwiseConv";

            //for max pooling and deptwise convolution and channel-major convolution we need to generate sparsity data
            //even if those layers does not support sparsity.
            if (isPooling || isDepthWiseConv || isChannelMajorConv)
            {
                uint16_t kernelW, kernelH;

                auto strides = dpuTask->get<std::array<unsigned short, 2>>("stride");
                auto inputChannels = dpuTask->getInputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION];
                auto outputChannels = dpuTask->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION];

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
                    kernelW = weightsShape[mv::KERNEL_WIDTH];
                    kernelH = weightsShape[mv::KERNEL_HEIGHT];
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

                int channelLenght = bitpattern.size();

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

                auto fakeSparsityMap = createFakeSparsityMap(om, dpuTask, mv::createFakeSparsityMapName(opName), sparsityShape, sparsityTensor.getIntData());
                fakeSparsityMap->set<int>("channelLength", channelLenght);

                dpuTask->set<bool>("fakeSparsity", true);
            }
            else if(sparsity)
            {
                //Here only in the case of ZMajorConvolution and sparsity is active
                auto weightsTensor = dpuTask->getInputTensor(1);
                weightsTensor->setOrder(mv::Order("NHWC"));

                // Sparsity map costant has to be created just once
                // and has to feed all the operations that are fed by
                // the tensor. The input slot will be the same of fakeSparsityMap
                if(weightsTensor->setSparse())
                {
                    //SparsityMap will be saved as attribute
                    auto smInternalTensor = weightsTensor->getSparsityMap();
                    auto sparsityMap = om.constantInt(smInternalTensor->getIntData(), smInternalTensor->getShape(), smInternalTensor->getDType(),
                                                      smInternalTensor->getOrder(), {{},{},{},{}}, smInternalTensor->getName());
                    auto weights = om.getSourceOp(weightsTensor);
                    om.getSourceOp(sparsityMap)->set<unsigned>("opId", weights->get<unsigned>("opId"));
                    auto outputFlows = mv::getOutputDataFlow(om, weights, false);
                    for(auto& output: outputFlows)
                    {
                        auto sink = output.first;
                        unsigned newSize = sink->addInputTensor(sparsityMap);
                        om.defineFlow(sparsityMap, sink, newSize - 1);
                    }
                }
            }
        }
    }
}


// The sparsity maps for unpopulated tensors have to be generated AFTER the dma passes.
// This is because sparsity for unpopulated tensor really resolves to allocating to extra unpopulated tensors
// That will be filled with information by the runtime. We don't need explicit DMA for them.
static void generateSparsityMapsUnpopulatedTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::json::Object&)
{
    mv::OpModel om(model);

    auto globalConfigParams = model.getGlobalConfigParams();

    bool sparsity = globalConfigParams->hasAttr("Sparsity") ? globalConfigParams->get<bool>("Sparsity") : false;

    if(sparsity)
    {
        for(auto dpuTask = om.opBegin(); dpuTask != om.opEnd(); ++dpuTask)
        {
            if(dpuTask->getOpType() == "DPUTask")
            {

            }
        }
    }
}
