#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include <math.h>

static void setSparsityFnc(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);


namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(Sparsity)
        .setFunc(setSparsityFnc)
        .setDescription(
            "Add sparsity map for layers/tensor that qualify"
        );
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

bool isZMajor(mv::Order order)
{
    return (order == mv::Order("NWHC") || order == mv::Order("WHC") || order == mv::Order("WHCN"));
}
void setSparsityFnc(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    pass.log(mv::Logger::MessageType::Debug, "Sparsity Optimization Started");

    mv::OpModel om(model);
    mv::DataModel dm(model);

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        bool isConv = (opIterator->getOpType() == "Conv");
        pass.log(mv::Logger::MessageType::Debug, " opType "  + opIterator->getOpType());
        bool isHWOp = opIterator->hasAttr("NCE1_Compatible") && opIterator->get<int>("NCE1_Compatible");
        bool isPooling = opIterator->getOpType() == "MaxPool" || opIterator->getOpType() == "AveragePool"; //AvgPool should be converted to DWConv
        bool isDepthWiseConv = opIterator->getOpType() == "DepthwiseConv";

        //for max pooling and deptwise convolution and channel-major convolution we need to generate sparsity data
        //even if those layers does not support sparsity.
        if (((isPooling || isDepthWiseConv) && isHWOp) ||
            (isConv && opIterator->getInputTensor().size() > 0 && opIterator->getInputTensor(0)->getOrder().isColMajorPlanar()))// == mv::Order("NCWH")))
        {

            uint16_t kernelW, kernelH;

            auto strides = opIterator->get<std::array<unsigned short, 2>>("stride");
            auto inputChannels = opIterator->getInputTensor(0)->getShape()[2];
            auto outputChannels = opIterator->getOutputTensor(0)->getShape()[2];

            if (isPooling)
            {
                auto kernelShape = opIterator->get<std::array<unsigned short, 2>>("kSize");
                kernelW = kernelShape[0];
                kernelH = kernelShape[1];
            }
            else// its a depthwise conv or conv with NCHW layout
            {
                auto weightsShape = opIterator->getInputTensor(1)->getShape();
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
                auto windowSparsitySize = static_cast<std::size_t>(std::ceil(windowsSize/8)); //how many bytes we need per window
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
            auto sparsityTensor = dm.defineTensor(opIterator->getName() + "_sparse_dw", sparsityShape, mv::DType("UInt8"), mv::Order("NCHW"), data);

            for(unsigned kx = 0; kx < sparsityShape[0]; ++kx)
                for(unsigned ky = 0; ky < sparsityShape[1]; ++ky)
                    for(unsigned ic = 0; ic < sparsityShape[2]; ++ic)
                        for(unsigned oc = 0; oc < sparsityShape[3]; ++oc)
                            sparsityTensor->at({kx, ky, ic, oc}) = static_cast<int64_t>(perChannelSparsity[ky*sparsityShape[0] + kx]);

            //Add sparsity map to conv
            om.addAttr(opIterator, "sparsityMap", sparsityTensor->getName());
            //TODO check Marco changed for weightsTable (added as Op rather than constant)
        }
        else
        {
            //ZMajor case => NWHC
            if (!isHWOp)
                continue;

            for (unsigned i = 0; i < opIterator->getInputTensor().size(); i++)
                if (isZMajor(opIterator->getInputTensor(i)->getOrder()))
                    opIterator->getInputTensor(i)->setSparse();
            for (unsigned i = 0; i < opIterator->getOutputTensor().size(); i++)
                if (isZMajor(opIterator->getOutputTensor(i)->getOrder()))
                    opIterator->getOutputTensor(i)->setSparse();

            //If HW layer and has weights
            if (opIterator->inputSlots() > 1 &&
                isZMajor(opIterator->getInputTensor(0)->getOrder()))
            {
                auto weights = opIterator->getInputTensor(1);
                weights->setOrder(mv::Order("NWHC"));
                weights->setSparse();
            }
        }

    }

    pass.log(mv::Logger::MessageType::Debug, "Sparsity Optimization Started");
}
