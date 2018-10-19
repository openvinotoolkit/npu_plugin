#include "include/mcm/computation/op/def/conv2d.hpp"
#include "include/mcm/deployer/blob_serialization/myriadX_hardware_descriptors.hpp"

mv::op::Conv2D::Conv2D(std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string& name) :
ComputationOp(OpType::Conv2D, name),
KernelOp(OpType::Conv2D, stride, padding, name),
SinkOp(OpType::Conv2D, 2, name)
{
    set<bool>("executable", true);
}

mv::Tensor mv::op::Conv2D::getOutputDef(std::size_t idx)
{

    // Will throw on error
    validOutputDef_(idx);

    auto input = getInputTensor(0);
    auto inputShape = input->getShape();
    auto weights = getInputTensor(1);
    auto weightsShape = weights->getShape();

    if (inputShape.ndims() != 3)
        throw(OpError(*this, "Invalid shape of the input tensor (input 0) - must have a dimensionality of 3, "
            " has " + std::to_string(inputShape.ndims())));

    if (weightsShape.ndims() != 4)
        throw(OpError(*this, "Invalid shape of the weights tensor (input 1) - must have a dimensionality of 4, "
            " has " + std::to_string(inputShape.ndims())));

    if (inputShape[2] != weightsShape[2])
        throw(OpError(*this, "Mismatch in channels dimensions between input tensor (input 0) and weights tensor (input 1) - "
            " input" + std::to_string(inputShape[3]) + ", weights " + std::to_string(weightsShape[2])));

    auto padding = get<std::array<unsigned short, 4>>("padding");
    auto stride = get<std::array<unsigned short, 2>>("stride");

    if (inputShape[0] + padding[0] + padding[1] < weightsShape[0])
        throw(OpError(*this, "Filter kernel width (" + std::to_string(weightsShape[0]) + ") exceeds the padded input width ("
            + std::to_string(inputShape[0] + padding[0] + padding[1]) + ")"));

    if (inputShape[1] + padding[2] + padding[3] < weightsShape[1])
        throw(OpError(*this, "Filter kernel height (" + std::to_string(weightsShape[1]) + ") exceeds the padded input height ("
            + std::to_string(inputShape[1] + padding[2] + padding[3]) + ")"));

    // Make sure that the result of subtract will not be negative
    Shape outputShape({(inputShape[0] + padding[0] + padding[1] - weightsShape[0]) / stride[0] + 1, (
        inputShape[1] + padding[2] + padding[3] - weightsShape[1]) / stride[1] + 1, weightsShape[3]});

    return Tensor(name_ + ":0", outputShape, input->getDType(), input->getOrder());

}

bool mv::op::Conv2D::isHardwarizeable(json::Object&)
{
    auto padding = get<std::array<unsigned short, 4>>("padding");
    auto stride = get<std::array<unsigned short, 2>>("stride");

    auto input = getInputTensor(0);
    auto inputShape = input->getShape();
    auto weights = getInputTensor(1);
    auto weightsShape = weights->getShape();

    // Check for supported padding
    if((padding[0] != 0 && padding[0] != weightsShape[0]/2) || (padding[2] != 0 && padding[2] != weightsShape[1]/2))
        return false;

    // Check for supported kernel sizes
    if(weightsShape[0] > 15 || weightsShape[1] > 15)
        return false;

    // Check for supported strides
    if(stride[0] > 8 || stride[1] > 8)
        return false;


    // Should handle dilation (WTF?) here

    return true;
}



void mv::op::Conv2D::gatherSerialFields()
{
    auto fp16_size = 2;

    if (this->hasAttr("NCE1_Compatible")){

        // Get all attrs:
        auto splits_over_H = this->get<size_t>("NCE1_SplitsOverHeight");
        auto DPUmodeVector = this->get<std::vector<size_t>>("NCE1_Modes");
        auto splits_over_iC = this->get<size_t>("NCE1_SplitsOverInputChannels");
        auto inputChannelsPadded = this->get<std::size_t>("NCE1_InputChannelsPadded");
        auto outputChannelsPadded = this->get<std::size_t>("NCE1_OutputChannelsPadded");
        auto inputWidthPadded = this->get<std::size_t>("NCE1_InputWidthPadded");
        auto outputWidthPadded = this->get<std::size_t>("NCE1_OutputWidthPadded");
        auto desc_count = this->get<std::size_t>("NCE1_DescriptorSplits");
        auto streamingMask = this->get<std::size_t>("NCE1_StreamingMask");

        auto input_lines_processed = this->get<std::vector<size_t>>("NCE1_InputLinesProcessed");
        auto output_lines_processed = this->get<std::vector<size_t>>("NCE1_OutputLinesProcessed");
        auto output_line_start = this->get<std::vector<size_t>>("NCE1_StartOutputLine");
        auto input_line_start = this->get<std::vector<size_t>>("NCE1_StartInputLine");

        auto radixX = this->getInputTensor(1)->getShape()[2];
        auto radixY = this->getInputTensor(1)->getShape()[3];

        this->set<unsigned>("SerialID", 33);    // To be moved?

        this->set<unsigned>("streamingMask", streamingMask );

        std::size_t total_size = this->getInputTensor(0)->getShape().totalSize();
        total_size *= inputChannelsPadded;
        total_size /= this->getInputTensor(0)->getShape()[2];
        this->set<unsigned>("inputSize", total_size*fp16_size);

        this->set<unsigned>("outputSize",
            this->getOutputTensor(0)->getShape().totalSize()*fp16_size);

        this->set<unsigned>("concatOffset", 0); // Not Supported...
        this->set<unsigned>("unloadCMX", 0); // Not Supported...
        this->set<unsigned>("overwriteInput", 0); // Not Supported...
        this->set<unsigned>("CMXSize", 256*1024);  // Magic Number...
        this->set<unsigned>("reluSHVAcc", 0); // Not Supported...
        this->set<unsigned>("shvNegSlope", 0); // Not Supported...
        this->set<unsigned>("shvPosSlope", 1065353216); // Magic Number...
        this->set<unsigned>("desc_count", desc_count);


        std::vector<unsigned> desc;
        std::vector<cnnConvolutionPoolStructure> descriptors = std::vector<cnnConvolutionPoolStructure>(desc_count);

        int i = -1;
        for (unsigned h = 0; h < splits_over_H; ++h)
        {
            for (unsigned oc = 0; oc < DPUmodeVector.size(); ++oc)
            {
                for (unsigned ic = 0; ic < splits_over_iC; ++ic)
                {
                    ++i;

                    auto input_width = inputWidthPadded;
                    auto output_channels = outputChannelsPadded;

                    descriptors[i].dataBaseAddr = 2 * input_width * input_line_start[h];    // TODO: Calculate 3f0 (1008)

                    if( this->getInputTensor(0)->getOrder() == mv::OrderType::RowInterleaved )
                    {
                        descriptors[i].dataBaseAddr *= inputChannelsPadded;    // TODO: Calculate 3f0 (1008)
                        // descriptors[i].dataLnStr = inputBlobTensor.strideY;
                        // descriptors[i].dataChStr = inputBlobTensor.strideZ;
                        descriptors[i].dataLnStr = 42;
                        descriptors[i].dataChStr = 42;
                    }
                    else
                    {
                        // descriptors[i].dataLnStr = inputBlobTensor.strideY;
                        // descriptors[i].dataChStr = inputBlobTensor.strideZ;
                        descriptors[i].dataLnStr = 42;
                        descriptors[i].dataChStr = 42;
                    }
                    descriptors[i].coeffBaseAddr = 0;
                    descriptors[i].biasBaseAddr = 0;
                    descriptors[i].scaleBaseAddr = 0;
                    //HACK FOR CONCAT
                    // descriptors[i].outBaseAddr = outputBlobTensor.strideZ * output_line_start[h];  // TODO: Calculate 3f0 (1008)
                    descriptors[i].outBaseAddr = 42;  // TODO: Calculate 3f0 (1008)

                    if( this->getOutputTensor(0)->getOrder() == mv::OrderType::RowInterleaved )
                    {
                        descriptors[i].outBaseAddr *= output_channels;    // TODO: Calculate 3f0 (1008)
                        // descriptors[i].outLnStr = outputBlobTensor.strideY;
                        // descriptors[i].outChStr = outputBlobTensor.strideZ;
                        descriptors[i].outLnStr = 42;
                        descriptors[i].outChStr = 42;
                    }
                    else
                    {
                        // descriptors[i].outLnStr = outputBlobTensor.strideY;
                        // descriptors[i].outChStr = outputBlobTensor.strideZ;
                        descriptors[i].outLnStr = 42;
                        descriptors[i].outChStr = 42;
                    }

                    auto weight_4dshape = this->getInputTensor(1)->getShape();

                    descriptors[i].coeffChStrIn = weight_4dshape[2]*weight_4dshape[3]*weight_4dshape[4]*2;
                    int inChans = inputChannelsPadded;

                    descriptors[i].coeffChStrOut = radixX * radixY * inChans * 2 * 8; // (fp16)

                    for(unsigned j = 0; j != 32; j++)
                        desc.push_back(((unsigned *) &descriptors[i])[j]);
                }

            }

        }

        this->set<std::vector<unsigned>>("descriptors", desc
        );
    }else{
        this->set<unsigned>("SerialID", 0);

        this->set<unsigned>("radixX",  this->getInputTensor(1)->getShape()[0]);
        this->set<unsigned>("radixY",  this->getInputTensor(1)->getShape()[1]);
        this->set<unsigned>("strideX",  this->get<std::array<unsigned short, 2>>("stride")[0]);
        this->set<unsigned>("strideY",  this->get<std::array<unsigned short, 2>>("stride")[1]);
        this->set<unsigned>("padX",  this->get<std::array<unsigned short, 4>>("padding")[0]);
        this->set<unsigned>("padY",  this->get<std::array<unsigned short, 4>>("padding")[2]);
        this->set<unsigned>("padStyle",  2);
        this->set<unsigned>("dilation",  1);

    }

}