#include "include/mcm/computation/op/def/fully_connected.hpp"

mv::op::FullyConnected::FullyConnected(const std::string &name) :
ComputationOp(OpType::FullyConnected, name),
SinkOp(OpType::FullyConnected, 2, name),
SourceOp(OpType::FullyConnected, 1, name)
{
    set<bool>("executable", true);
}


mv::Tensor mv::op::FullyConnected::getOutputDef(std::size_t idx)
{

    // Will throw on error
    validOutputDef_(idx);

    auto input0 = getInputTensor(0);
    auto input0Shape = input0->getShape();
    auto input1Shape = getInputTensor(1)->getShape();

    if (input1Shape.ndims() != 2)
        throw(OpError(*this, "Invalid shape of the weights tensor (input 1) - must have a dimensionality of 2, "
            " has " + std::to_string(input1Shape.ndims())));

    if (input0Shape.totalSize() != input1Shape[0])
        throw(OpError(*this, "Inconsistent total size of input tensor (input 0) " + std::to_string(input0Shape.totalSize()) +
            " and 1st dimension of weights tensor (input 1) " + std::to_string(input1Shape[0])));

    return Tensor(name_ + ":0", {1, input1Shape[1]}, input0->getDType(), input0->getOrder());

}

bool mv::op::FullyConnected::isHardwarizeable(json::Object&)
{
    return false;
}

void mv::op::FullyConnected::gatherSerialFields()
{
    auto fp16_size = 2;

    if (this->hasAttr("NCE1_Compatible") && this->get<int>("NCE1_Compatible") ){

        this->set<unsigned>("SerialID", 35);
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

        this->set<unsigned>("SerialID", 34);    // To be moved?

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

        this->set<std::vector<unsigned>>("descriptors", desc);
    }else{
        this->set<unsigned>("SerialID", 4);
    }
}