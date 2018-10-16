#include "include/mcm/deployer/blob_serialization/bPooling_MX.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include <numeric>
#include <vector>


namespace mv
{

    void bPooling_MX::writeStageInfo(mv::OpModel& om, mv::Blob_buffer* b)
    {

        int fp16_size = 2;

        mv::DataModel dm(om);
        mv::ControlModel cm(om);

        if (this->NCE1_Compatible)
        {
            // Hardware
            b->AddBytes(4, this->streamingMask);
            b->AddBytes(4, this->input->getShape().totalSize()*fp16_size);
            b->AddBytes(4, this->output->getShape().totalSize()*fp16_size);
            b->AddBytes(4, this->concatOffset);
            b->AddBytes(4, this->unloadCMX);
            b->AddBytes(4, this->overwriteInput);
            b->AddBytes(4, this->CMXSize);
            b->AddBytes(4, this->reluSHVAcc);
            b->AddBytes(4, this->shvNegSlope);
            b->AddBytes(4, this->shvPosSlope);
            b->AddBytes(4, this->desc_count);


            mv::Data::TensorIterator pool_bias = dm.tensorEnd();
            mv::Data::TensorIterator pool_scale = dm.tensorEnd();
            mv::Data::TensorIterator pool_taps = dm.tensorEnd();


            Blob_Tensor inputBlobTensor = Blob_Tensor(dm, cm, b->reloc_table, this->input);
            Blob_Tensor outputBlobTensor = Blob_Tensor(dm, cm, b->reloc_table, this->output);

            Blob_Tensor scale = Blob_Tensor(dm, cm, b->reloc_table, pool_scale);
            Blob_Tensor bias = Blob_Tensor(dm, cm, b->reloc_table, pool_bias);
            Blob_Tensor taps = Blob_Tensor(dm, cm, b->reloc_table, pool_taps);

            std::cout << "Input width padded " << this->inputWidthPadded << std::endl;
            std::cout << "Input channels padded " << this->inputChannelsPadded << std::endl;
            std::cout << "Output channels padded " << this->outputChannelsPadded << std::endl;

            //NOTE/TODO: This part probably has to be changed when split over IC come really into play.
            int i = -1;
            for (unsigned h = 0; h < splits_over_H; ++h)
            {
                for (unsigned oc = 0; oc < this->DPUmodeVector.size(); ++oc)
                {
                    ++i;

                    //std::cout << "Filling descriptor " << i << std::endl;

                    auto input_width = this->inputWidthPadded; //input_shape[1];
                    auto output_channels = this->outputChannelsPadded;
                    auto output_channels_performed_so_far = std::accumulate(this->outputChannelPerformed.begin(), this->outputChannelPerformed.begin()+oc, 0);

                    // this->descriptors[i].dataBaseAddr = i*0x3f0;    // TODO: Calculate 3f0 (1008)

                    this->descriptors[i].dataBaseAddr = 2 * input_width * output_channels_performed_so_far;    // TODO: Calculate 3f0 (1008)

                    if( this->input->getOrder() == mv::OrderType::RowInterleaved )
                    {
                        this->descriptors[i].dataBaseAddr += this->input->getShape()[2] * 2 * input_width * this->input_line_start[h] ;    // TODO: Calculate 3f0 (1008)
                        this->descriptors[i].dataLnStr = inputBlobTensor.strideY;
                        this->descriptors[i].dataChStr = inputBlobTensor.strideZ;
                    }
                    else
                    {
                        this->descriptors[i].dataLnStr = inputBlobTensor.strideY;
                        this->descriptors[i].dataChStr = inputBlobTensor.strideZ;
                    }
                    //std::cout << "Descriptor " << i << " dataBaseAddr " << this->descriptors[i].dataBaseAddr << std::endl;
                    this->descriptors[i].coeffBaseAddr = 0;
                    this->descriptors[i].biasBaseAddr = 0;
                    this->descriptors[i].scaleBaseAddr = 0;
                    //HACK FOR CONCAT
                    this->descriptors[i].outBaseAddr = 2 * outputWidthPadded * output_channels_performed_so_far;  // TODO: Calculate 3f0 (1008)
                    if( this->output->getOrder() == mv::OrderType::RowInterleaved )
                    {
                        this->descriptors[i].outBaseAddr += outputChannelsPadded * 2 * outputWidthPadded * this->output_line_start[h];    // TODO: Calculate 3f0 (1008)
                        this->descriptors[i].outLnStr = outputBlobTensor.strideY;
                        this->descriptors[i].outChStr = outputBlobTensor.strideZ;
                    }
                    else
                    {
                        this->descriptors[i].outLnStr = outputBlobTensor.strideY;
                        this->descriptors[i].outChStr = outputBlobTensor.strideZ;
                    }

                    for(unsigned j = 0; j != 32; j++)
                        b->AddBytes(4, ((int *) &this->descriptors[i])[j]);
                }
            }

            b->reloc_table.push_entry(std::pair<int, bLocation>(0, bLocation::Constant ));
            b->reloc_table.push_entry(std::pair<int, bLocation>(0, bLocation::Constant ));
            b->reloc_table.push_entry(std::pair<int, bLocation>(0, bLocation::Constant ));

            inputBlobTensor.write(b);
            outputBlobTensor.write(b);
            scale.write(b);
            bias.write(b);
            taps.write(b);
        }
        else
        {
            // Software

            printf("Serialization Warning: Manual Override of Pooling Software layer order\n");
            this->output->setOrder(mv::OrderType::RowMajorPlanar);
            this->input->setOrder(mv::OrderType::RowMajorPlanar);

            Blob_Tensor inputBlobTensor = Blob_Tensor(dm, cm, b->reloc_table, this->input);
            Blob_Tensor outputBlobTensor = Blob_Tensor(dm, cm, b->reloc_table, this->output);

            b->AddBytes(4, this->radixX);
            b->AddBytes(4, this->radixY);
            b->AddBytes(4, this->strideX);
            b->AddBytes(4, this->strideY);
            b->AddBytes(4, this->padX);
            b->AddBytes(4, this->padY);
            b->AddBytes(4, this->padStyle);

            inputBlobTensor.write(b);
            outputBlobTensor.write(b);

        }
    }

    bPooling_MX::bPooling_MX(mv::Control::OpListIterator it)
        :
          Blob_Op_Definition(),
          input((it->getInputTensor(0))),
          output((it->getOutputTensor(0))),
          radixX(it->get<std::array<short unsigned, 2>>("kSize")[0]),
          radixY(it->get<std::array<short unsigned, 2>>("kSize")[1]),
          strideX(it->get<std::array<short unsigned, 2>>("stride")[0]),
          strideY(it->get<std::array<short unsigned, 2>>("stride")[1]),
          padX(0),
          padY(0),
          padStyle(2)
    {


        int mx_valid = 0;
        if (! it->hasAttr("NCE1_Compatible"))
            printf("Serializer Info: attribute NCE1_Compatible not present. Assuming False.\n");
        else
            mx_valid = it->get<int>("NCE1_Compatible");

        this->NCE1_Compatible = mx_valid;

        if(this->NCE1_Compatible)
        {
            // printf("Serializing a HW Pooling\n");

            //No padding on channels needed for pooling
            this->inputChannelsPadded = it->get<size_t>("NCE1_InputChannelsPadded");
            this->outputChannelsPadded = it->get<size_t>("NCE1_OutputChannelsPadded");

            int cmxSize = 256*1024;

            this->splits_over_H = it->get<size_t>("NCE1_SplitsOverHeight");
            this->desc_count = it->get<std::size_t>("NCE1_DescriptorSplits");
            this->streamingMask = it->get<std::size_t>("NCE1_StreamingMask");
            this->DPUmodeVector = it->get<std::vector<size_t>>("NCE1_Modes");
            this->input_lines_processed = it->get<std::vector<size_t>>("NCE1_InputLinesProcessed");
            this->output_lines_processed = it->get<std::vector<size_t>>("NCE1_OutputLinesProcessed");
            this->output_line_start = it->get<std::vector<size_t>>("NCE1_StartOutputLine");
            this->input_line_start = it->get<std::vector<size_t>>("NCE1_StartInputLine");

            this->concatOffset = 0; // Concat not supported currently
            this->unloadCMX = 0;
            this->overwriteInput = 0;

            this->CMXSize = cmxSize;
            this->reluSHVAcc = 0;
            double val = 0;
            this->shvNegSlope = *(int * )(&val);
            this->shvPosSlope = 1065353216; //*(int * )(&val2);

            // this->descriptors = (cnnConvolutionPoolStructure *)malloc(128 * this->desc_count);
            this->descriptors = std::vector<cnnConvolutionPoolStructure>(this->desc_count);

            std::vector<std::size_t> chPerRamBlock;
            std::vector<size_t> topJunk, bottomJunk;
            int localLS = 1;
            std::vector<std::size_t> localCS;
            std::vector<std::size_t> LPC;
            std::vector<std::size_t> minLines;
            int stride = 1;
            int padEn = 1;

            chPerRamBlock = it->get<std::vector<std::size_t>>("NCE1_InputChannelsRamBlock");
            bottomJunk = it->get<std::vector<size_t>>("NCE1_JunkOutputAfter");
            topJunk = it->get<std::vector<size_t>>("NCE1_JunkOutputBefore");
            localLS = it->get<std::size_t>("NCE1_LocalLineStride");
            minLines = it->get<std::vector<std::size_t>>("NCE1_MinLines");
            stride = it->get<std::array<unsigned short, 2>>("stride")[0];
            padEn = it->get<std::array<unsigned short, 4>>("padding")[0];
            LPC = it->get<std::vector<std::size_t>>("NCE1_LinesPerChannel");
            localCS = it->get<std::vector<std::size_t>>("NCE1_LocalChannelStride");
            this->inputWidthPadded = it->get<std::size_t>("NCE1_InputWidthPadded");
            this->outputChannelPerformed = it->get<std::vector<std::size_t>>("NCE1_OutputChannelsPerformed");
            this->outputWidthPadded = it->get<std::size_t>("NCE1_OutputWidthPadded");

            int i = -1;

            //Pooling's number of splitting over input channel is the same of splitting over output channels
            for (unsigned h = 0; h < splits_over_H; ++h)
            {
                for (unsigned oc = 0; oc < DPUmodeVector.size(); ++oc)
                {
                    ++i;
                    // Relations to other Descriptors
                    if (i+1 == (int)this->desc_count)
                        this->descriptors[i].Line0.linkAddress = 0; // Last.
                    else
                        this->descriptors[i].Line0.linkAddress = 32*4*(oc+1);

                    this->descriptors[i].Line0.id = 0;

                    // Layer Meta Information - Layout & DataTypes
                    this->descriptors[i].Line0.type = NCE1_POOL;

                    if( this->input->getOrder() == mv::OrderType::RowInterleaved )
                        this->descriptors[i].Line0.interleavedInput = 1;
                    else
                        this->descriptors[i].Line0.interleavedInput = 0;

                    if( this->output->getOrder() == mv::OrderType::RowInterleaved )
                    {
                        this->descriptors[i].Line0.interleavedOutput = 1;
                        this->descriptors[i].rsvd3_interleaved = 1;
                    }
                    else
                        this->descriptors[i].Line0.interleavedOutput = 0;

                    this->descriptors[i].Line0.cm = NCE1_DTYPE_FP16;
                    this->descriptors[i].Line0.dm = NCE1_DTYPE_FP16;


                    this->descriptors[i].chStride = stride - 1;  // Stride of Kernel (Square only)

                    if (padEn > 0)
                        this->descriptors[i].padEn = 1;
                    else
                        this->descriptors[i].padEn = 0;

                    this->descriptors[i].padType = 15;   // The very innovative ??? Padding

                    this->descriptors[i].inputWidth = this->input->getShape()[0] - 1;

                    unsigned int current_height;
                    current_height = this->input_lines_processed[h];

                    this->descriptors[i].inputHeight =  current_height - 1;


                    this->descriptors[i].outputChannels = this->outputChannelPerformed[oc] - 1;
                    this->descriptors[i].inputChannels =  this->descriptors[i].outputChannels;

                    // Myriad X DPU Assignment & Execution Configuration
                    this->descriptors[i].Line0.mode = this->DPUmodeVector[oc];
                    this->descriptors[i].Line0.it = 0;  // Interrupt Trigger
                    this->descriptors[i].Line0.disInt = 0;  // 0 - Interrupts Enabled, 1 - Interrupts disabled.

                    this->descriptors[i].chPerRamBlock = chPerRamBlock[oc] -1;        // Input Channels per Ram Block

                    // Myriad X Compensation Fields
                    this->descriptors[i].topOutputJunk = topJunk[h];
                    this->descriptors[i].bottomOutputJunk = bottomJunk[h];

                    this->descriptors[i].localLs =  localLS;

                    if(splits_over_H == 1)
                        this->descriptors[i].linesPerCh = LPC[oc] - 1;
                    else
                         this->descriptors[i].linesPerCh = input_lines_processed[h] - 1;
                    this->descriptors[i].localCs = (this->descriptors[i].linesPerCh + 1) * this->descriptors[i].localLs;

                    this->descriptors[i].rud = 0;   // Re-Use bit

                    this->descriptors[i].minLines = minLines[oc] - 1;     // Minimum lines of data required to carry out function

                    this->descriptors[i].coeffLpb = (this->descriptors[i].chPerRamBlock+1) * (this->descriptors[i].kernelWidth+1) * (this->descriptors[i].kernelHeight+1) - 1;
                    this->descriptors[i].css = (this->descriptors[i].kernelWidth + 1) * (this->descriptors[i].kernelHeight + 1) -1 ;
                    this->descriptors[i].outputX = this->output->getShape()[0];

                    // Myriad X - Splitting groups
                    this->descriptors[i].sohGroup = h;
                    this->descriptors[i].sodGroup = 0;

                    // Fused ReLU
                    if(it->hasAttr("postOpType") && it->get<mv::OpType>("postOpType") == mv::OpType::ReLU)
                    {
                        this->descriptors[i].t0 = 0;
                        this->descriptors[i].a0 = 0;
                        this->descriptors[i].a1 = 1;
                        this->descriptors[i].reluxEn = 0;
                        this->descriptors[i].reluEn = 1;
                    }
                    else
                    {
                        this->descriptors[i].t0 = 0;
                        this->descriptors[i].a0 = 0;
                        this->descriptors[i].a1 = 0;
                        this->descriptors[i].reluxEn = 0;
                        this->descriptors[i].reluEn = 0;
                    }

                    this->descriptors[i].avgPoolX = 0;
                    this->descriptors[i].poolType = it->getOpType() == mv::OpType::AvgPool2D ? 1 : 0;
                    this->descriptors[i].poolEn = 1;
                    this->descriptors[i].poolKernelHeight = radixY - 1;
                    this->descriptors[i].poolKernelWidth = radixX - 1;

                    // Reserved fields of the hw descriptor. Leave as zero or live in eternal fear.
                    this->descriptors[i].Line0.rsvd1 = 0;
                    this->descriptors[i].rsvd2 = 0;
                    this->descriptors[i].rsvd3 = 0;
                    this->descriptors[i].rsvd4 = 0;
                    this->descriptors[i].rsvd5 = 0;
                    this->descriptors[i].rsvd6 = 0;
                    this->descriptors[i].rsvd7 = 0;
                    this->descriptors[i].rsvd9 = 0;
                    this->descriptors[i].rsvd10 = 0;
                    this->descriptors[i].rsvd13 = 0;
                    this->descriptors[i].rsvd8 = 0;

                    // Palette for Weights Lookup (Currently Unsupported).
                    this->descriptors[i].p0 = 0;
                    this->descriptors[i].p1 = 0;
                    this->descriptors[i].p2 = 0;
                    this->descriptors[i].p3 = 0;
                    this->descriptors[i].p4 = 0;
                    this->descriptors[i].p5 = 0;
                    this->descriptors[i].p6 = 0;
                    this->descriptors[i].p7 = 0;
                    this->descriptors[i].p8 = 0;
                    this->descriptors[i].p9 = 0;
                    this->descriptors[i].p10 = 0;
                    this->descriptors[i].p11 = 0;
                    this->descriptors[i].p12 = 0;
                    this->descriptors[i].p13 = 0;
                    this->descriptors[i].p14 = 0;
                    this->descriptors[i].p15 = 0;
                }
            }
        }
        else
        {
            this->radixX = it->getInputTensor(0)->getShape()[0];
            this->radixY = it->getInputTensor(0)->getShape()[1];
            this->strideX = it->get<std::array<unsigned short, 2>>("stride")[0];
            this->strideY = it->get<std::array<unsigned short, 2>>("stride")[1];
            this->padX = it->get<std::array<unsigned short, 4>>("padding")[0];
            this->padY = it->get<std::array<unsigned short, 4>>("padding")[2];
            this->padStyle = 2; // HARDCODED.
            this->dilation = 1; // HARDCODED.
        }
    }
}
