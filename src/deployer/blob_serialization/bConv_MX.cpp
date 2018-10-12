#include <stdio.h>
#include "include/mcm/deployer/blob_serialization/bConv_MX.hpp"

namespace mv
{

    void bConv2D::writeStageInfo(mv::OpModel *om, mv::Blob_buffer *b)
    {


        std::cout << "RADIX : " << this->radixX << "*" <<  this->radixY << std::endl;

        int fp16_size = 2;

        mv::DataModel dm(*om);
        mv::ControlModel cm(*om);

        mv::Data::TensorIterator conv_bias = dm.tensorEnd();
        mv::Data::TensorIterator conv_scale = dm.tensorEnd();

        if(this->bias_name != "")
        {
            this->bias = dm.findTensor(this->bias_name);
            conv_bias = bias;
        }
        else
            std::cout << "Has No Bias" << std::endl;
        
        if(this->scale_name != "")
        {
            this->scale = dm.findTensor(this->scale_name);
            conv_scale = scale;
        }
        else
            std::cout << "Has No Bias" << std::endl;


        if (this->NCE1_Compatible)
        {

            // Hardware
            b->AddBytes(4, this->streamingMask);
            std::size_t total_size = this->input->getShape().totalSize();
            total_size /= this->input->getShape()[2];
            total_size *= this->inputChannelsPadded;
            b->AddBytes(4, total_size*fp16_size);
            b->AddBytes(4, this->output->getShape().totalSize()*fp16_size);
            b->AddBytes(4, this->concatOffset);
            b->AddBytes(4, this->unloadCMX);
            b->AddBytes(4, this->overwriteInput);
            b->AddBytes(4, this->CMXSize);
            b->AddBytes(4, this->reluSHVAcc);
            b->AddBytes(4, this->shvNegSlope);
            b->AddBytes(4, this->shvPosSlope);
            b->AddBytes(4, this->desc_count);


            std::cout << "in" << std::endl;
            Blob_Tensor inputBlobTensor = Blob_Tensor(dm, cm, b->reloc_table, this->input);
            std::cout << "Warning: forced Output, Taps Layout" << std::endl;
            std::cout << "out" << std::endl;
            Blob_Tensor outputBlobTensor = Blob_Tensor(dm, cm, b->reloc_table, this->output);
            std::cout << "taps" << std::endl;
            Blob_Tensor tapsBlobTensor = Blob_Tensor(dm, cm, b->reloc_table, this->taps);
            std::cout << "bias" << std::endl;
            Blob_Tensor biasBlobTensor = Blob_Tensor(dm, cm, b->reloc_table, conv_bias);
            std::cout << "scale" << std::endl;
            Blob_Tensor scaleBlobTensor = Blob_Tensor(dm, cm, b->reloc_table, conv_scale);


            auto input_shape = this->input->getShape();
            auto output_shape = this->output->getShape();

            /*
            std::cout << "Serializing a convolution performed in "
                       << this->DPUmodeVector.size()
                       << " shots, using "
                       << splits_over_iC
                       << " splits over IC and "
                       << splits_over_H
                       << " splits over H "
                       << std::endl;
            */

            //NOTE/TODO: This part probably has to be changed when split over IC come really into play.
            unsigned i = 0;
            for (unsigned oc = 0; oc != this->DPUmodeVector.size(); ++oc)
            {
                for (unsigned ic = 0; ic != splits_over_iC; ++ic)
                {
                    for (unsigned h = 0; h != splits_over_H; ++h)
                    {
                        i = oc*splits_over_iC*splits_over_H + ic*splits_over_H + h;

                        //unsigned int current_height, output_height;
                        //current_height = this->input_lines_processed[i];
                        //output_height = this->output_lines_processed[i];

                        //auto output_width = this->outputWidthPadded ; //output_shape[1];
                        auto input_width = this->inputWidthPadded; //input_shape[1];

                        //auto input_channels = input_shape[2];
                        auto output_channels = output_shape[2];

                        // this->descriptors[i].dataBaseAddr = i*0x3f0;    // TODO: Calculate 3f0 (1008)

                        this->descriptors[i].dataBaseAddr = 2*input_width*this->input_line_start[i];    // TODO: Calculate 3f0 (1008)

                        if( this->input->getOrder() == mv::OrderType::RowInterleaved ){
                            this->descriptors[i].dataBaseAddr *= this->inputChannelsPadded;    // TODO: Calculate 3f0 (1008)
                            this->descriptors[i].dataLnStr = inputBlobTensor.strideY;
                            this->descriptors[i].dataChStr = inputBlobTensor.strideZ;
                        }else{
                            this->descriptors[i].dataLnStr = inputBlobTensor.strideY;
                            this->descriptors[i].dataChStr = inputBlobTensor.strideZ;
                        }
                        this->descriptors[i].coeffBaseAddr = 0;
                        this->descriptors[i].biasBaseAddr = 0;
                        this->descriptors[i].scaleBaseAddr = 0;
                        //HACK FOR CONCAT
                        this->descriptors[i].outBaseAddr = 2* outputBlobTensor.strideZ*this->output_line_start[i];  // TODO: Calculate 3f0 (1008)
                        if( this->output->getOrder() == mv::OrderType::RowInterleaved ){
                            this->descriptors[i].outBaseAddr *= output_channels;    // TODO: Calculate 3f0 (1008)
                            this->descriptors[i].outLnStr = outputBlobTensor.strideY;
                            this->descriptors[i].outChStr = outputBlobTensor.strideZ;
                        }else{
                            this->descriptors[i].outLnStr = outputBlobTensor.strideY;
                            this->descriptors[i].outChStr = outputBlobTensor.strideZ;
                        }


                        auto weight_4dshape = this->taps->getShape();


                        this->descriptors[i].coeffChStrIn = weight_4dshape[2]*weight_4dshape[3]*weight_4dshape[4]*2;
                        int inChans = this->inputChannelsPadded;

                        this->descriptors[i].coeffChStrOut = this->radixX * this->radixY * inChans * 2 * 8; // (fp16)

                        char *byteArr = static_cast<char*>(static_cast<void*>(&this->descriptors[i]));
                        for(unsigned j = 0; j != 32; j++)
                            b->AddBytes(4, byteArr[j]);

                    }

                }
                
            }

            std::cout << "Finished convolution serialization" << std::endl;

            inputBlobTensor.write(b);
            outputBlobTensor.write(b);
            tapsBlobTensor.write(b);
            biasBlobTensor.write(b);
            scaleBlobTensor.write(b);

        }else{
            // Software

            b->AddBytes(4, this->radixX );
            b->AddBytes(4, this->radixY );
            b->AddBytes(4, this->strideX); //strideX  (0x70)
            b->AddBytes(4, this->strideY); //strideY

            // Ignore asymmetric padding (ignore elements elements p_r and p_b from padding = [p_l, p_r, p_t, p_b])
            b->AddBytes(4, this->padX);  // padX
            b->AddBytes(4, this->padY);  // padY
            b->AddBytes(4, this->padStyle);   // 0x80
            b->AddBytes(4, this->dilation);

            Blob_Tensor inputBlobTensor = Blob_Tensor(dm, cm, b->reloc_table, this->input);
            Blob_Tensor outputBlobTensor = Blob_Tensor(dm, cm, b->reloc_table, this->output);
            Blob_Tensor tapsBlobTensor = Blob_Tensor(dm, cm, b->reloc_table, this->taps);
            Blob_Tensor biasBlobTensor = Blob_Tensor(dm, cm, b->reloc_table, conv_bias);

            inputBlobTensor.write(b);
            outputBlobTensor.write(b);
            tapsBlobTensor.write(b);
            biasBlobTensor.write(b);

        }
    }

    bConv2D::bConv2D(mv::Control::OpListIterator it)
        :
          Blob_Op_Definition(),
          input((it->getInputTensor(0))),
          output((it->getOutputTensor(0))),
          taps((it->getInputTensor(1))),
          radixX(it->getInputTensor(1)->getShape()[2]),
          radixY(it->getInputTensor(1)->getShape()[3]),
          descriptors(nullptr)
    {

        if (it->hasAttr("bias"))
            this->bias_name = it->get<std::string>("bias");
        else
            this->bias_name = "";

        if (it->hasAttr("scale"))
        {
            this->scale_name = it->get<std::string>("scale");
            std::cout << "   in bConvHW contructor : scale tensor name = "<< this->scale_name << std::endl;

        }
        else
            this->scale_name = "";

        int mx_valid = 0;
        if (!it->hasAttr("NCE1_Compatible"))
            printf("Serializer Info: attribute NCE1_Compatible not present. Assuming False.\n");
        else
            mx_valid = it->get<int>("NCE1_Compatible");
        
        this->NCE1_Compatible = mx_valid;

        if(this->NCE1_Compatible)
        {
            // printf("Serializing a HW Conv\n");

            int cmxSize = 256*1024;

            if (! it->hasAttr("NCE1_AssignedCMX"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_AssignedCMX'. Defaulting to 256*1024\n");
            }
            else
            {
                cmxSize = it->get<int>("NCE1_AssignedCMX");
                printf("Serializer Info: Overriding attribute 'NCE1_AssignedCMX' to 256*1024\n");
                cmxSize = 256*1024;
            }
            if (! it->hasAttr("NCE1_SplitsOverHeight"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_SplitsOverHeight'. Defaulting to 1\n");
            }
            else
            {
                this->splits_over_H = it->get<size_t>("NCE1_SplitsOverHeight");
            }

            if (! it->hasAttr("NCE1_SplitsOverInputChannels"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_SplitsOverInputChannels'. Defaulting to 1\n");
            }
            else
            {
                this->splits_over_iC = it->get<size_t>("NCE1_SplitsOverInputChannels");
            }

            int descriptors_count = this->splits_over_iC * this->splits_over_H;

            if (!it->hasAttr("NCE1_StreamingMask"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_StreamingMask'. Defaulting to 1\n");
                this->streamingMask = 1;
            }
            else
            {
                this->streamingMask = it->get<std::size_t>("NCE1_StreamingMask");
            }
            printf("Serializer Info: Forcing Attribute 'NCE1_StreamingMask' to 1\n");
            this->streamingMask = 1;

            if (!it->hasAttr("NCE1_Modes"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_Modes'. Defaulting to 0\n");

                this->DPUmodeVector = {0};
                for(int i = 1; i != descriptors_count - 1; i++)
                {
                    this->DPUmodeVector.push_back(0);
                }
            }
            else
            {
                this->DPUmodeVector = it->get<std::vector<size_t>>("NCE1_Modes");
            }

            if (! it->hasAttr("NCE1_InputLinesProcessed"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_InputLinesProcessed'. Defaulting to 0\n");

                this->input_lines_processed = {0};
                for( int i = 1; i != descriptors_count - 1; i++)
                {
                    this->input_lines_processed.push_back(0);
                }
            }
            else
            {
                this->input_lines_processed = it->get<std::vector<size_t>>("NCE1_InputLinesProcessed");
            }
            if (! it->hasAttr("NCE1_OutputLinesProcessed"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_OutputLinesProcessed'. Defaulting to 0\n");

                this->output_lines_processed = {0};
                for( int i = 1; i != descriptors_count - 1; i++)
                {
                    this->output_lines_processed.push_back(0);
                }
            }
            else
            {
                this->output_lines_processed = it->get<std::vector<size_t>>("NCE1_StartOutputLine");
            }

            if (! it->hasAttr("NCE1_StartOutputLine"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_StartOutputLine'. Defaulting to 0\n");

                this->output_line_start = {0};
                for( int i = 1; i != descriptors_count - 1; i++)
                {
                    this->output_line_start.push_back(0);
                }
            }
            else
            {
                this->output_line_start = it->get<std::vector<size_t>>("NCE1_StartOutputLine");
            }
            if (! it->hasAttr("NCE1_StartInputLine"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_StartInputLine'. Defaulting to 0\n");

                this->input_line_start = {0};
                for( int i = 1; i != descriptors_count - 1; i++)
                {
                    this->input_line_start.push_back(0);
                }
            }
            else
            {
                this->input_line_start = it->get<std::vector<size_t>>("NCE1_StartInputLine");
            }
            this->concatOffset = 0; // Concat not supported currently
            this->unloadCMX = 0;
            this->overwriteInput = 0;

            this->CMXSize = cmxSize;
            this->reluSHVAcc = 0;
            double val = 0;
            this->shvNegSlope = static_cast<uint32_t>(val);
            this->shvPosSlope = 1065353216; //*(int * )(&val2);

            this->desc_count = it->get<std::size_t>("NCE1_DescriptorSplits");

            // this->descriptors = (cnnConvolutionPoolStructure *)malloc(128 * this->desc_count);
            this->descriptors = new cnnConvolutionPoolStructure[this->desc_count];

            std::vector<std::size_t> chPerRamBlock;
            std::vector<size_t> topJunk, bottomJunk;
            int localLS = 1;
            std::vector<std::size_t> localCS;
            std::vector<std::size_t> LPC;
            std::vector<std::size_t> minLines;
            int stride = 1;
            int padEn = 1;


            if (! it->hasAttr("NCE1_InputChannelsRamBlock"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_InputChannelsRamBlock'. Defaulting to 1\n");

                chPerRamBlock = {0};
                for( int i = 1; i != descriptors_count - 1; i++)
                {
                    chPerRamBlock.push_back(0);
                }
            }
            else
            {
                chPerRamBlock = it->get<std::vector<std::size_t>>("NCE1_InputChannelsRamBlock");
            }

            if (! it->hasAttr("NCE1_JunkOutputAfter"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_JunkOutputAfter'. Defaulting to 0\n");

                bottomJunk = {0};
                for( int i = 1; i != descriptors_count - 1; i++)
                {
                    bottomJunk.push_back(0);
                }
            }
            else
            {
                bottomJunk = it->get<std::vector<size_t>>("NCE1_JunkOutputAfter");
            }

            if (! it->hasAttr("NCE1_JunkOutputBefore"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_JunkOutputBefore'. Defaulting to 0\n");

                topJunk = {0};
                for( int i = 1; i != descriptors_count - 1; i++)
                {
                    topJunk.push_back(0);
                }
            }
            else
            {
                topJunk = it->get<std::vector<size_t>>("NCE1_JunkOutputBefore");
            }

            if (! it->hasAttr("NCE1_LocalLineStride"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_LocalLineStride'. Defaulting to 1\n");
            }
            else
            {
                localLS = it->get<std::size_t>("NCE1_LocalLineStride");
            }

            if (! it->hasAttr("NCE1_MinLines"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_MinLines'. Defaulting to 1\n");

                minLines = {1};
                for( int i = 1; i != descriptors_count - 1; i++)
                {
                    minLines.push_back(1);
                }
            }
            else
            {
                minLines = it->get<std::vector<std::size_t>>("NCE1_MinLines");
            }

            if (! it->hasAttr("stride"))
            {
                printf("Serializer Info: Needs Attribute 'stride'. Defaulting to 1\n");
            }
            else
            {
                stride = it->get<std::array<unsigned short, 2>>("stride")[0];
            }

            if (! it->hasAttr("padding"))
            {
                printf("Serializer Info: Needs Attribute 'padding'. Defaulting to 1\n");
            }
            else
            {
                padEn = it->get<std::array<unsigned short, 4>>("padding")[0];
            }

            if (! it->hasAttr("NCE1_LinesPerChannel"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_LinesPerChannel'. Defaulting to 1\n");

                LPC = {1};
                for( int i = 1; i != descriptors_count - 1; i++)
                {
                    LPC.push_back(1);
                }
            }
            else
            {
                LPC = it->get<std::vector<std::size_t>>("NCE1_LinesPerChannel");
            }

            if (! it->hasAttr("NCE1_LocalChannelStride"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_LocalChannelStride'. Defaulting to 1\n");

                localCS = {1};
                for( int i = 1; i != descriptors_count - 1; i++)
                {
                    localCS.push_back(1);
                }
            }
            else
            {
                localCS = it->get<std::vector<std::size_t>>("NCE1_LocalChannelStride");
            }

            if (! it->hasAttr("NCE1_InputChannelsPadded"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_InputChannelsPadded'. Defaulting to 1\n");

                this->inputChannelsPadded = 1;
            }
            else
            {
                this->inputChannelsPadded = it->get<std::size_t>("NCE1_InputChannelsPadded");
            }

            if (! it->hasAttr("NCE1_InputWidthPadded"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_InputWidthPadded'. Defaulting to 1\n");

                this->inputWidthPadded = 1;
            }
            else
            {
                this->inputWidthPadded = it->get<std::size_t>("NCE1_InputWidthPadded");
            }

            if (! it->hasAttr("NCE1_OutputWidthPadded"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_OutputWidthPadded'. Defaulting to 1\n");

                this->outputWidthPadded = 1;
            }
            else
            {
                this->outputWidthPadded = it->get<std::size_t>("NCE1_OutputWidthPadded");
            }


            unsigned i;
            for (unsigned oc = 0; oc != DPUmodeVector.size(); oc++)
            {
                for (unsigned ic = 0; ic != splits_over_iC; ic++)
                {
                    for (unsigned h = 0; h != splits_over_H; h++)
                    {
                        
                        i = oc*splits_over_iC*splits_over_H + ic*splits_over_H + h;
                        //this->descriptors[i] =  cnnConvolutionPoolStructure();

                        // Relations to other Descriptors
                        if (i+1 == this->desc_count)
                        {
                            this->descriptors[i].Line0.linkAddress = 0; // Last.
                        }else{
                            this->descriptors[i].Line0.linkAddress = 32*4;
                        }

                        this->descriptors[i].Line0.id = 0;

                        // Layer Meta Information - Layout & DataTypes
                        this->descriptors[i].Line0.type = NCE1_CONV;

                        if( this->input->getOrder() == mv::OrderType::RowInterleaved )
                            this->descriptors[i].Line0.interleavedInput = 1;
                        else
                            this->descriptors[i].Line0.interleavedInput = 0;

                        if( this->output->getOrder() == mv::OrderType::RowInterleaved ){
                            this->descriptors[i].Line0.interleavedOutput = 1;
                            this->descriptors[i].rsvd3_interleaved = 1;
                        }
                        else
                            this->descriptors[i].Line0.interleavedOutput = 0;

                        this->descriptors[i].Line0.cm = NCE1_DTYPE_FP16;
                        this->descriptors[i].Line0.dm = NCE1_DTYPE_FP16;


                        // Standard Fields for Convolution
                        // MX WEIGHTS SHAPE ASSUMED!!!
                        this->descriptors[i].kernelWidth = this->taps->getShape()[2] -1;
                        this->descriptors[i].kernelHeight = this->taps->getShape()[3] -1;

                        this->descriptors[i].chStride = stride -1;  // Stride of Kernel (Square only)

                        if (padEn > 0)
                            this->descriptors[i].padEn = 1;
                        else
                            this->descriptors[i].padEn = 0;

                        this->descriptors[i].padType = 0;   // Zero Padding

                        this->descriptors[i].inputWidth = this->input->getShape()[0] -1;

                        unsigned int current_height;
                        current_height = this->input_lines_processed[i];

                        this->descriptors[i].inputHeight =  current_height - 1;
                        this->descriptors[i].inputChannels = this->inputChannelsPadded -1;

                        this->descriptors[i].outputChannels = this->output->getShape()[2] -1;

                        // Myriad X DPU Assignment & Execution Configuration
                       
                        this->descriptors[i].Line0.mode = this->DPUmodeVector[oc];
                        this->descriptors[i].Line0.it = 0;  // Interrupt Trigger
                        this->descriptors[i].Line0.disInt = 0;  // 0 - Interrupts Enabled, 1 - Interrupts disabled.
                        this->descriptors[i].chPerRamBlock = chPerRamBlock[ic] -1;        // Input Channels per Ram Block


                        // Myriad X Compensation Fields
                        this->descriptors[i].topOutputJunk = topJunk[i];
                        this->descriptors[i].bottomOutputJunk = bottomJunk[i];

                        this->descriptors[i].localLs =  localLS;

                        this->descriptors[i].linesPerCh = std::min(LPC[oc] - 1, input_lines_processed[h] - 1);
                        this->descriptors[i].localCs = (this->descriptors[i].linesPerCh + 1) * this->descriptors[i].localLs;

                        this->descriptors[i].rud = 0;   // Re-Use bit
                        this->descriptors[i].minLines = minLines[ic] - 1;     // Minimum lines of data required to carry out function

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

                        // Fused Pooling
                        if (0)
                        {
                            this->descriptors[i].Line0.type = NCE1_CONV_POOL;
                        }
                        this->descriptors[i].avgPoolX = 0;
                        this->descriptors[i].poolType = 0;
                        this->descriptors[i].poolEn = 0;
                        this->descriptors[i].poolKernelHeight = 0;
                        this->descriptors[i].poolKernelWidth = 0;

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
        } else {
            // printf("Serializing a SW Conv\n");
            this->radixX = it->getInputTensor(1)->getShape()[0];
            this->radixY = it->getInputTensor(1)->getShape()[1];
            this->strideX = it->get<std::array<unsigned short, 2>>("stride")[0];
            this->strideY = it->get<std::array<unsigned short, 2>>("stride")[1];
            this->padX = it->get<std::array<unsigned short, 4>>("padding")[0];
            this->padY = it->get<std::array<unsigned short, 4>>("padding")[2];
            this->padStyle = 2; // HARDCODED.
            this->dilation = 1; // HARDCODED.

        }
    }

    bConv2D::~bConv2D()
    {
        if (this->descriptors != nullptr)
                delete [] this->descriptors;
    }

}
