#include <stdio.h>
#include "include/mcm/deployer/blob_serialization/bConv_MX.hpp"

namespace mv
{

    void bConv2D::writeStageInfo(mv::OpModel * om, mv::Blob_buffer* b)
    {


        std::cout << "RADIX : " << this->radixX << "*" <<  this->radixY << std::endl;

        int fp16_size = 2;

        mv::DataModel dm(*om);
        mv::ControlModel cm(*om);

        mv::Data::TensorIterator *conv_bias;
        mv::Data::TensorIterator *conv_scale;

        if(this->bias_name != "")
        {
            this->bias = dm.findTensor(this->bias_name);
            conv_bias = &this->bias;
        }
        else
        {
            conv_bias = NULL ;
        }

        if(this->scale_name != "")
        {
            this->scale = dm.findTensor(this->scale_name);
            conv_scale = &this->scale;
        }
        else
        {
            conv_scale = NULL ;
        }


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


            Blob_Tensor inputBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->input);
            Blob_Tensor outputBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->output);
            Blob_Tensor tapsBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->taps);
            Blob_Tensor biasBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, conv_bias);
            Blob_Tensor scaleBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, conv_scale);

            for (unsigned i = 0; i != this->desc_count; i++)
            {

                unsigned int original_height = this->input->getShape()[1];
                unsigned int current_height;
                if (i+1 == this->desc_count){   // Last Descriptor may be an unequal height to the rest.
                    int surplus = ceil(original_height/(float)this->desc_count)*this->desc_count - original_height;
                    current_height = ceil(original_height/(float)this->desc_count) - surplus;
                }else{
                    current_height = ceil(original_height/(float)this->desc_count);
                }


                auto input_shape = this->input->getShape();
                auto output_shape = this->output->getShape();
                // this->descriptors[i].dataBaseAddr = i*0x3f0;    // TODO: Calculate 3f0 (1008)
                this->descriptors[i].dataBaseAddr = i*2*input_shape[1]*current_height;    // TODO: Calculate 3f0 (1008)
                this->descriptors[i].coeffBaseAddr = 0;
                this->descriptors[i].biasBaseAddr = 0;
                this->descriptors[i].scaleBaseAddr = 0;
                this->descriptors[i].outBaseAddr = i*2*output_shape[1]*current_height;  // TODO: Calculate 3f0 (1008)

                this->descriptors[i].dataChStr = inputBlobTensor.strideZ;
                this->descriptors[i].dataLnStr = inputBlobTensor.strideY;

                auto weight_4dshape = this->taps->getShape();

                this->descriptors[i].coeffChStrIn = weight_4dshape[4]*2;
                int inChans = weight_4dshape[1];

                this->descriptors[i].coeffChStrOut = this->radixX * this->radixY * inChans * 2 * 8; // (fp16)

                this->descriptors[i].outLnStr = outputBlobTensor.strideY;
                this->descriptors[i].outChStr = outputBlobTensor.strideZ;

                for(unsigned j = 0; j != 32; j++){
                    b->AddBytes(4, ((int *) &this->descriptors[i])[j]);
                }
            }

            b->reloc_table.push_entry(std::pair<int, bLocation>(0, bLocation::Constant ));
            b->reloc_table.push_entry(std::pair<int, bLocation>(0, bLocation::Constant ));

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


            int fp16_size = 2;
            mv::DataModel dm(*om);
            mv::ControlModel cm(*om);

            Blob_Tensor inputBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->input);
            Blob_Tensor outputBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->output);
            Blob_Tensor tapsBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->taps);
            Blob_Tensor biasBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, conv_bias);

            inputBlobTensor.write(b);
            outputBlobTensor.write(b);
            tapsBlobTensor.write(b);
            biasBlobTensor.write(b);

        }
    }

    bConv2D::bConv2D(mv::ComputationOp* it)
        :
          Blob_Op_Definition(),
          input((it->getInputTensor(0))),
          output((it->getOutputTensor(0))),
          taps((it->getInputTensor(1))),
          radixX(it->getInputTensor(1)->getShape()[2]),
          radixY(it->getInputTensor(1)->getShape()[3])
    {

        if (it->hasAttr("bias"))
        {
            this->bias_name = it->getAttr("bias").getContent<std::string>();
        }
        else
        {
            this->bias_name = "";
        }

        if (it->hasAttr("scale"))
        {
            this->scale_name = it->getAttr("scale").getContent<std::string>();
            std::cout << "   in bConvHW contructor : scale tensor name = "<< this->scale_name << std::endl;

        }
        else
        {
            this->scale_name = "";
        }


        int mx_valid = 0;
        if (! it->hasAttr("NCE1_Compatible"))
        {
            printf("Serializer Info: attribute NCE1_Compatible not present. Assuming False.\n");
        }
        else
        {
            mx_valid = it->getAttr("NCE1_Compatible").getContent<int>();
        }
        this->NCE1_Compatible = mx_valid;

        if(this->NCE1_Compatible){
            // printf("Serializing a HW Conv\n");

            int cmxSize = 256*1024;
            int splits_over_H = 1, splits_over_oC = 1;

            if (! it->hasAttr("NCE1_AssignedCMX"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_AssignedCMX'. Defaulting to 256*1024\n");
            }
            else
            {
                cmxSize = it->getAttr("NCE1_AssignedCMX").getContent<int>();
                printf("Serializer Info: Overriding attribute 'NCE1_AssignedCMX' to 256*1024\n");
                cmxSize = 256*1024;
            }

            if (! it->hasAttr("NCE1_SplitsOverH"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_SplitsOverH'. Defaulting to 1\n");
            }
            else
            {
                splits_over_H = it->getAttr("NCE1_SplitsOverH").getContent<int>();
            }

            if (! it->hasAttr("NCE1_SplitsOverC"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_SplitsOverC'. Defaulting to 1\n");
            }
            else
            {
                splits_over_oC = it->getAttr("NCE1_SplitsOverC").getContent<int>();
            }

            int descriptors_count = splits_over_oC * splits_over_H;

            if (! it->hasAttr("NCE1_StreamingMask"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_StreamingMask'. Defaulting to 1\n");
                this->streamingMask = 1;
            }
            else
            {
                this->streamingMask = it->getAttr("NCE1_StreamingMask").getContent<int>();
            }
            if (! it->hasAttr("NCE1_Modes"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_Modes'. Defaulting to 0\n");

                this->DPUmodeVector = {0};
                for( int i = 1; i != descriptors_count - 1; i++)
                {
                    this->DPUmodeVector.push_back(0);
                }
            }
            else
            {
                this->DPUmodeVector = it->getAttr("NCE1_Modes").getContent<dynamic_vector<unsigned>>();
            }

            this->concatOffset = 0; // Concat not supported currently
            this->unloadCMX = 0;
            this->overwriteInput = 0;

            this->CMXSize = cmxSize;
            this->reluSHVAcc = 0;
            float val = 0;
            float val2 = 1;
            this->shvNegSlope = *(int * )(&val);
            this->shvPosSlope = *(int * )(&val2);

            this->desc_count = descriptors_count;

            // this->descriptors = (cnnConvolutionPoolStructure *)malloc(128 * this->desc_count);
            this->descriptors = new cnnConvolutionPoolStructure[this->desc_count];

            dynamic_vector<unsigned> chPerRamBlock;
            int topJunk = 0, bottomJunk = 0;
            int localLS = 1;
            dynamic_vector<unsigned> localCS;
            dynamic_vector<unsigned> LPC;
            dynamic_vector<unsigned> minLines;
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
                chPerRamBlock = it->getAttr("NCE1_InputChannelsRamBlock").getContent<dynamic_vector<unsigned>>();
            }

            if (! it->hasAttr("NCE1_TopOutputJunk"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_TopOutputJunk'. Defaulting to 0\n");
            }
            else
            {
                topJunk = it->getAttr("NCE1_TopOutputJunk").getContent<int>();
            }

            if (! it->hasAttr("NCE1_BottomOutputJunk"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_BottomOutputJunk'. Defaulting to 0\n");
            }
            else
            {
                bottomJunk = it->getAttr("NCE1_BottomOutputJunk").getContent<int>();
            }

            if (! it->hasAttr("NCE1_LocalLineStride"))
            {
                printf("Serializer Info: Needs Attribute 'NCE1_LocalLineStride'. Defaulting to 1\n");
            }
            else
            {
                localLS = it->getAttr("NCE1_LocalLineStride").getContent<int>();
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
                minLines = it->getAttr("NCE1_MinLines").getContent<dynamic_vector<unsigned>>() ;
            }

            if (! it->hasAttr("stride"))
            {
                printf("Serializer Info: Needs Attribute 'stride'. Defaulting to 1\n");
            }
            else
            {
                stride = it->getAttr("stride").getContent<mv::UnsignedVector2D>().e0;
            }

            if (! it->hasAttr("padding"))
            {
                printf("Serializer Info: Needs Attribute 'padding'. Defaulting to 1\n");
            }
            else
            {
                padEn = it->getAttr("padding").getContent<mv::UnsignedVector4D>().e0;
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
                LPC = it->getAttr("NCE1_LinesPerChannel").getContent<dynamic_vector<unsigned>>();
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
                localCS = it->getAttr("NCE1_LocalChannelStride").getContent<dynamic_vector<unsigned>>();
            }


            int splits_over_iC= 1;
            int i;
            for (unsigned oc = 0; oc != splits_over_oC; oc++)
            {
                for (unsigned ic = 0; ic != splits_over_iC; ic++)
                {
                    for (unsigned h = 0; h != splits_over_H; h++)
                    {

                        i = oc*splits_over_iC*splits_over_H + ic*splits_over_H + h;

                        this->descriptors[i] =  cnnConvolutionPoolStructure();

                        // Relations to other Descriptors
                        if (i+1 == this->desc_count)
                        {
                            this->descriptors[i].Line0.linkAddress = 0; // Last.
                        }else{
                            this->descriptors[i].Line0.linkAddress = 32*4;
                        }
                        // printf("linkAddress: %d\n", 32*4);

                        this->descriptors[i].Line0.id = 0;

                        // Layer Meta Information - Layout & DataTypes
                        this->descriptors[i].Line0.type = NCE1_CONV;
                        this->descriptors[i].Line0.interleavedInput = 0;
                        this->descriptors[i].Line0.interleavedOutput = 0;
                        this->descriptors[i].Line0.cm = NCE1_DTYPE_FP16;
                        this->descriptors[i].Line0.dm = NCE1_DTYPE_FP16;


                        // Standard Fields for Convolution
                        this->descriptors[i].kernelWidth = this->taps->getShape()[2] -1;
                        this->descriptors[i].kernelHeight = this->taps->getShape()[3] -1;

                        this->descriptors[i].chStride = stride -1;  // Stride of Kernel (Square only)

                        if (padEn > 0)
                        {
                            this->descriptors[i].padEn = 1;
                        }
                        else
                        {
                            this->descriptors[i].padEn = 0;
                        }

                        this->descriptors[i].padType = 0;   // Zero Padding

                        this->descriptors[i].inputWidth = this->input->getShape()[0] -1;

                        unsigned int original_height = this->input->getShape()[1];
                        unsigned int current_height;
                        if (splits_over_H > 1){
                            // TODO: Different types of split?
                            if (i+1 == this->desc_count){   // Last Descriptor may be an unequal height to the rest.
                                int surplus = ceil(original_height/(float)splits_over_H)*splits_over_H - original_height;
                                current_height = ceil(original_height/(float)splits_over_H) - surplus;
                            }else{
                                current_height = ceil(original_height/(float)splits_over_H);
                            }
                        }else{
                            current_height = original_height;
                        }

                        this->descriptors[i].inputHeight =  current_height - 1;
                        this->descriptors[i].inputChannels = this->input->getShape()[2] -1;

                        this->descriptors[i].outputChannels = this->output->getShape()[2] -1;

                        // Myriad X DPU Assignment & Execution Configuration
                        this->descriptors[i].Line0.mode = this->DPUmodeVector[i];
                        this->descriptors[i].Line0.it = 0;  // Interrupt Trigger
                        this->descriptors[i].Line0.disInt = 0;  // 0 - Interrupts Enabled, 1 - Interrupts disabled.

                        this->descriptors[i].chPerRamBlock = chPerRamBlock[i] -1;        // Input Channels per Ram Block


                        // Myriad X Compensation Fields
                        this->descriptors[i].topOutputJunk = topJunk;
                        this->descriptors[i].bottomOutputJunk = bottomJunk;

                        this->descriptors[i].localLs =  localLS;
                        this->descriptors[i].localCs =  localCS[i];

                        this->descriptors[i].linesPerCh = LPC[i] -1;

                        this->descriptors[i].rud = 0;   // Re-Use bit

                        this->descriptors[i].minLines = minLines[i] - 1;     // Minimum lines of data required to carry out function

                        this->descriptors[i].coeffLpb = (this->descriptors[i].chPerRamBlock+1) * (this->descriptors[i].kernelWidth+1) * (this->descriptors[i].kernelHeight+1) - 1;
                        this->descriptors[i].css = (this->descriptors[i].kernelWidth + 1) * (this->descriptors[i].kernelHeight + 1) -1 ;
                        this->descriptors[i].outputX = this->output->getShape()[0];

                        // Myriad X - Splitting groups
                        this->descriptors[i].sohGroup = h;
                        this->descriptors[i].sodGroup = 0;

                        // Fused ReLU
                        this->descriptors[i].t0 = 0;
                        this->descriptors[i].a0 = 0;
                        this->descriptors[i].a1 = 0;
                        this->descriptors[i].reluxEn = 0;
                        this->descriptors[i].reluEn = 0;

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
            this->strideX = it->getAttr("stride").getContent<mv::UnsignedVector2D>().e0;
            this->strideY = it->getAttr("stride").getContent<mv::UnsignedVector2D>().e1;
            this->padX = it->getAttr("padding").getContent<mv::UnsignedVector4D>().e0;
            this->padY = it->getAttr("padding").getContent<mv::UnsignedVector4D>().e2;
            this->padStyle = 2; // HARDCODED.
            this->dilation = 1; // HARDCODED.


            printf("Serializer Info: Manual Override of Convolution Software layer order\n");
            this->output->setOrder(Order::RowMajor);
            this->input->setOrder(Order::RowMajor);
            this->taps->setOrder(Order::TBDLayout);
        }
    }
}
